"""Uncertainty scoring utilities for active learning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .chunk import Chunk
from .schemas import ExtractedReport


FINANCIAL_KEYWORDS = {
    "guidance",
    "material",
    "significant",
    "forecast",
    "outlook",
    "eps",
    "revenue",
    "margin",
}


@dataclass
class UncertaintyWeights:
    sentiment: float = 0.25
    extraction: float = 0.25
    disagreement: float = 0.25
    anomaly: float = 0.25


@dataclass
class UncertaintyItem:
    document_id: str
    chunk_id: Optional[str]
    field_name: Optional[str]
    score: float
    metadata: Dict[str, float]


def compute_sentiment_uncertainty(chunk: Chunk) -> float:
    probs = [p for p in [chunk.p_positive, chunk.p_neutral, chunk.p_negative] if p is not None]
    if not probs:
        return 0.0
    max_prob = max(probs)
    sorted_probs = sorted(probs, reverse=True)
    top_gap = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else sorted_probs[0]
    base_uncertainty = 1.0 - max_prob
    gap_uncertainty = 1.0 - top_gap
    keyword_boost = 1.0
    if any(keyword in (chunk.text or "").lower() for keyword in FINANCIAL_KEYWORDS):
        keyword_boost = 1.2
    return max(base_uncertainty * 0.6 + gap_uncertainty * 0.4, 0.0) * keyword_boost


def compute_extraction_uncertainty(report: ExtractedReport) -> List[UncertaintyItem]:
    items: List[UncertaintyItem] = []
    for section_name in ["guidance", "risk_factors", "financial_metrics", "management_tone", "analyst_concerns"]:
        section_items = getattr(report, section_name, []) or []
        if not section_items:
            items.append(
                UncertaintyItem(
                    document_id=report.document_id or "unknown",
                    chunk_id=None,
                    field_name=f"{section_name}.missing",
                    score=0.8,
                    metadata={"type": "missing_section"},
                )
            )
            continue

        for idx, entry in enumerate(section_items):
            entry_dict = entry if isinstance(entry, dict) else entry.model_dump()
            for field, value in entry_dict.items():
                if field.endswith("_confidence"):
                    if value is None:
                        continue
                    score = max(0.0, 1.0 - float(value))
                    if score > 0.3:
                        items.append(
                            UncertaintyItem(
                                document_id=report.document_id or "unknown",
                                chunk_id=None,
                                field_name=f"{section_name}.{idx}.{field}",
                                score=score,
                                metadata={"type": "low_confidence"},
                            )
                        )
    return items


def compute_cross_model_disagreement(primary: ExtractedReport, secondary: Optional[ExtractedReport]) -> float:
    if not secondary:
        return 0.0
    disagreements = 0
    total = 0
    for field in ["guidance", "financial_metrics"]:
        primary_items = getattr(primary, field, []) or []
        secondary_items = getattr(secondary, field, []) or []
        compare_len = min(len(primary_items), len(secondary_items))
        for idx in range(compare_len):
            total += 1
            p_item = primary_items[idx] if isinstance(primary_items[idx], dict) else primary_items[idx].model_dump()
            s_item = secondary_items[idx] if isinstance(secondary_items[idx], dict) else secondary_items[idx].model_dump()
            p_val = p_item.get("value")
            s_val = s_item.get("value")
            if p_val and s_val and p_val != s_val:
                disagreements += 1
    if total == 0:
        return 0.0
    return disagreements / total


def compute_anomaly_score(report: ExtractedReport, historical_metrics: Optional[Iterable[Dict[str, float]]] = None) -> float:
    if not historical_metrics:
        return 0.0
    anomalies = 0
    total = 0
    history = list(historical_metrics)
    if not history:
        return 0.0
    for metric in report.financial_metrics:
        metric_dict = metric if isinstance(metric, dict) else metric.model_dump()
        name = metric_dict.get("name")
        value = metric_dict.get("value")
        if name is None or value is None:
            continue
        try:
            current_val = float(value)
        except (TypeError, ValueError):
            continue
        history_vals = [m[name] for m in history if name in m]
        if len(history_vals) < 3:
            continue
        mean_val = sum(history_vals) / len(history_vals)
        variance = sum((hv - mean_val) ** 2 for hv in history_vals) / len(history_vals)
        std_dev = math.sqrt(variance)
        total += 1
        if std_dev > 0 and abs(current_val - mean_val) > 2 * std_dev:
            anomalies += 1
    if total == 0:
        return 0.0
    return anomalies / total


def score_document(
    *,
    document_id: str,
    chunks: Iterable[Chunk],
    report: ExtractedReport,
    historical_metrics: Optional[Iterable[Dict[str, float]]] = None,
    secondary_report: Optional[ExtractedReport] = None,
    weights: Optional[UncertaintyWeights] = None,
) -> List[UncertaintyItem]:
    weights = weights or UncertaintyWeights()
    items: List[UncertaintyItem] = []

    for chunk in chunks:
        uncertainty = compute_sentiment_uncertainty(chunk)
        if uncertainty > 0.3:
            items.append(
                UncertaintyItem(
                    document_id=document_id,
                    chunk_id=chunk.chunk_id,
                    field_name="sentiment",
                    score=uncertainty * weights.sentiment,
                    metadata={"type": "sentiment"},
                )
            )

    extraction_items = compute_extraction_uncertainty(report)
    for item in extraction_items:
        item.score *= weights.extraction
        items.append(item)

    disagreement = compute_cross_model_disagreement(report, secondary_report)
    if disagreement > 0:
        items.append(
            UncertaintyItem(
                document_id=document_id,
                chunk_id=None,
                field_name="extraction.disagreement",
                score=disagreement * weights.disagreement,
                metadata={"type": "disagreement"},
            )
        )

    anomaly = compute_anomaly_score(report, historical_metrics)
    if anomaly > 0:
        items.append(
            UncertaintyItem(
                document_id=document_id,
                chunk_id=None,
                field_name="extraction.anomaly",
                score=anomaly * weights.anomaly,
                metadata={"type": "anomaly"},
            )
        )

    return sorted(items, key=lambda x: x.score, reverse=True)
