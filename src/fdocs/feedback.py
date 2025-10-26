"""Feedback logging utilities."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


FEEDBACK_ROOT = Path("artifacts/feedback")
SENTIMENT_FILE = FEEDBACK_ROOT / "sentiment_corrections.parquet"
EXTRACTION_FILE = FEEDBACK_ROOT / "extraction_corrections.jsonl"
RAG_FILE = FEEDBACK_ROOT / "rag_corrections.parquet"


def _ensure_root():
    FEEDBACK_ROOT.mkdir(parents=True, exist_ok=True)


def log_sentiment_correction(
    *,
    document_id: str,
    chunk_id: str,
    original_label: str,
    corrected_label: str,
    original_confidence: float,
    text_snippet: str,
    reviewer_id: str,
    correction_reason: Optional[str] = None,
    review_time_seconds: Optional[int] = None,
):
    if corrected_label not in {"positive", "neutral", "negative"}:
        raise ValueError("Invalid sentiment label")
    _ensure_root()
    entry = {
        "document_id": document_id,
        "chunk_id": chunk_id,
        "original_label": original_label,
        "corrected_label": corrected_label,
        "original_confidence": original_confidence,
        "text_snippet": text_snippet,
        "reviewer_id": reviewer_id,
        "timestamp": datetime.utcnow(),
        "correction_reason": correction_reason,
        "review_time_seconds": review_time_seconds,
    }
    df = pd.DataFrame([entry])
    if SENTIMENT_FILE.exists():
        existing = pd.read_parquet(SENTIMENT_FILE)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(SENTIMENT_FILE, index=False)


def log_extraction_correction(entry: Dict):
    _ensure_root()
    entry.setdefault("timestamp", datetime.utcnow().isoformat())
    REQUIRED = {"document_id", "field_path", "original_value", "corrected_value", "reviewer_id"}
    if not REQUIRED.issubset(entry):
        missing = REQUIRED - set(entry)
        raise ValueError(f"Missing fields: {missing}")
    line = json.dumps(entry)
    with EXTRACTION_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_rag_correction(entry: Dict):
    _ensure_root()
    entry.setdefault("timestamp", datetime.utcnow())
    df = pd.DataFrame([entry])
    if RAG_FILE.exists():
        existing = pd.read_parquet(RAG_FILE)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(RAG_FILE, index=False)
