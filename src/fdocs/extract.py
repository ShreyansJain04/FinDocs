"""Structured extraction via LLMs with schema validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pydantic import ValidationError

from .schemas import (
    AnalystConcern,
    ExtractedReport,
    FinancialMetric,
    GuidanceItem,
    ManagementTone,
    RiskFactor,
)


class ExtractionError(RuntimeError):
    """Raised when extraction fails completely."""


@dataclass
class ExtractionConfig:
    """Configuration for extraction models."""

    primary_endpoint: Optional[str] = None
    fallback_endpoint: Optional[str] = None
    max_retries: int = 2
    output_dir: Path = Path("artifacts/extractions")
    extraction_version: str = "v1"
    mode: str = "remote"  # remote | local


class LLMClient:
    """Flexible client supporting generic /generate and Ollama APIs.

    Usage:
      - HTTP server exposing POST {endpoint}/generate with payload {prompt, temperature, max_tokens}
      - Ollama by setting endpoint to "ollama:<model>", e.g., "ollama:llama3.1"
    """

    def __init__(self, endpoint: str):
        self.endpoint = (endpoint or "").strip()
        self.kind = "http"
        self.ollama_model: Optional[str] = None
        self.ollama_base = "http://localhost:11434"

        # Detect Ollama endpoints
        if self.endpoint.startswith("ollama:"):
            self.kind = "ollama"
            # Format: ollama:<model>
            self.ollama_model = self.endpoint.split(":", 1)[1].strip()
        else:
            # Generic HTTP endpoint expected to expose /generate
            self.endpoint = self.endpoint.rstrip("/")

    def generate(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2048) -> str:
        """Call the configured generation backend and return text."""
        if self.kind == "ollama":
            return self._generate_via_ollama(prompt, temperature=temperature, max_tokens=max_tokens)
        return self._generate_via_http(prompt, temperature=temperature, max_tokens=max_tokens)

    def _generate_via_http(self, prompt: str, temperature: float, max_tokens: int) -> str:
        import requests

        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        response = requests.post(f"{self.endpoint}/generate", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return (
            data.get("text")
            or data.get("response")
            or data.get("choices", [{}])[0].get("text", "")
        )

    def _generate_via_ollama(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call local Ollama server: POST /api/generate with stream=false.

        Expects Ollama running at http://localhost:11434.
        """
        import requests

        if not self.ollama_model:
            raise ValueError("Ollama model not set. Use endpoint format 'ollama:<model>'.")

        url = f"{self.ollama_base}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                # Ollama uses num_predict for max tokens to generate
                "num_predict": int(max_tokens),
            },
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        # Non-streaming returns a single JSON with 'response'
        return data.get("response", "")


SECTION_SCHEMAS: Dict[str, Dict[str, type]] = {
    "guidance": {"items": GuidanceItem},
    "risk_factors": {"items": RiskFactor},
    "financial_metrics": {"items": FinancialMetric},
    "management_tone": {"items": ManagementTone},
    "analyst_concerns": {"items": AnalystConcern},
}


class StructureExtractor:
    """Structured extraction orchestrator."""

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = (config.mode or "remote").lower()
        if self.mode not in {"remote", "local"}:
            raise ValueError(f"Unsupported extraction mode: {self.mode}")

        if self.mode == "remote":
            if not config.primary_endpoint:
                raise ValueError("Remote extraction mode requires primary_endpoint")
            self.primary = LLMClient(config.primary_endpoint)
            self.fallback = (
                LLMClient(config.fallback_endpoint)
                if config.fallback_endpoint
                else None
            )
        else:
            self.primary = None
            self.fallback = None

    def extract(self, *, company: str, document_id: str, document_path: str, sections: Dict[str, List[str]], metadata: Optional[dict] = None) -> ExtractedReport:
        """Run extraction for grouped sections."""

        if self.mode == "remote":
            report = self._extract_remote(
                company=company,
                document_id=document_id,
                document_path=document_path,
                sections=sections,
                metadata=metadata,
            )
        else:
            report = self._extract_local(
                company=company,
                document_id=document_id,
                document_path=document_path,
                sections=sections,
                metadata=metadata,
            )

        self._persist_report(report)
        return report

    # ---------------------------------------------------------------------
    # Remote extraction helpers
    # ---------------------------------------------------------------------

    def _extract_remote(
        self,
        *,
        company: str,
        document_id: str,
        document_path: str,
        sections: Dict[str, List[str]],
        metadata: Optional[dict],
    ) -> ExtractedReport:
        raw_sections: Dict[str, dict] = {}
        aggregated = {
            "company": company,
            "document_id": document_id,
            "document_path": document_path,
            "metadata": metadata or {},
            "extraction_version": self.config.extraction_version,
        }

        for section, texts in sections.items():
            if not texts:
                continue

            prompt = self._build_prompt(section, texts)
            output_text = self._invoke_model(prompt)
            parsed = self._parse_section(section, output_text)
            raw_sections[section] = parsed.get("raw", {})
            aggregated.setdefault(self._section_field(section), []).extend(parsed.get("items", []))

        return ExtractedReport(
            **aggregated,
            raw_sections=raw_sections,
            generated_at=datetime.utcnow(),
        )

    # ---------------------------------------------------------------------
    # Local extraction helpers
    # ---------------------------------------------------------------------

    def _extract_local(
        self,
        *,
        company: str,
        document_id: str,
        document_path: str,
        sections: Dict[str, List[str]],
        metadata: Optional[dict],
    ) -> ExtractedReport:
        combined_sections: Dict[str, str] = {
            section: "\n".join(texts)
            for section, texts in sections.items()
            if texts
        }
        full_text = "\n".join(combined_sections.values())

        guidance_text = self._select_section_text(combined_sections, {"guidance", "outlook"}, full_text)
        risk_text = self._select_section_text(combined_sections, {"risk", "headwind"}, full_text)
        metrics_text = self._select_section_text(combined_sections, {"financial", "metric"}, full_text)
        tone_text = self._select_section_text(combined_sections, {"management", "tone"}, full_text)
        concerns_text = self._select_section_text(combined_sections, {"question", "q&a", "analyst"}, full_text)

        guidance_items = self._local_guidance_items(guidance_text)
        risk_items = self._local_risk_items(risk_text)
        metric_items = self._local_metric_items(metrics_text or full_text)
        tone_items = self._local_tone_items(tone_text or full_text)
        concern_items = self._local_concern_items(concerns_text or full_text)

        report = ExtractedReport(
            company=company,
            document_id=document_id,
            document_path=document_path,
            extraction_version=self.config.extraction_version,
            metadata={"mode": "local", **(metadata or {})},
            raw_sections={section: text for section, text in combined_sections.items()},
            guidance=guidance_items,
            risk_factors=risk_items,
            financial_metrics=metric_items,
            management_tone=tone_items,
            analyst_concerns=concern_items,
            generated_at=datetime.utcnow(),
        )
        return report

    def _select_section_text(self, sections: Dict[str, str], keywords: Iterable[str], fallback: str) -> str:
        for name, text in sections.items():
            section_lower = name.lower()
            if any(keyword in section_lower for keyword in keywords):
                return text
        return fallback

    def _split_sentences(self, text: str) -> List[str]:
        text = text or ""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _local_guidance_items(self, text: str) -> List[GuidanceItem]:
        sentences = self._split_sentences(text)
        keywords = {"guidance", "expect", "forecast", "outlook", "target"}
        items: List[GuidanceItem] = []
        for sentence in sentences:
            lower = sentence.lower()
            if any(keyword in lower for keyword in keywords):
                items.append(
                    GuidanceItem(
                        metric="outlook",
                        metric_confidence=0.55,
                        new_value=sentence[:240],
                        new_value_confidence=0.5,
                        confidence=0.5,
                    )
                )
            if len(items) >= 3:
                break
        if not items and sentences:
            items.append(
                GuidanceItem(
                    metric="summary",
                    metric_confidence=0.4,
                    new_value=sentences[0][:240],
                    new_value_confidence=0.4,
                    confidence=0.35,
                )
            )
        return items

    def _local_risk_items(self, text: str) -> List[RiskFactor]:
        sentences = self._split_sentences(text)
        keywords = {"risk", "headwind", "challenge", "uncertain", "pressure"}
        items: List[RiskFactor] = []
        for sentence in sentences:
            lower = sentence.lower()
            if any(keyword in lower for keyword in keywords):
                items.append(
                    RiskFactor(
                        category="macro",
                        category_confidence=0.6,
                        description=sentence[:240],
                        description_confidence=0.55,
                        severity="medium" if "headwind" in lower else "low",
                        severity_confidence=0.45,
                        confidence=0.5,
                    )
                )
            if len(items) >= 3:
                break
        if not items and sentences:
            items.append(
                RiskFactor(
                    category="general",
                    category_confidence=0.4,
                    description=sentences[0][:240],
                    description_confidence=0.4,
                    severity="unknown",
                    severity_confidence=0.3,
                    confidence=0.35,
                )
            )
        return items

    def _local_metric_items(self, text: str) -> List[FinancialMetric]:
        items: List[FinancialMetric] = []
        pattern = re.compile(r"([A-Za-z][A-Za-z\s%-]{2,40})[:\-]?\s*(\$?\d[\d,\.]*\s*(?:million|billion|crore|lakhs|%)?)", re.IGNORECASE)
        for match in pattern.finditer(text or ""):
            name = match.group(1).strip().strip("-:")
            value = match.group(2).strip()
            items.append(
                FinancialMetric(
                    name=name[:80],
                    name_confidence=0.55,
                    value=value[:40],
                    value_confidence=0.6,
                    confidence=0.5,
                )
            )
            if len(items) >= 5:
                break
        if not items:
            sentences = self._split_sentences(text)
            for sentence in sentences:
                if any(char.isdigit() for char in sentence):
                    items.append(
                        FinancialMetric(
                            name="metric",
                            name_confidence=0.35,
                            value=sentence[:40],
                            value_confidence=0.35,
                            confidence=0.3,
                        )
                    )
                    break
        return items

    def _local_tone_items(self, text: str) -> List[ManagementTone]:
        text_lower = (text or "").lower()
        positive_tokens = {"strong", "positive", "confident", "growth", "improvement", "record"}
        negative_tokens = {"caution", "pressure", "decline", "challenging", "risk", "headwind"}
        pos_count = sum(token in text_lower for token in positive_tokens)
        neg_count = sum(token in text_lower for token in negative_tokens)
        if pos_count > neg_count:
            sentiment = "positive"
            confidence = 0.6
        elif neg_count > pos_count:
            sentiment = "negative"
            confidence = 0.55
        else:
            sentiment = "neutral"
            confidence = 0.5
        return [
            ManagementTone(
                overall_sentiment=sentiment,
                overall_sentiment_confidence=confidence,
                confidence_indicators=[],
                confidence_indicators_confidence=0.4,
                hedging_language=[],
                hedging_language_confidence=0.4,
                confidence=confidence,
            )
        ]

    def _local_concern_items(self, text: str) -> List[AnalystConcern]:
        sentences = self._split_sentences(text)
        items: List[AnalystConcern] = []
        for sentence in sentences:
            if "?" in sentence or sentence.lower().startswith("question"):
                items.append(
                    AnalystConcern(
                        analyst_name="Analyst",
                        analyst_name_confidence=0.3,
                        topic=sentence[:160],
                        topic_confidence=0.45,
                        sentiment="neutral",
                        sentiment_confidence=0.4,
                        confidence=0.4,
                    )
                )
            if len(items) >= 3:
                break
        if not items and sentences:
            items.append(
                AnalystConcern(
                    analyst_name="Analyst",
                    analyst_name_confidence=0.3,
                    topic=sentences[0][:160],
                    topic_confidence=0.3,
                    sentiment="neutral",
                    sentiment_confidence=0.35,
                    confidence=0.35,
                )
            )
        return items

    def _persist_report(self, report: ExtractedReport):
        output_file = self.config.output_dir / f"{report.company}_{report.document_id}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    def _invoke_model(self, prompt: str) -> str:
        if self.mode != "remote":
            raise RuntimeError("_invoke_model is only available in remote mode")

        clients = [self.primary]
        if self.fallback:
            clients.append(self.fallback)

        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries):
            client = clients[min(attempt, len(clients) - 1)]
            try:
                return client.generate(prompt)
            except Exception as exc:
                last_error = exc
        raise ExtractionError(f"LLM extraction failed: {last_error}")

    def _build_prompt(self, section: str, texts: List[str]) -> str:
        examples = self._few_shot_examples(section)
        prompt = [
            "You are an expert financial analyst extracting structured insights as JSON.",
            "Return valid JSON only. Follow the schema exactly.",
            f"Section: {section}",
        ]
        if examples:
            prompt.append("Examples:")
            prompt.extend(examples)
        prompt.append("Content:")
        prompt.extend(texts)
        prompt.append("JSON Response:")
        return "\n\n".join(prompt)

    def _few_shot_examples(self, section: str) -> List[str]:
        return []

    def _parse_section(self, section: str, output_text: str) -> Dict[str, List[dict]]:
        text = output_text.strip()
        if not text:
            return {"items": [], "raw": {}}
        try:
            parsed_json = json.loads(text)
        except json.JSONDecodeError:
            parsed_json = {"raw_text": text}
            return {"items": [], "raw": parsed_json}

        schema_info = SECTION_SCHEMAS.get(section)
        if not schema_info:
            return {"items": [], "raw": parsed_json}

        item_schema = schema_info["items"]
        raw_items = parsed_json if isinstance(parsed_json, list) else parsed_json.get("items", [])
        validated_items = []
        for item in raw_items:
            try:
                validated_items.append(item_schema(**item).model_dump())
            except ValidationError:
                continue
        return {"items": validated_items, "raw": parsed_json}

    def _section_field(self, section: str) -> str:
        mapping = {
            "guidance": "guidance",
            "risk_factors": "risk_factors",
            "financial_metrics": "financial_metrics",
            "management_tone": "management_tone",
            "analyst_concerns": "analyst_concerns",
        }
        return mapping.get(section, section)
