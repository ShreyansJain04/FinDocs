"""Simple HTTP endpoint that mimics a text-generation model.

This lightweight server implements the `/generate` API that
`StructureExtractor` expects when running in ``remote`` mode.  It
produces deterministic JSON snippets tailored to the requested
section, allowing the rest of the FinDocs system (extraction,
uncertainty scoring, review queue, Streamlit console) to operate
without an actual LLM deployment.

Usage::

    python -m src.fdocs.services.mock_llm --host 0.0.0.0 --port 8000

The server intentionally keeps dependencies minimal by relying on the
standard library only.  It is not designed for production use but
provides enough fidelity for demos and local development.
"""

from __future__ import annotations

import argparse
import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Tuple


SECTION_FACTORIES = {
    "guidance": "_build_guidance",
    "risk_factors": "_build_risk_factors",
    "financial_metrics": "_build_financial_metrics",
    "management_tone": "_build_management_tone",
    "analyst_concerns": "_build_analyst_concerns",
}


def _clean_text(text: str, max_len: int = 320) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text[:max_len]


def _extract_content(prompt: str) -> Tuple[str, str]:
    """Return (section, content) extracted from the prompt."""

    section_match = re.search(r"Section:\s*(\w+)", prompt, re.IGNORECASE)
    section = section_match.group(1).lower() if section_match else "general"

    content_match = re.search(r"Content:\s*(.*?)\s*JSON Response:", prompt, re.DOTALL | re.IGNORECASE)
    content = content_match.group(1) if content_match else prompt
    return section, _clean_text(content)


def _build_guidance(content: str) -> Dict[str, object]:
    sentences = re.split(r"(?<=[.!?])\s+", content)
    focus = sentences[0] if sentences else content
    return {
        "items": [
            {
                "metric": "outlook",
                "metric_confidence": 0.55,
                "new_value": _clean_text(focus, 200),
                "new_value_confidence": 0.5,
                "confidence": 0.5,
            }
        ]
    }


def _build_risk_factors(content: str) -> Dict[str, object]:
    tokens = [
        sentence
        for sentence in re.split(r"(?<=[.!?])\s+", content)
        if any(word in sentence.lower() for word in ("risk", "headwind", "challenge", "uncertain"))
    ]
    if not tokens:
        tokens = [content]
    return {
        "items": [
            {
                "category": "macro",
                "category_confidence": 0.5,
                "description": _clean_text(tokens[0], 200),
                "description_confidence": 0.55,
                "severity": "medium",
                "severity_confidence": 0.45,
                "confidence": 0.5,
            }
        ]
    }


def _build_financial_metrics(content: str) -> Dict[str, object]:
    pattern = re.compile(r"([A-Za-z][A-Za-z%\s]{2,40})[:\-]?\s*(\$?\d[\d,\.]*\s*(?:million|billion|crore|%|bps)?)", re.IGNORECASE)
    matches = pattern.findall(content)
    if not matches:
        matches = [("metric", content[:48])]
    items = []
    for name, value in matches[:5]:
        items.append(
            {
                "name": _clean_text(name, 60),
                "name_confidence": 0.55,
                "value": _clean_text(value or "n/a", 32),
                "value_confidence": 0.6,
                "confidence": 0.5,
            }
        )
    return {"items": items}


def _build_management_tone(content: str) -> Dict[str, object]:
    text = content.lower()
    positive = sum(token in text for token in ("strong", "growth", "improvement", "positive", "record"))
    negative = sum(token in text for token in ("caution", "pressure", "decline", "risk", "headwind"))
    if positive > negative:
        sentiment, confidence = "positive", 0.6
    elif negative > positive:
        sentiment, confidence = "negative", 0.55
    else:
        sentiment, confidence = "neutral", 0.5
    return {
        "items": [
            {
                "overall_sentiment": sentiment,
                "overall_sentiment_confidence": confidence,
                "confidence": confidence,
            }
        ]
    }


def _build_analyst_concerns(content: str) -> Dict[str, object]:
    sentences = re.split(r"(?<=[.!?])\s+", content)
    focus = sentences[0] if sentences else content
    return {
        "items": [
            {
                "analyst_name": "Analyst",
                "analyst_name_confidence": 0.3,
                "topic": _clean_text(focus, 160),
                "topic_confidence": 0.45,
                "sentiment": "neutral",
                "sentiment_confidence": 0.4,
                "confidence": 0.4,
            }
        ]
    }


def build_response(prompt: str) -> str:
    section, content = _extract_content(prompt)
    factory_name = SECTION_FACTORIES.get(section, "_build_guidance")
    factory = globals()[factory_name]
    payload = factory(content)
    return json.dumps(payload)


class GenerationHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 (name required by BaseHTTPRequestHandler)
        if self.path != "/generate":
            self._write_response(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        content_length = self.headers.get("Content-Length")
        if content_length is None:
            self._write_response(HTTPStatus.LENGTH_REQUIRED, {"error": "Content-Length missing"})
            return

        try:
            length = int(content_length)
        except ValueError:
            self._write_response(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length"})
            return

        raw_body = self.rfile.read(length)
        try:
            payload = json.loads(raw_body or b"{}")
        except json.JSONDecodeError:
            self._write_response(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON"})
            return

        prompt = payload.get("prompt", "")
        response_text = build_response(prompt)
        self._write_response(HTTPStatus.OK, {"text": response_text})

    def log_message(self, format: str, *args) -> None:  # noqa: A003 (format shadowing)
        return  # suppress default logging noise

    def _write_response(self, status: HTTPStatus, data: Dict[str, object]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock LLM text-generation endpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), GenerationHandler)
    print(f"Mock LLM endpoint running on http://{args.host}:{args.port}/generate", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
