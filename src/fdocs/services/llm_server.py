"""HTTP inference server that proxies /generate calls to an Ollama model."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional

import requests


LOGGER = logging.getLogger("ollama_llm_server")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


@dataclass
class OllamaConfig:
    base_url: str
    model: str
    temperature: Optional[float]
    num_predict: Optional[int]
    timeout: float

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")


class OllamaGenerator:
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.session = requests.Session()

    def generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload: Dict[str, object] = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
        }

        options: Dict[str, object] = {}
        resolved_temp = self._resolve(temperature, self.config.temperature)
        if resolved_temp is not None:
            options["temperature"] = resolved_temp

        resolved_tokens = self._resolve(max_tokens, self.config.num_predict)
        if resolved_tokens is not None:
            options["num_predict"] = int(resolved_tokens)

        if options:
            payload["options"] = options

        LOGGER.debug("Dispatching prompt to Ollama (model=%s)", self.config.model)
        response = self.session.post(
            f"{self.config.base_url}/api/generate",
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        data = response.json()
        text = data.get("response", "").strip()
        if not text:
            raise RuntimeError("Empty response from Ollama")
        return self._sanitize(text)

    @staticmethod
    def _resolve(value: Optional[float], default: Optional[float]) -> Optional[float]:
        return value if value is not None else default

    @staticmethod
    def _sanitize(text: str) -> str:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return match.group(0)
        safe = " ".join(text.split())
        return json.dumps({"raw_text": safe})


class GenerationHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    generator: Optional[OllamaGenerator] = None

    def do_POST(self) -> None:  # noqa: N802
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

        body = self.rfile.read(length) if length > 0 else b""

        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError:
            self._write_response(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON"})
            return

        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            self._write_response(HTTPStatus.BAD_REQUEST, {"error": "Prompt required"})
            return

        temperature = self._to_float(payload.get("temperature"))
        max_tokens = self._to_int(payload.get("max_tokens") or payload.get("max_new_tokens"))

        try:
            generated = self.generator.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        except requests.RequestException as exc:
            LOGGER.error("Ollama request failed: %s", exc)
            self._write_response(HTTPStatus.BAD_GATEWAY, {"error": "upstream failure", "detail": str(exc)})
            return
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Generation error")
            self._write_response(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "generation failed", "detail": str(exc)})
            return

        self._write_response(HTTPStatus.OK, {"text": generated})

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        LOGGER.debug("%s - %s", self.address_string(), format % args)

    @staticmethod
    def _to_float(value: object) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int(value: object) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _write_response(self, status: HTTPStatus, data: Dict[str, object]) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(
    host: str,
    port: int,
    base_url: str,
    model: str,
    max_new_tokens: Optional[int],
    temperature: Optional[float],
    timeout: float,
) -> None:
    config = OllamaConfig(
        base_url=base_url,
        model=model,
        temperature=temperature,
        num_predict=max_new_tokens,
        timeout=timeout,
    )
    GenerationHandler.generator = OllamaGenerator(config)
    server = ThreadingHTTPServer((host, port), GenerationHandler)
    LOGGER.info(
        "LLM server listening on http://%s:%s/generate (proxy=%s, model=%s)",
        host,
        port,
        base_url,
        model,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        LOGGER.info("Shutting down ...")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ollama-backed text-generation endpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="llama3")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    serve(
        host=args.host,
        port=args.port,
        base_url=args.ollama_base_url,
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
