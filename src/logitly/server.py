from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Any

from .backends.factory import create_backend
from .engine import DecisionEngine

MAX_REQUEST_BYTES = 1_000_000


def execute_decision(engine: DecisionEngine, payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    kind = payload.get("type")
    state = payload.get("state")
    question = payload.get("question")
    if kind == "noul":
        result = engine.noul(state, question)
    elif kind == "choice":
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and all(isinstance(item, dict) for item in choices):
            try:
                choices = {str(item["id"]): str(item["description"]) for item in choices}
            except KeyError as error:
                raise ValueError("each choice requires id and description") from error
        result = engine.choice(state, question, choices)
    elif kind == "score":
        result = engine.score(state, question, payload.get("levels"))
    else:
        raise ValueError("type must be one of: noul, choice, score")
    return {"type": kind, "result": asdict(result)}


def serve_playground(model: str, host: str = "0.0.0.0", port: int = 8000, *, engine=None) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    print(f"Loading {model} for the direct-logit playground...", flush=True)
    engine = engine if engine is not None else DecisionEngine(create_backend(model))
    inference_lock = threading.Lock()
    html = files("logitly").joinpath("web/index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "Logitly/0.2"

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send_bytes(HTTPStatus.OK, html, "text/html; charset=utf-8")
                return
            if path == "/api/health":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ready",
                        "model": engine.backend.model_name,
                        "revision": engine.backend.spec.revision,
                        "mode": "restricted-next-token-logits",
                        "generated_tokens": 0,
                    },
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/api/decision":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError(f"request body must be between 1 and {MAX_REQUEST_BYTES} bytes")
                payload = json.loads(self.rfile.read(length))
                started = time.perf_counter()
                with inference_lock:
                    response = execute_decision(engine, payload)
                response["elapsed_seconds"] = time.perf_counter() - started
                response["model"] = engine.backend.model_name
                response["generated_tokens"] = 0
                self._send_json(HTTPStatus.OK, response)
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:  # keep the playground alive after an inference failure
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}", flush=True)

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self._send_bytes(status, body, "application/json; charset=utf-8")

        def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Playground ready at http://localhost:{port} using {engine.backend.model_name}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.backend.close()
