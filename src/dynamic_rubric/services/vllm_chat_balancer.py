from __future__ import annotations

import argparse
import json
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ForwardedResponse:
    status_code: int
    body: bytes
    content_type: str
    request_id: str | None
    upstream: str


@dataclass(slots=True)
class ChatBalancerState:
    upstreams: tuple[str, ...]
    served_model: str
    timeout_seconds: float = 600.0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _inflight: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _requests: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _cursor: int = field(default=0, init=False, repr=False)
    _models_body: bytes | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        normalized = tuple(dict.fromkeys(url.rstrip("/") for url in self.upstreams if url))
        if len(normalized) < 2:
            raise ValueError("at least two distinct upstreams are required")
        if not self.served_model:
            raise ValueError("served_model is required")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.upstreams = normalized
        self._inflight = {url: 0 for url in normalized}
        self._requests = {url: 0 for url in normalized}

    @staticmethod
    def _model_identity(payload: Mapping[str, Any], served_model: str) -> dict[str, Any]:
        rows = payload.get("data", [])
        matches = [
            row
            for row in rows
            if isinstance(row, Mapping) and row.get("id") == served_model
        ]
        if len(matches) != 1:
            raise ValueError(
                f"upstream must expose exactly one {served_model!r} model, got {len(matches)}"
            )
        row = matches[0]
        return {
            key: row.get(key)
            for key in ("id", "root", "max_model_len", "owned_by")
        }

    def validate_upstreams(self) -> None:
        canonical_identity: dict[str, Any] | None = None
        canonical_body: bytes | None = None
        for upstream in self.upstreams:
            with urllib.request.urlopen(
                f"{upstream}/v1/models", timeout=min(self.timeout_seconds, 30.0)
            ) as response:
                body = response.read()
            payload = json.loads(body)
            if not isinstance(payload, Mapping):
                raise ValueError(f"invalid models payload from {upstream}")
            identity = self._model_identity(payload, self.served_model)
            if canonical_identity is None:
                canonical_identity = identity
                canonical_body = body
            elif identity != canonical_identity:
                raise ValueError(
                    "judge semantic identity mismatch: "
                    f"expected={canonical_identity}, upstream={upstream}, got={identity}"
                )
        self._models_body = canonical_body

    def models_response(self) -> ForwardedResponse:
        if self._models_body is None:
            raise RuntimeError("upstreams have not been validated")
        return ForwardedResponse(
            status_code=200,
            body=self._models_body,
            content_type="application/json",
            request_id=None,
            upstream="validated-canonical-identity",
        )

    def _ordered_upstreams(self) -> tuple[str, ...]:
        with self._lock:
            start = self._cursor
            self._cursor = (self._cursor + 1) % len(self.upstreams)
            rotated = self.upstreams[start:] + self.upstreams[:start]
            return tuple(sorted(rotated, key=lambda url: self._inflight[url]))

    def _begin(self, upstream: str) -> None:
        with self._lock:
            self._inflight[upstream] += 1
            self._requests[upstream] += 1

    def _finish(self, upstream: str) -> None:
        with self._lock:
            self._inflight[upstream] -= 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "served_model": self.served_model,
                "inflight": dict(self._inflight),
                "requests": dict(self._requests),
            }

    def forward_chat(self, body: bytes) -> ForwardedResponse:
        errors: list[str] = []
        for upstream in self._ordered_upstreams():
            self._begin(upstream)
            request = urllib.request.Request(
                f"{upstream}/v1/chat/completions",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                response = urllib.request.urlopen(request, timeout=self.timeout_seconds)
                with response:
                    return ForwardedResponse(
                        status_code=int(getattr(response, "status", 200)),
                        body=response.read(),
                        content_type=response.headers.get(
                            "content-type", "application/json"
                        ),
                        request_id=response.headers.get("x-request-id"),
                        upstream=upstream,
                    )
            except urllib.error.HTTPError as error:
                error_body = error.read()
                if error.code < 500:
                    return ForwardedResponse(
                        status_code=error.code,
                        body=error_body,
                        content_type=error.headers.get(
                            "content-type", "application/json"
                        ),
                        request_id=error.headers.get("x-request-id"),
                        upstream=upstream,
                    )
                errors.append(f"{upstream}: HTTP {error.code}")
            except (OSError, TimeoutError, urllib.error.URLError) as error:
                errors.append(f"{upstream}: {type(error).__name__}: {error}")
            finally:
                self._finish(upstream)
        raise RuntimeError("all judge upstreams failed: " + "; ".join(errors))


def build_app(state: ChatBalancerState) -> Any:
    from fastapi import FastAPI, Response  # pyright: ignore[reportMissingImports]

    app = FastAPI()

    def to_response(forwarded: ForwardedResponse) -> Response:
        headers = {"x-judge-upstream": forwarded.upstream}
        if forwarded.request_id:
            headers["x-request-id"] = forwarded.request_id
        return Response(
            content=forwarded.body,
            status_code=forwarded.status_code,
            media_type=forwarded.content_type,
            headers=headers,
        )

    @app.get("/health")
    def health() -> Mapping[str, Any]:
        return state.stats()

    @app.get("/v1/models")
    def models() -> Response:
        return to_response(state.models_response())

    @app.post("/v1/chat/completions")
    def chat_completions(payload: dict[str, Any]) -> Response:
        return to_response(
            state.forward_chat(json.dumps(payload, ensure_ascii=False).encode())
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", action="append", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28013)
    args = parser.parse_args()
    state = ChatBalancerState(
        upstreams=tuple(args.upstream),
        served_model=args.served_model,
        timeout_seconds=args.timeout_seconds,
    )
    state.validate_upstreams()
    import uvicorn  # pyright: ignore[reportMissingImports]

    uvicorn.run(build_app(state), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
