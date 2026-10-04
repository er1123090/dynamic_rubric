from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ForwardedResponse:
    status_code: int
    body: bytes
    content_type: str
    request_id: str | None


@dataclass(frozen=True, slots=True)
class PolicyProxyState:
    upstream: str
    model_path: Path
    served_model: str
    model_revision: str
    tokenizer_revision: str
    checkpoint_hash: str
    timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        normalized = self.upstream.rstrip("/")
        if not normalized or not self.served_model or not self.checkpoint_hash:
            raise ValueError("upstream, served_model, and checkpoint_hash are required")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        object.__setattr__(self, "upstream", normalized)

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "served_model": self.served_model,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "thinking": False,
            "checkpoint_hash": self.checkpoint_hash,
            "model_path": str(self.model_path.resolve()),
            "identity_source": "validated-vllm-policy-proxy",
        }

    def validate_upstream(self) -> None:
        if not self.model_path.is_dir():
            raise ValueError(f"policy model path is absent: {self.model_path}")
        with urllib.request.urlopen(
            f"{self.upstream}/v1/models", timeout=self.timeout_seconds
        ) as response:
            payload = json.loads(response.read())
        rows = payload.get("data", []) if isinstance(payload, Mapping) else []
        matches = [
            row
            for row in rows
            if isinstance(row, Mapping) and row.get("id") == self.served_model
        ]
        if len(matches) != 1:
            model_ids = sorted(
                str(row.get("id")) for row in rows if isinstance(row, Mapping)
            )
            raise ValueError(
                f"upstream policy model mismatch: expected={self.served_model}, got={model_ids}"
            )
        upstream_root = matches[0].get("root")
        if not upstream_root or Path(str(upstream_root)).resolve() != self.model_path.resolve():
            raise ValueError(
                "upstream policy path mismatch: "
                f"expected={self.model_path.resolve()}, got={upstream_root}"
            )

    def forward(
        self,
        path: str,
        *,
        method: str = "GET",
        body: bytes | None = None,
    ) -> ForwardedResponse:
        if path not in {"/v1/models", "/v1/chat/completions"}:
            raise ValueError(f"unsupported policy proxy path: {path}")
        request = urllib.request.Request(
            f"{self.upstream}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout_seconds)
        except urllib.error.HTTPError as error:
            return ForwardedResponse(
                status_code=error.code,
                body=error.read(),
                content_type=error.headers.get("content-type", "application/json"),
                request_id=error.headers.get("x-request-id"),
            )
        with response:
            return ForwardedResponse(
                status_code=int(getattr(response, "status", 200)),
                body=response.read(),
                content_type=response.headers.get("content-type", "application/json"),
                request_id=response.headers.get("x-request-id"),
            )


def build_app(state: PolicyProxyState) -> Any:
    from fastapi import FastAPI, Response  # pyright: ignore[reportMissingImports]

    app = FastAPI()

    def to_response(forwarded: ForwardedResponse) -> Response:
        headers = {"x-request-id": forwarded.request_id} if forwarded.request_id else None
        return Response(
            content=forwarded.body,
            status_code=forwarded.status_code,
            media_type=forwarded.content_type,
            headers=headers,
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/dynamic-rubric/identity")
    def identity() -> Mapping[str, Any]:
        return state.identity

    @app.get("/v1/models")
    def models() -> Response:
        return to_response(state.forward("/v1/models"))

    @app.post("/v1/chat/completions")
    def chat_completions(payload: dict[str, Any]) -> Response:
        return to_response(
            state.forward(
                "/v1/chat/completions",
                method="POST",
                body=json.dumps(payload, ensure_ascii=False).encode(),
            )
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    state = PolicyProxyState(
        upstream=args.upstream,
        model_path=args.model_path,
        served_model=args.served_model,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        checkpoint_hash=args.checkpoint_hash,
        timeout_seconds=args.timeout_seconds,
    )
    state.validate_upstream()
    import uvicorn  # pyright: ignore[reportMissingImports]

    uvicorn.run(build_app(state), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
