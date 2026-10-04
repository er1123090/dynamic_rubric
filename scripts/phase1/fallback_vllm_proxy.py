#!/usr/bin/env python3
"""Route a non-streaming vLLM endpoint to a local primary or remote fallback."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from urllib.parse import urlsplit


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class ProxyState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.chat_requests = 0
        self.active_chat_requests = 0
        self.primary_requests = 0
        self.fallback_requests = 0
        self.last_chat_at = 0.0

    def begin_chat(self) -> None:
        with self.lock:
            self.chat_requests += 1
            self.active_chat_requests += 1
            self.last_chat_at = time.time()

    def finish_chat(self) -> None:
        with self.lock:
            self.active_chat_requests -= 1
            self.last_chat_at = time.time()

    def record_route(self, primary: bool) -> None:
        with self.lock:
            if primary:
                self.primary_requests += 1
            else:
                self.fallback_requests += 1

    def snapshot(self, primary_ready: bool) -> dict[str, object]:
        with self.lock:
            return {
                "started_at": self.started_at,
                "chat_requests": self.chat_requests,
                "active_chat_requests": self.active_chat_requests,
                "primary_requests": self.primary_requests,
                "fallback_requests": self.fallback_requests,
                "last_chat_at": self.last_chat_at,
                "primary_ready": primary_ready,
            }


class VLLMFallbackProxy(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    primary: ClassVar[str]
    fallback: ClassVar[str]
    upstream_timeout: ClassVar[float]
    state: ClassVar[ProxyState]
    _health_lock: ClassVar[threading.Lock] = threading.Lock()
    _health_checked_at: ClassVar[float] = 0.0
    _health_value: ClassVar[bool] = False

    def log_message(self, fmt: str, *args: object) -> None:
        print(
            f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] "
            f"{self.client_address[0]} {fmt % args}",
            flush=True,
        )

    @classmethod
    def primary_ready(cls) -> bool:
        now = time.monotonic()
        with cls._health_lock:
            if now - cls._health_checked_at < 1.0:
                return cls._health_value
            cls._health_checked_at = now
            try:
                with urllib.request.urlopen(
                    f"{cls.primary}/v1/models", timeout=0.5
                ) as response:
                    cls._health_value = response.status == 200
            except (OSError, urllib.error.URLError):
                cls._health_value = False
            return cls._health_value

    def _status(self) -> None:
        body = json.dumps(
            self.state.snapshot(self.primary_ready()), sort_keys=True
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _forward(self) -> None:
        is_chat = self.command == "POST" and self.path.startswith(
            "/v1/chat/completions"
        )
        if is_chat:
            self.state.begin_chat()
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else None
            routing_key = body if body is not None else self.path.encode()
            # Stable 50:50 routing keeps retries on the same server and lets
            # Inference A continue sharing work after the Trainer backend is ready.
            primary = self.primary_ready() and hashlib.sha256(routing_key).digest()[0] % 2 == 0
            upstream = self.primary if primary else self.fallback
            self.state.record_route(primary)
            target = urlsplit(upstream)
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in HOP_BY_HOP_HEADERS
                and key.lower() not in {"host", "content-length"}
            }
            connection = http.client.HTTPConnection(
                target.hostname, target.port, timeout=self.upstream_timeout
            )
            try:
                connection.request(self.command, self.path, body=body, headers=headers)
                response = connection.getresponse()
                response_body = response.read()
                self.send_response(response.status, response.reason)
                for key, value in response.getheaders():
                    if (
                        key.lower() not in HOP_BY_HOP_HEADERS
                        and key.lower() != "content-length"
                    ):
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
            finally:
                connection.close()
        except Exception as error:  # The caller owns the retry policy.
            body = json.dumps({"error": f"upstream unavailable: {error}"}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        finally:
            if is_chat:
                self.state.finish_chat()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/_proxy_status":
            self._status()
        else:
            self._forward()

    def do_POST(self) -> None:  # noqa: N802
        self._forward()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28011)
    parser.add_argument("--primary", default="http://127.0.0.1:28013")
    parser.add_argument("--fallback", default="http://127.0.0.1:28111")
    parser.add_argument("--upstream-timeout", type=float, default=620.0)
    args = parser.parse_args()

    VLLMFallbackProxy.primary = args.primary.rstrip("/")
    VLLMFallbackProxy.fallback = args.fallback.rstrip("/")
    VLLMFallbackProxy.upstream_timeout = args.upstream_timeout
    VLLMFallbackProxy.state = ProxyState()
    server = ThreadingHTTPServer((args.host, args.port), VLLMFallbackProxy)
    server.daemon_threads = True
    print(
        f"proxy listening on {args.host}:{args.port}; "
        f"primary={VLLMFallbackProxy.primary} fallback={VLLMFallbackProxy.fallback}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
