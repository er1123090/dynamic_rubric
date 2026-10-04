#!/usr/bin/env python3
"""Small failover-aware HTTP proxy for replicated OpenAI-compatible servers."""

from __future__ import annotations

import argparse
import http.client
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class Router:
    def __init__(self, upstreams: list[str], weights: list[int], timeout: float) -> None:
        if len(upstreams) != len(weights):
            raise ValueError("each upstream needs exactly one weight")
        if not upstreams or any(weight <= 0 for weight in weights):
            raise ValueError("upstreams and positive weights are required")

        self.upstreams = tuple(upstream.rstrip("/") for upstream in upstreams)
        self.slots = tuple(
            index for index, weight in enumerate(weights) for _ in range(weight)
        )
        self.timeout = timeout
        self._next_slot = 0
        self._lock = threading.Lock()
        self.requests = [0] * len(self.upstreams)
        self.failures = [0] * len(self.upstreams)

    def route_order(self) -> list[int]:
        with self._lock:
            preferred = self.slots[self._next_slot % len(self.slots)]
            self._next_slot += 1
            order = [preferred]
            order.extend(index for index in range(len(self.upstreams)) if index != preferred)
            return order

    def record_request(self, index: int) -> None:
        with self._lock:
            self.requests[index] += 1

    def record_failure(self, index: int) -> None:
        with self._lock:
            self.failures[index] += 1

    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "upstreams": list(self.upstreams),
                "slots": [self.upstreams[index] for index in self.slots],
                "requests": list(self.requests),
                "failures": list(self.failures),
            }


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    router: Router

    def do_GET(self) -> None:  # noqa: N802
        self._proxy()

    def do_POST(self) -> None:  # noqa: N802
        self._proxy()

    def _proxy(self) -> None:
        if self.path == "/__lb_health":
            payload = json.dumps(self.router.status(), sort_keys=True).encode()
            self._send(200, [("Content-Type", "application/json")], payload)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        errors: list[str] = []

        for index in self.router.route_order():
            upstream = urlsplit(self.router.upstreams[index])
            connection = http.client.HTTPConnection(
                upstream.hostname,
                upstream.port or 80,
                timeout=self.router.timeout,
            )
            try:
                path = f"{upstream.path.rstrip('/')}{self.path}"
                connection.request(self.command, path, body=body, headers=headers)
                response = connection.getresponse()
                payload = response.read()
                response_headers = [
                    (key, value)
                    for key, value in response.getheaders()
                    if key.lower() not in HOP_BY_HOP_HEADERS
                    and key.lower() != "content-length"
                ]
                self.router.record_request(index)
                self._send(response.status, response_headers, payload)
                return
            except (OSError, http.client.HTTPException) as error:
                self.router.record_failure(index)
                errors.append(f"{self.router.upstreams[index]}: {type(error).__name__}: {error}")
            finally:
                connection.close()

        payload = json.dumps({"error": "; ".join(errors)}).encode()
        self._send(502, [("Content-Type", "application/json")], payload)

    def _send(self, status: int, headers: list[tuple[str, str]], payload: bytes) -> None:
        self.send_response(status)
        for key, value in headers:
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def log_message(self, format: str, *args: object) -> None:
        return


class ProxyServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream", action="append", required=True)
    parser.add_argument("--weight", action="append", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=1200.0)
    args = parser.parse_args()

    router = Router(args.upstream, args.weight, args.timeout)
    ProxyHandler.router = router
    server = ProxyServer((args.host, args.port), ProxyHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
