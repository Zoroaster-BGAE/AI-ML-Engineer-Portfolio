"""Request correlation and timing — implemented as pure ASGI middleware.

WHY NOT `BaseHTTPMiddleware`
---------------------------
The obvious implementation subclasses Starlette's `BaseHTTPMiddleware`. It was
the first version of this file, and profiling the container rejected it.

`BaseHTTPMiddleware` wraps every request in an `anyio` task group and pipes the
response body through a memory object stream so it can hand you a `Response`
object. That machinery costs a task spawn, a stream, and several extra context
switches per request — on a service whose entire inference cost is 0.46ms, it
was a material share of the request.

Measured on the running container (1 vCPU, load generator in a separate
container on the same Docker network):

                                          BaseHTTPMiddleware   pure ASGI
    GET  /health/live  p50 (framework floor)    1.843ms         1.546ms
    POST /fraud-score  p50                      3.204ms         3.069ms
    POST /fraud-score  serial throughput         269 rps         309 rps
    POST /fraud-score  saturated throughput      275 rps         352 rps
    POST /fraud-score  p99 @ 32 concurrent     266.5ms         177.5ms

+28% saturated throughput and a 33% cut in the p99 under load, for a rewrite
that changes no behaviour. The client-observed figures include the Python
`http.client` generator's own per-request cost, so treat them as an upper
bound; the server-side histogram in `/metrics` is the authoritative view.

A pure ASGI middleware is a callable that takes `(scope, receive, send)` and
wraps `send`. No task group, no stream, no `Request`/`Response` objects. It
does exactly three things:

  1. Establishes a correlation ID (honouring an upstream `X-Request-ID`).
  2. Times the request and stamps response headers.
  3. Binds both into structlog's contextvars so every log line in the
     request's lifetime carries them without being threaded through call args.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import MutableMapping
from typing import Any, Optional

import structlog
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from core.logging import get_logger
from observability.metrics import REQUEST_LATENCY, REQUESTS_TOTAL

logger = get_logger("http")

REQUEST_ID_HEADER = "X-Request-ID"
RESPONSE_TIME_HEADER = "X-Response-Time-Ms"
_REQUEST_ID_KEY = b"x-request-id"


def _upstream_request_id(scope: Scope) -> Optional[str]:
    """Read `X-Request-ID` straight off the raw scope headers.

    Cheaper than constructing a `Request` just to read one header.
    """
    for key, value in scope["headers"]:
        if key == _REQUEST_ID_KEY:
            return str(value.decode("latin-1"))
    return None


class CorrelationMiddleware:
    """Pure-ASGI correlation, timing and access logging."""

    __slots__ = ("app",)

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _upstream_request_id(scope) or uuid.uuid4().hex
        # `request.state.request_id` in the handlers reads this dict.
        state: MutableMapping[str, Any] = scope.setdefault("state", {})
        state["request_id"] = request_id

        path: str = scope["path"]
        method: str = scope["method"]

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id, path=path, method=method)

        started = time.perf_counter()
        observed: dict[str, int] = {"status": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                observed["status"] = message["status"]
                elapsed = time.perf_counter() - started
                headers = MutableHeaders(scope=message)
                headers.append(REQUEST_ID_HEADER, request_id)
                headers.append(RESPONSE_TIME_HEADER, f"{elapsed * 1000:.3f}")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            elapsed = time.perf_counter() - started
            logger.exception("http.request.unhandled", duration_ms=round(elapsed * 1000, 3))
            REQUESTS_TOTAL.labels(path, method, "5xx").inc()
            raise

        elapsed = time.perf_counter() - started
        REQUEST_LATENCY.labels(path).observe(elapsed)
        REQUESTS_TOTAL.labels(path, method, f"{observed['status'] // 100}xx").inc()

        # One structured line per request. No feature values (see core.logging).
        logger.info(
            "http.request",
            status=observed["status"],
            duration_ms=round(elapsed * 1000, 3),
        )
