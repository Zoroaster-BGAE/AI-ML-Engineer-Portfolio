"""Structured JSON logging.

Two production constraints drive this configuration:

1. **Latency.** Log emission sits on the request path. Rendering JSON and
   writing to stdout at 5k RPS will show up in the p99 tail if it is done
   naively. `structlog`'s pipeline is cheap, we bind context once per request
   via contextvars rather than passing dicts around, and payload-level logging
   is sampled (see `should_sample_payload`).

2. **PII.** V1..V28 are PCA components and Amount is a monetary value. Nothing
   here is a PAN, but feature vectors are treated as sensitive by default: they
   are never logged in full on the happy path, only under sampling or on error.
   Never log cardholder data; consume tokens only.
"""

from __future__ import annotations

import logging
import random
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

#: Fraction of successful requests whose full feature vector is logged.
PAYLOAD_LOG_SAMPLE_RATE = 0.01


def _drop_color_message(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """uvicorn duplicates `event` into `color_message`; drop the noise."""
    event_dict.pop("color_message", None)
    return event_dict


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    """Configure structlog + stdlib logging once, at process start."""
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,  # request_id et al, zero-cost binding
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _drop_color_message,
    ]

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # uvicorn installs its own handlers; drop them so its records flow through
    # our JSON formatter instead of being written twice in two formats.
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True

    # uvicorn.access is muted at the logger, not just at its handlers.
    # `--no-access-log` only clears uvicorn's own handlers; the records still
    # propagate to root and get written by ours. That is a second serialise +
    # write per request on the hot path, duplicating CorrelationMiddleware's
    # structured access log for no benefit.
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.propagate = False


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    logger: structlog.stdlib.BoundLogger = structlog.stdlib.get_logger(name)
    return logger


def should_sample_payload() -> bool:
    """True for the small fraction of requests whose payload we persist.

    Sampled vectors are the raw material for training/serving skew detection:
    ship them to Kinesis Firehose -> S3 and replay them through the offline
    model to prove the online path computes what training computed.
    """
    return random.random() < PAYLOAD_LOG_SAMPLE_RATE  # noqa: S311 — sampling, not crypto


def redact(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip anything that must never reach a log sink."""
    forbidden = {"pan", "card_number", "cvv", "cvc", "expiry", "authorization"}
    return {k: v for k, v in payload.items() if k.lower() not in forbidden}
