"""Outbound API schemas (Pydantic v2).

Two things here are load-bearing:

1. `alias_generator=_kebab` produces the exact contract key the brief asks for
   (`fraud-score`), which is not a legal Python identifier. FastAPI serialises
   with `response_model_by_alias=True` by default, so the wire payload is
   kebab-case while the Python attribute stays snake_case.

2. `protected_namespaces=()` — Pydantic v2 reserves the `model_` prefix and
   would otherwise emit a UserWarning for `model_version`. `model_version` is
   the single most useful field on this response (it is what lets you attribute
   a production regression to a model change rather than a code change), so we
   clear the namespace rather than rename the field.

The degraded-response fields (`feature-completeness`, `imputed-features`,
`feature-source`) are present from day one even though this build reads every
feature from the request. When the feature store lands (see docs), a Redis
timeout must be a *visible, machine-readable* degradation on a successful
response — never a 5xx that blocks the payment. Adding the fields later would
be a breaking contract change for every caller.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


def _kebab(field_name: str) -> str:
    """snake_case attribute -> kebab-case wire key."""
    return field_name.replace("_", "-")


class _WireModel(BaseModel):
    """Base for every response body: kebab-case on the wire, immutable in Python."""

    model_config = ConfigDict(
        alias_generator=_kebab,
        populate_by_name=True,
        frozen=True,
        protected_namespaces=(),  # allow `model_version`; see module docstring
    )


class FeatureCompleteness(str, Enum):
    """How much of the feature vector was sourced as intended.

    COMPLETE  — every feature came from its primary source.
    PARTIAL   — some features were imputed (store timeout, cold-start entity);
                the score is still usable but the caller may want to apply a
                more conservative threshold.
    DEGRADED  — enough features were missing that the score should be treated
                as advisory only and the caller should fall back to its rules
                engine.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    DEGRADED = "degraded"


class FeatureSource(str, Enum):
    """Where the feature vector came from — a dimension on every metric."""

    REQUEST = "request"  # precomputed features supplied by the caller
    ONLINE_STORE = "online-store"
    CACHE = "cache"  # in-process TTL cache for slow-moving features
    FALLBACK = "fallback"  # hierarchical defaults; store unavailable


class Decision(str, Enum):
    """Server-side interpretation of the score against configured thresholds.

    The score is the model's output; the decision is *policy*. Thresholds live
    in configuration, not in the image, because fraud operations retune them
    continuously against the precision/recall trade-off.
    """

    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


class FraudScoreResponse(_WireModel):
    """Response for `POST /fraud-score`.

    Wire shape:
        {
          "fraud-score": 0.0731,
          "decision": "allow",
          "model-version": "fraud-model@1.0.2-sha256:1a2b...",
          "request-id": "0f3c...",
          "scored-at": "2026-07-29T12:00:00.000Z",
          "latency-ms": 1.42,
          "feature-completeness": "complete",
          "imputed-features": [],
          "feature-source": "request"
        }
    """

    fraud_score: float = Field(ge=0.0, le=1.0, description="P(fraud) for this transaction")
    decision: Decision
    model_version: str
    request_id: str
    scored_at: datetime
    latency_ms: float = Field(ge=0.0, description="Server-side processing time")

    transaction_id: Optional[str] = None

    # --- degradation contract -------------------------------------------------
    feature_completeness: FeatureCompleteness = FeatureCompleteness.COMPLETE
    imputed_features: tuple[str, ...] = ()
    feature_source: FeatureSource = FeatureSource.REQUEST


class BatchFraudScoreResponse(_WireModel):
    """Response for `POST /fraud-score/batch`."""

    results: list[FraudScoreResponse]
    count: int
    latency_ms: float = Field(ge=0.0)
    model_version: str
    request_id: str


class ErrorResponse(_WireModel):
    """RFC 7807-shaped error body.

    Never leaks a stack trace or an internal path: the `request-id` is the key
    the caller quotes to support, and the detail lives in our structured logs.
    """

    type: str = "about:blank"
    title: str
    status: int
    detail: Optional[str] = None
    request_id: Optional[str] = None
    errors: Optional[list[dict[str, Any]]] = None  # field-level validation failures


class HealthResponse(_WireModel):
    """Liveness / readiness body."""

    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    sklearn_version: Optional[str] = None
