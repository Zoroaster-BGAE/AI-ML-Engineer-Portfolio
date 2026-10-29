"""Prometheus metrics.

Two categories, deliberately separated, because they answer different questions
and page different people:

* **Service metrics** (RED) — is the API healthy? Owned by the on-call engineer.
* **ML metrics** — is the *model* healthy? Owned by the ML team. A service can
  be 100% available, sub-100ms, zero errors, and still be quietly broken
  because the score distribution has collapsed or the flag rate has tripled.

Everything is a histogram, never an average. An average latency of 40ms tells
you nothing about a 200ms p99 SLO.

On AWS, the same values are emitted to CloudWatch via EMF (Embedded Metric
Format): metrics are written as structured fields inside the log line that is
already being produced, so custom metrics cost zero additional API calls on the
request path. `PutMetricData` per request would itself blow the latency budget.
"""

from __future__ import annotations

from typing import Final

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY: Final[CollectorRegistry] = CollectorRegistry(auto_describe=True)

# --- Service / RED ---------------------------------------------------------

REQUESTS_TOTAL = Counter(
    "fraud_api_requests_total",
    "Requests by endpoint and status class.",
    labelnames=("endpoint", "method", "status"),
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "fraud_api_request_duration_seconds",
    "End-to-end in-process request latency.",
    labelnames=("endpoint",),
    # Buckets chosen around the 200ms SLO with resolution where we actually
    # live (sub-10ms) and where we start caring (100-200ms).
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0),
    registry=REGISTRY,
)

INFERENCE_LATENCY = Histogram(
    "fraud_api_inference_duration_seconds",
    "predict_proba time only — isolates model cost from framework/network cost.",
    labelnames=("mode",),  # single | batch
    buckets=(0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.05),
    registry=REGISTRY,
)

EVENT_LOOP_LAG = Gauge(
    "fraud_api_event_loop_lag_seconds",
    "Delay between a scheduled wake-up and its execution. The leading indicator "
    "of event-loop starvation under burst load.",
    registry=REGISTRY,
)

LOAD_SHED_TOTAL = Counter(
    "fraud_api_load_shed_total",
    "Requests rejected with 503 because the offload pool was saturated.",
    labelnames=("reason",),
    registry=REGISTRY,
)

# --- ML-specific -----------------------------------------------------------

FRAUD_SCORE = Histogram(
    "fraud_api_score",
    "Distribution of returned fraud scores. The cheapest drift signal available: "
    "it needs no labels and moves before any business metric does. Compare "
    "against a frozen training baseline with PSI.",
    labelnames=("model_version",),
    buckets=(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    registry=REGISTRY,
)

DECISIONS_TOTAL = Counter(
    "fraud_api_decisions_total",
    "Decisions by outcome. The flag rate (block+review / total) is the metric "
    "fraud ops watches; a step change means an attack or a broken pipeline.",
    labelnames=("decision", "model_version"),
    registry=REGISTRY,
)

FEATURE_COMPLETENESS_TOTAL = Counter(
    "fraud_api_feature_completeness_total",
    "Responses by feature completeness. A rising 'partial'/'degraded' rate means "
    "the online store is unhealthy even though the API is still returning 200s.",
    labelnames=("completeness", "source"),
    registry=REGISTRY,
)

VALIDATION_FAILURES_TOTAL = Counter(
    "fraud_api_validation_failures_total",
    "422s by offending field — a caller silently changing units or dropping a "
    "feature shows up here first.",
    labelnames=("field",),
    registry=REGISTRY,
)

MODEL_INFO = Gauge(
    "fraud_api_model_info",
    "Static model metadata, exposed as labels on a constant gauge.",
    labelnames=("model_version", "sklearn_version", "n_estimators", "sha256"),
    registry=REGISTRY,
)


def render() -> bytes:
    """Serialise the registry for the /metrics endpoint."""
    return generate_latest(REGISTRY)
