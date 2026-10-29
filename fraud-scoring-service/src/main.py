"""FastAPI application: real-time payment fraud scoring.

Runs on port 8888 and exposes `POST /fraud-score` per the service contract,
plus a batch endpoint, split liveness/readiness probes, and `/metrics`.

CONCURRENCY MODEL — the most important design decision in this file
-------------------------------------------------------------------
`/fraud-score` is `async def` and calls `predict_proba` **inline**, blocking the
event loop for the duration.

That looks like the classic FastAPI anti-pattern, so the reasoning matters:

* A sync `def` handler is run by Starlette in an `anyio` worker thread. The hop
  costs ~50-150us of scheduling plus GIL hand-off, and every concurrent request
  occupies a thread from a bounded pool.
* Inference here is a MEASURED 0.336ms p50 / 0.592ms p99 (full benchmark table
  in `ml.predictor`).
* So the threadpool hop costs a meaningful fraction of the work it offloads,
  while adding thread-pool saturation as a new failure mode.

Inline is therefore faster *for this model*. The trade-off is real and bounded:
each request adds ~0.34ms of head-of-line blocking for everything else on the
loop. At 1,000 RPS on one worker that is ~34% loop utilisation — acceptable,
with the event-loop lag gauge watching it.

This decision is contingent on that measurement, and the measurement moved it.
Stock `RandomForestClassifier.predict_proba` costs 2.517ms p50 on this
artifact — 7.5x the fast path, and past the ~1-2ms crossover where inline
blocking begins to dominate the tail. Had the fast path not been provably
identical, this handler would be a sync `def`. If a future model pushes
inference back over ~1ms, change it back; the load test is what tells you.

Defences against event-loop starvation are explicit, not hoped for:
  1. **Bounded batch inline.** Only batches <= `inline_batch_limit` run inline.
     Bigger batches are offloaded to a thread, because a 1,000-row batch is
     tens of milliseconds of uninterruptible loop time and one caller must not
     be able to stall every other in-flight request.
  2. **Bounded offload + load shedding.** Offloads are gated by a semaphore. If
     no slot frees within `offload_acquire_timeout_s`, we return 503 fast
     rather than queueing unboundedly behind an already-saturated CPU. Shedding
     load at a known latency beats timing out at an unknown one.
  3. **Event-loop lag gauge.** Directly measured and exported, so starvation is
     observable before it becomes a latency alert.
  4. **`--limit-concurrency` in uvicorn** caps in-flight requests per worker at
     the process boundary.
  5. **One uvicorn worker per container, many containers.** Horizontal scaling
     replaces in-process parallelism; the GIL is then a per-container concern
     with a small blast radius.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse, PlainTextResponse, Response
from starlette.concurrency import run_in_threadpool

from core.config import Settings, get_settings
from core.logging import configure_logging, get_logger, should_sample_payload
from ml import loader
from ml import predictor as predictor_module
from ml.loader import ModelNotLoadedError, load_artifact
from ml.predictor import Predictor, get_predictor
from observability import metrics
from observability.middleware import CorrelationMiddleware
from schemas.request import Transaction, TransactionBatch
from schemas.response import (
    BatchFraudScoreResponse,
    ErrorResponse,
    FeatureCompleteness,
    FeatureSource,
    FraudScoreResponse,
    HealthResponse,
)

logger = get_logger(__name__)


class BackpressureError(Exception):
    """Raised when the offload pool is saturated; mapped to a fast 503."""


#: Bounds concurrent threadpool offloads. Sized from settings at startup.
_offload_semaphore: asyncio.Semaphore = asyncio.Semaphore(4)

#: Interval for the event-loop lag probe.
_LOOP_PROBE_INTERVAL_S = 0.5


async def _monitor_event_loop_lag(interval: float = _LOOP_PROBE_INTERVAL_S) -> None:
    """Continuously measure how late the loop is running scheduled callbacks.

    `sleep(0.5)` returning at 0.53s means something held the loop for ~30ms.
    This is the leading indicator of the async trap: it moves before p99 does.
    """
    while True:
        started = time.perf_counter()
        await asyncio.sleep(interval)
        lag = max(0.0, time.perf_counter() - started - interval)
        metrics.EVENT_LOOP_LAG.set(lag)
        if lag > 0.050:
            logger.warning("event_loop.lag.high", lag_ms=round(lag * 1000, 2))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown.

    Everything expensive happens here, before readiness flips true:
    load -> verify digest -> validate contract -> pin threads -> warm up.
    If any step fails the exception propagates and the container dies, which is
    exactly right: a task that cannot score must never join the ALB target
    group. Fail the deploy, not the customer's payment.
    """
    settings: Settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)

    global _offload_semaphore
    _offload_semaphore = asyncio.Semaphore(settings.max_concurrent_offload)

    logger.info("startup.begin", environment=settings.environment, port=settings.port)

    artifact = load_artifact(
        settings.model_path,
        expected_sha256=settings.model_sha256,
        version=settings.model_version,
        strict_version_check=settings.strict_sklearn_version_check,
        warmup_iterations=settings.warmup_iterations,
    )
    loader.set_artifact(artifact)
    predictor = Predictor(
        artifact,
        review_threshold=settings.review_threshold,
        block_threshold=settings.block_threshold,
        use_fast_path=settings.use_fast_path,
    )
    predictor_module.set_predictor(predictor)
    logger.info("predictor.ready", fast_path_enabled=predictor.fast_path_enabled)

    metrics.MODEL_INFO.labels(
        model_version=artifact.version,
        sklearn_version=artifact.sklearn_version,
        n_estimators=str(artifact.n_estimators),
        sha256=artifact.sha256[:16],
    ).set(1)

    monitor = asyncio.create_task(_monitor_event_loop_lag())
    logger.info(
        "startup.ready",
        model_version=artifact.version,
        fast_path_enabled=predictor.fast_path_enabled,
    )

    try:
        yield
    finally:
        # Graceful drain: the ALB deregistration delay gives in-flight requests
        # time to finish before the task is removed from the target group.
        monitor.cancel()
        with suppress(asyncio.CancelledError):
            await monitor
        logger.info("shutdown.complete")


app = FastAPI(
    title="Fraud Scoring Service",
    version="1.0.0",
    description="Real-time payment fraud scoring (<200ms p99).",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)
app.add_middleware(CorrelationMiddleware)


# ---------------------------------------------------------------------------
# Error handling — structured, no stack traces on the wire
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> Response:
    """422 with field-level detail, and a metric per offending field.

    A caller who starts sending a feature in different units, or drops one,
    shows up in `fraud_api_validation_failures_total` before anyone notices a
    model-quality regression.
    """
    fields: list[dict[str, Any]] = []
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"][1:]) or "body"
        metrics.VALIDATION_FAILURES_TOTAL.labels(field=field).inc()
        fields.append({"field": field, "message": error["msg"], "type": error["type"]})

    request_id = getattr(request.state, "request_id", None)
    logger.warning("request.invalid", errors=fields)
    body = ErrorResponse(
        title="Invalid request body",
        status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="One or more feature values failed validation.",
        request_id=request_id,
        errors=fields,
    )
    return ORJSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=body.model_dump(by_alias=True)
    )


@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError) -> Response:
    """503 — unreachable in production, since readiness gates on the model."""
    body = ErrorResponse(
        title="Service not ready",
        status=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Model artifact is not loaded.",
        request_id=getattr(request.state, "request_id", None),
    )
    return ORJSONResponse(status_code=503, content=body.model_dump(by_alias=True))


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> Response:
    """500 — the caller gets a request ID; the detail stays in our logs."""
    logger.exception("request.unhandled_error")
    body = ErrorResponse(
        title="Internal server error",
        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An unexpected error occurred. Quote the request-id to support.",
        request_id=getattr(request.state, "request_id", None),
    )
    return ORJSONResponse(status_code=500, content=body.model_dump(by_alias=True))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@app.post(
    "/fraud-score",
    response_model=FraudScoreResponse,
    status_code=status.HTTP_200_OK,
    tags=["scoring"],
    summary="Score a single transaction",
)
async def fraud_score(transaction: Transaction, request: Request) -> FraudScoreResponse:
    """Score one transaction.

    Runs inline on the event loop — see the module docstring for why that is
    the right call for a 15-tree/depth-5 forest, and where the crossover lies.
    """
    started = time.perf_counter()
    predictor = get_predictor()

    vector = transaction.to_vector()
    score, inference_seconds = predictor.score_one(vector)
    decision = predictor.decide(score)

    elapsed = time.perf_counter() - started
    metrics.INFERENCE_LATENCY.labels(mode="single").observe(inference_seconds)
    metrics.FRAUD_SCORE.labels(model_version=predictor.model_version).observe(score)
    metrics.DECISIONS_TOTAL.labels(
        decision=decision.value, model_version=predictor.model_version
    ).inc()
    # Phase 2: completeness/source are driven by the feature-store client.
    metrics.FEATURE_COMPLETENESS_TOTAL.labels(
        completeness=FeatureCompleteness.COMPLETE.value, source=FeatureSource.REQUEST.value
    ).inc()

    if should_sample_payload():
        # 1% sample -> Kinesis Firehose -> S3. The raw material for
        # training/serving skew detection and offline replay.
        logger.info(
            "prediction.sampled",
            fraud_score=score,
            decision=decision.value,
            features=vector.tolist(),
            model_version=predictor.model_version,
        )

    return FraudScoreResponse(
        fraud_score=score,
        decision=decision,
        model_version=predictor.model_version,
        request_id=getattr(request.state, "request_id", ""),
        transaction_id=transaction.transaction_id,
        scored_at=datetime.now(timezone.utc),
        latency_ms=round(elapsed * 1000, 3),
        feature_completeness=FeatureCompleteness.COMPLETE,
        imputed_features=(),
        feature_source=FeatureSource.REQUEST,
    )


@app.post(
    "/fraud-score/batch",
    response_model=BatchFraudScoreResponse,
    status_code=status.HTTP_200_OK,
    tags=["scoring"],
    summary="Score up to 1,000 transactions in one call",
)
async def fraud_score_batch(batch: TransactionBatch, request: Request) -> BatchFraudScoreResponse:
    """Vectorised scoring.

    Small batches run inline; large ones are offloaded to a bounded threadpool
    so one caller's 1,000-row request cannot stall the loop for everyone else.
    When the pool is saturated we shed load with a fast 503 instead of queueing.
    """
    started = time.perf_counter()
    predictor = get_predictor()
    settings = get_settings()

    matrix = batch.to_matrix()
    row_count = matrix.shape[0]

    if row_count <= settings.inline_batch_limit:
        scores, inference_seconds = predictor.score_many(matrix)
    else:
        try:
            await asyncio.wait_for(
                _offload_semaphore.acquire(), timeout=settings.offload_acquire_timeout_s
            )
        except asyncio.TimeoutError:
            metrics.LOAD_SHED_TOTAL.labels(reason="offload_saturated").inc()
            logger.warning("request.shed", rows=row_count)
            raise BackpressureError() from None
        try:
            scores, inference_seconds = await run_in_threadpool(predictor.score_many, matrix)
        finally:
            _offload_semaphore.release()

    elapsed = time.perf_counter() - started
    metrics.INFERENCE_LATENCY.labels(mode="batch").observe(inference_seconds)

    now = datetime.now(timezone.utc)
    request_id = getattr(request.state, "request_id", "")
    results: list[FraudScoreResponse] = []
    for transaction, score in zip(batch.transactions, scores):
        decision = predictor.decide(score)
        metrics.FRAUD_SCORE.labels(model_version=predictor.model_version).observe(score)
        metrics.DECISIONS_TOTAL.labels(
            decision=decision.value, model_version=predictor.model_version
        ).inc()
        results.append(
            FraudScoreResponse(
                fraud_score=score,
                decision=decision,
                model_version=predictor.model_version,
                request_id=request_id,
                transaction_id=transaction.transaction_id,
                scored_at=now,
                latency_ms=round(elapsed * 1000 / row_count, 4),
                feature_completeness=FeatureCompleteness.COMPLETE,
                feature_source=FeatureSource.REQUEST,
            )
        )

    return BatchFraudScoreResponse(
        results=results,
        count=row_count,
        latency_ms=round(elapsed * 1000, 3),
        model_version=predictor.model_version,
        request_id=request_id,
    )


@app.exception_handler(BackpressureError)
async def backpressure_handler(request: Request, exc: BackpressureError) -> Response:
    """503 + Retry-After. Explicit load shedding beats unbounded queueing."""
    body = ErrorResponse(
        title="Service overloaded",
        status=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Inference capacity saturated; retry shortly or reduce batch size.",
        request_id=getattr(request.state, "request_id", None),
    )
    return ORJSONResponse(
        status_code=503, content=body.model_dump(by_alias=True), headers={"Retry-After": "1"}
    )


# ---------------------------------------------------------------------------
# Health & observability
# ---------------------------------------------------------------------------


@app.get("/health/live", response_model=HealthResponse, tags=["health"])
async def liveness() -> HealthResponse:
    """Process is up. Never touches the model — a slow model must not trigger
    a container restart loop."""
    return HealthResponse(status="alive", model_loaded=loader.is_loaded())


@app.get("/health/ready", response_model=HealthResponse, tags=["health"])
async def readiness(response: Response) -> HealthResponse:
    """Ready to serve traffic: model loaded, validated and warm.

    This is the ALB target-group health check. Separating it from liveness is
    what stops a task joining the target group before its first (slow)
    prediction has been paid for.
    """
    if not loader.is_loaded():
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="not-ready", model_loaded=False)

    artifact = loader.get_artifact()
    return HealthResponse(
        status="ready",
        model_loaded=True,
        model_version=artifact.version,
        sklearn_version=artifact.sklearn_version,
    )


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> PlainTextResponse:
    """Prometheus scrape target (ADOT/OTel collector sidecar -> AMP)."""
    return PlainTextResponse(
        content=metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8"
    )
