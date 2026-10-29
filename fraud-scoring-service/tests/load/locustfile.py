"""Load profile for the fraud scoring service.

Run against the CONTAINER, never against an in-process TestClient — the point
is to measure the whole path (kernel, ASGI server, event loop, GIL), not the
handler in isolation. Run the generator on a separate machine or container so
you are not measuring locust competing with uvicorn for the same core.

    locust -f tests/load/locustfile.py --headless -u 200 -r 25 -t 120s \
           --host http://localhost:8888

Report p50/p95/p99 against RPS, and state the hardware. A p99 number without
the load level and the machine it was measured on is not a number.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from locust import HttpUser, between, task

BASE_PAYLOAD: dict[str, Any] = json.loads(
    (
        Path(__file__).resolve().parents[1] / "contract" / "fixtures" / "mock-request-body.json"
    ).read_text()
)


def jittered_payload() -> dict[str, Any]:
    """Vary the input so we exercise different tree paths and defeat any
    accidental caching. A load test against one identical row measures the
    branch predictor, not the service."""
    payload = dict(BASE_PAYLOAD)
    for key in ("V1", "V4", "V14", "Amount"):
        payload[key] = payload[key] * random.uniform(0.5, 1.5)
    payload["Time"] = random.uniform(0, 172_000)
    return payload


class SingleScoreUser(HttpUser):
    """The realtime authorisation path — this is the SLO that matters."""

    wait_time = between(0.0, 0.01)

    @task(10)
    def score(self) -> None:
        with self.client.post(
            "/fraud-score", json=jittered_payload(), name="POST /fraud-score", catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"status={response.status_code}")
            elif response.elapsed.total_seconds() > 0.2:
                # Surface SLO breaches as failures so the headline number in
                # the locust report is the thing we actually promised.
                response.failure("exceeded 200ms SLO")

    @task(1)
    def health(self) -> None:
        self.client.get("/health/ready", name="GET /health/ready")


class BatchScoreUser(HttpUser):
    """Offline/replay callers. Proves batching amortises the per-call overhead
    and that large batches are offloaded rather than stalling the event loop."""

    wait_time = between(0.5, 1.0)

    @task
    def score_batch(self) -> None:
        payload = {"transactions": [jittered_payload() for _ in range(200)]}
        self.client.post("/fraud-score/batch", json=payload, name="POST /fraud-score/batch")
