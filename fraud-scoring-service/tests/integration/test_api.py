"""Integration tests against the assembled ASGI app.

Exercises the real routes, middleware, exception handlers and response
serialisation — everything except the network and the container.
"""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.integration


def test_fraud_score_returns_the_contract_shape(
    app_client: Any, mock_request_body: dict[str, Any]
) -> None:
    """The minimum contract: POST /fraud-score -> {'fraud-score': p}."""
    response = app_client.post("/fraud-score", json=mock_request_body)

    assert response.status_code == 200
    body = response.json()
    assert "fraud-score" in body
    assert 0.0 <= body["fraud-score"] <= 1.0
    assert body["decision"] in {"allow", "review", "block"}
    assert body["feature-completeness"] == "complete"
    assert body["imputed-features"] == []
    assert body["model-version"].startswith("fraud-model@")


def test_correlation_id_is_echoed(app_client: Any, mock_request_body: dict[str, Any]) -> None:
    """An upstream request ID must survive end to end — it is the key that
    joins the caller's logs to ours."""
    response = app_client.post(
        "/fraud-score", json=mock_request_body, headers={"X-Request-ID": "upstream-123"}
    )
    assert response.headers["X-Request-ID"] == "upstream-123"
    assert response.json()["request-id"] == "upstream-123"
    assert float(response.headers["X-Response-Time-Ms"]) >= 0


def test_invalid_payload_returns_structured_422(app_client: Any) -> None:
    response = app_client.post("/fraud-score", json={"Time": 0.0})
    assert response.status_code == 422
    body = response.json()
    assert body["status"] == 422
    assert body["request-id"]
    assert len(body["errors"]) >= 29  # 29 missing features
    # No stack traces, no internal paths on the wire.
    assert "Traceback" not in response.text


def test_batch_endpoint_scores_every_row(
    app_client: Any, mock_request_body: dict[str, Any]
) -> None:
    payload = {"transactions": [mock_request_body for _ in range(50)]}
    response = app_client.post("/fraud-score/batch", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 50
    assert len(body["results"]) == 50
    assert len({r["fraud-score"] for r in body["results"]}) == 1


def test_batch_size_limit_is_enforced(app_client: Any, mock_request_body: dict[str, Any]) -> None:
    """Payload bounds are a DoS control, not a nicety."""
    payload = {"transactions": [mock_request_body for _ in range(1001)]}
    assert app_client.post("/fraud-score/batch", json=payload).status_code == 422


def test_health_endpoints(app_client: Any) -> None:
    assert app_client.get("/health/live").status_code == 200

    ready = app_client.get("/health/ready")
    assert ready.status_code == 200
    body = ready.json()
    assert body["status"] == "ready"
    assert body["model-loaded"] is True
    assert body["sklearn-version"] == "1.0.2"


def test_metrics_endpoint_exposes_ml_signals(
    app_client: Any, mock_request_body: dict[str, Any]
) -> None:
    app_client.post("/fraud-score", json=mock_request_body)
    text = app_client.get("/metrics").text

    assert "fraud_api_request_duration_seconds" in text
    assert "fraud_api_inference_duration_seconds" in text
    assert "fraud_api_score" in text  # score distribution -> PSI
    assert "fraud_api_decisions_total" in text  # flag rate
    assert "fraud_api_event_loop_lag_seconds" in text
