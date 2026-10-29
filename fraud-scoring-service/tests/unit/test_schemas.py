"""Schema/validation unit tests — the API's edge defences."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from ml.feature_order import FEATURE_ORDER
from schemas.request import Transaction
from schemas.response import Decision, FeatureCompleteness, FraudScoreResponse


def test_all_thirty_features_are_required(mock_request_body: dict[str, Any]) -> None:
    for missing in ("Time", "V14", "Amount"):
        payload = {k: v for k, v in mock_request_body.items() if k != missing}
        with pytest.raises(ValidationError):
            Transaction.model_validate(payload)


def test_unexpected_field_is_rejected(mock_request_body: dict[str, Any]) -> None:
    """extra='forbid': a caller sending a feature we silently drop is a
    correctness incident on a fraud path, not a warning."""
    payload = {**mock_request_body, "V29": 0.1}
    with pytest.raises(ValidationError):
        Transaction.model_validate(payload)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_rejected(
    mock_request_body: dict[str, Any], bad_value: float
) -> None:
    """NaN/Inf would propagate into the tree comparisons and yield a
    well-formed but meaningless score."""
    payload = {**mock_request_body, "Amount": bad_value}
    with pytest.raises(ValidationError):
        Transaction.model_validate(payload)


def test_non_numeric_value_is_rejected(mock_request_body: dict[str, Any]) -> None:
    payload = {**mock_request_body, "V1": "not-a-number"}
    with pytest.raises(ValidationError):
        Transaction.model_validate(payload)


def test_integers_are_coerced_to_float(mock_request_body: dict[str, Any]) -> None:
    payload = {**mock_request_body, "Time": 0}
    assert Transaction.model_validate(payload).Time == 0.0


def test_schema_fields_cover_feature_order_exactly() -> None:
    """The schema and the feature contract cannot drift apart."""
    schema_fields = set(Transaction.model_fields) - {"transaction_id"}
    assert schema_fields == set(FEATURE_ORDER)


def test_response_serialises_the_exact_contract_key() -> None:
    """The brief requires `{'fraud-score': ...}`. A hyphen is not a legal Python
    identifier, so the alias generator is doing real work here."""
    response = FraudScoreResponse(
        fraud_score=0.42,
        decision=Decision.REVIEW,
        model_version="fraud-model@sklearn-1.0.2+abc",
        request_id="req-1",
        scored_at="2026-07-29T12:00:00Z",  # type: ignore[arg-type]
        latency_ms=1.2,
    )
    payload = response.model_dump(by_alias=True)
    assert payload["fraud-score"] == 0.42
    assert payload["model-version"].startswith("fraud-model@")
    assert payload["feature-completeness"] == FeatureCompleteness.COMPLETE
    assert payload["imputed-features"] == ()
    assert "fraud_score" not in payload


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_response_rejects_out_of_range_scores(score: float) -> None:
    with pytest.raises(ValidationError):
        FraudScoreResponse(
            fraud_score=score,
            decision=Decision.ALLOW,
            model_version="v",
            request_id="r",
            scored_at="2026-07-29T12:00:00Z",  # type: ignore[arg-type]
            latency_ms=1.0,
        )
