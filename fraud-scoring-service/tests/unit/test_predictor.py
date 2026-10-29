"""Predictor unit tests: output contract, decision policy, batch equivalence."""

from __future__ import annotations

import statistics
import time
from typing import Any

import numpy as np
import pytest

from ml.feature_order import N_FEATURES
from ml.loader import ModelArtifact
from ml.predictor import Predictor
from schemas.request import Transaction
from schemas.response import Decision


def test_score_is_a_probability(predictor: Predictor, mock_request_body: dict[str, Any]) -> None:
    score, elapsed = predictor.score_one(Transaction.model_validate(mock_request_body).to_vector())
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert elapsed > 0.0


def test_batch_equals_single(predictor: Predictor, mock_request_body: dict[str, Any]) -> None:
    """Vectorised and single-row paths must agree exactly — otherwise the batch
    endpoint is a second, subtly different model."""
    vector = Transaction.model_validate(mock_request_body).to_vector()
    single, _ = predictor.score_one(vector)
    batch, _ = predictor.score_many(np.tile(vector, (5, 1)))
    assert batch == [single] * 5


def test_thresholds_map_to_decisions(artifact: ModelArtifact) -> None:
    predictor = Predictor(artifact, review_threshold=0.30, block_threshold=0.80)
    assert predictor.decide(0.00) is Decision.ALLOW
    assert predictor.decide(0.2999) is Decision.ALLOW
    assert predictor.decide(0.30) is Decision.REVIEW
    assert predictor.decide(0.7999) is Decision.REVIEW
    assert predictor.decide(0.80) is Decision.BLOCK
    assert predictor.decide(1.00) is Decision.BLOCK


def test_invalid_threshold_configuration_is_rejected(artifact: ModelArtifact) -> None:
    """Fail at startup, not on the first transaction that lands between them."""
    with pytest.raises(ValueError):
        Predictor(artifact, review_threshold=0.9, block_threshold=0.1)


def test_artifact_matches_documented_characteristics(artifact: ModelArtifact) -> None:
    """Guards the assumptions the whole latency argument rests on."""
    assert artifact.n_features == N_FEATURES == 30
    assert artifact.n_estimators == 15
    assert artifact.max_depth == 5
    assert artifact.sklearn_version == "1.0.2"


def test_fast_path_is_enabled_and_bitwise_identical(
    artifact: ModelArtifact, predictor: Predictor
) -> None:
    """The optimisation is only legitimate if it is provably free.

    `_predict_proba_fast` skips sklearn's 16-times-per-prediction input
    re-validation. This test is the licence to do that: identical output, not
    approximately identical.
    """
    assert predictor.fast_path_enabled is True

    rng = np.random.default_rng(seed=7)
    sample = rng.standard_normal((500, N_FEATURES)).astype(np.float32)

    expected = artifact.estimator.predict_proba(sample)
    actual = predictor._predict_proba_fast(sample)  # — testing the internal
    assert np.array_equal(expected, actual)


def test_fast_path_can_be_disabled(
    artifact: ModelArtifact, mock_request_body: dict[str, Any]
) -> None:
    """The kill switch must produce the same score, or it is not a kill switch."""
    vector = Transaction.model_validate(mock_request_body).to_vector()
    fast = Predictor(artifact, review_threshold=0.3, block_threshold=0.8, use_fast_path=True)
    stock = Predictor(artifact, review_threshold=0.3, block_threshold=0.8, use_fast_path=False)

    assert stock.fast_path_enabled is False
    assert fast.score_one(vector)[0] == stock.score_one(vector)[0]


@pytest.mark.perf
def test_inference_latency_is_negligible(
    predictor: Predictor, mock_request_body: dict[str, Any]
) -> None:
    """A smoke test, not a load test.

    Establishes the claim the architecture is built on: inference is a
    rounding error against a 200ms budget. Real p99 numbers come from
    tests/load/ against the running container. The bound is deliberately loose
    (5ms) so this never flakes on a noisy CI runner while still failing loudly
    if inference regresses by an order of magnitude.
    """
    vector = Transaction.model_validate(mock_request_body).to_vector()
    for _ in range(200):  # warm
        predictor.score_one(vector)

    samples = []
    for _ in range(2_000):
        started = time.perf_counter()
        predictor.score_one(vector)
        samples.append((time.perf_counter() - started) * 1000)

    samples.sort()
    p50 = statistics.median(samples)
    p99 = samples[int(len(samples) * 0.99)]
    print(f"\ninference p50={p50:.3f}ms p99={p99:.3f}ms")
    assert p99 < 5.0, f"inference p99 regressed to {p99:.3f}ms"
