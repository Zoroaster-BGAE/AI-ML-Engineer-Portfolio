"""Golden-output regression tests.

WHAT THIS GUARDS
----------------
This is the canary for silent numerical drift. A unit test proves the code does
what the code says; this proves the *whole stack* — interpreter, numpy ABI,
scikit-learn build, feature ordering, positive-class resolution, dtype
handling — still produces the identical score for a known input.

It is the test that catches:

* **Dependency drift.** scikit-learn >= 1.3 added `missing_go_to_left` to the
  Cython `Node` struct, taking it from 56 to 64 bytes. Unpickling a 1.0.2
  artifact under a modern sklearn fails outright with
  `ValueError: node array from the pickle has an incompatible dtype`.
  Reproduced against 1.7.0 before this repo was written; it is the reason
  requirements.txt pins the exact version.
* **NumPy ABI drift.** NumPy 2.0 changed the C ABI. Every extension module in
  the sklearn 1.0.2 wheel was compiled against the 1.x headers.
* **Feature reordering.** `FEATURE_ORDER_SHA256` fails before the score does,
  pointing straight at the cause.
* **Positive-class inversion.** A retrain with a different label encoding would
  flip the score; the recorded `positive_class_index` catches it.
* **Silent numerical changes** from a dtype or preprocessing "cleanup".

Regenerate deliberately, inside the production image:  `make golden`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sklearn

from ml.feature_order import FEATURE_ORDER, FEATURE_ORDER_SHA256, N_FEATURES
from ml.loader import ModelArtifact
from ml.predictor import Predictor
from schemas.request import Transaction

pytestmark = pytest.mark.contract

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN_PATH = FIXTURES / "golden.json"

#: Exact-equality tolerance. Not "close enough" — bitwise-stable inference on a
#: fixed toolchain is the property under test. Anything looser would let a real
#: numerical regression through.
TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"Golden fixture missing at {GOLDEN_PATH}.\n"
            "Generate it once inside the production image:  make golden\n"
            "It must never be generated from an ad-hoc local environment."
        )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Environment guards — these fail *before* the score assertion, so a dependency
# bump produces a diagnosis rather than a mystery.
# ---------------------------------------------------------------------------


def test_sklearn_version_matches_golden(golden: dict[str, Any]) -> None:
    assert sklearn.__version__ == golden["sklearn_version"], (
        f"scikit-learn is {sklearn.__version__}, golden recorded "
        f"{golden['sklearn_version']}. Tree pickles are not portable across "
        "versions; predictions are not guaranteed to be reproducible."
    )


def test_numpy_major_version_matches_golden(golden: dict[str, Any]) -> None:
    running = np.__version__.split(".")[0]
    recorded = golden["numpy_version"].split(".")[0]
    assert running == recorded, (
        f"NumPy major version drifted ({np.__version__} vs {golden['numpy_version']}). "
        "NumPy 2.x is an ABI break for extension modules compiled against 1.x."
    )


def test_feature_order_contract_unchanged(golden: dict[str, Any]) -> None:
    assert golden["feature_order_sha256"] == FEATURE_ORDER_SHA256, (
        "FEATURE_ORDER changed. Every score produced since the change is suspect. "
        "If the reorder is intentional, regenerate the golden fixture and say so "
        "in the commit message."
    )


def test_artifact_digest_unchanged(artifact: ModelArtifact, golden: dict[str, Any]) -> None:
    assert artifact.sha256 == golden["model_sha256"], (
        "The model binary changed. A new model is a new golden fixture and a new "
        "model-version stamp."
    )


def test_positive_class_index_unchanged(artifact: ModelArtifact, golden: dict[str, Any]) -> None:
    """`classes_` is `[0., 1.]` (floats). Hardcoding column 1 would be a latent
    defect; this proves the resolved index is still the fraud column."""
    assert artifact.positive_class_index == golden["positive_class_index"]


# ---------------------------------------------------------------------------
# The golden assertion
# ---------------------------------------------------------------------------


def test_golden_scores_match_exactly(predictor: Predictor, golden: dict[str, Any]) -> None:
    for case in golden["cases"]:
        payload = json.loads((FIXTURES / case["input_file"]).read_text(encoding="utf-8"))
        score, _ = predictor.score_one(Transaction.model_validate(payload).to_vector())

        assert score == pytest.approx(case["fraud_score"], abs=TOLERANCE), (
            f"{case['name']}: fraud-score drifted.\n"
            f"  expected {case['fraud_score']!r}\n"
            f"  actual   {score!r}\n"
            "The model binary, the dependency set, or the feature pipeline changed."
        )
        assert predictor.decide(score).value == case["decision"]


def test_scoring_is_deterministic_across_calls(
    predictor: Predictor, mock_request_body: dict[str, Any]
) -> None:
    """No RNG, no state: 50 calls, one value."""
    vector = Transaction.model_validate(mock_request_body).to_vector()
    scores = {predictor.score_one(vector)[0] for _ in range(50)}
    assert len(scores) == 1


# ---------------------------------------------------------------------------
# Feature-ordering guards — the highest-severity silent defect in the service
# ---------------------------------------------------------------------------


def test_json_key_order_does_not_change_the_score(
    predictor: Predictor, mock_request_body: dict[str, Any]
) -> None:
    """A caller reordering their JSON keys must be a no-op.

    Proves the vector is built from FEATURE_ORDER, not from payload insertion
    order.
    """
    baseline, _ = predictor.score_one(Transaction.model_validate(mock_request_body).to_vector())
    reversed_payload = dict(reversed(list(mock_request_body.items())))
    shuffled, _ = predictor.score_one(Transaction.model_validate(reversed_payload).to_vector())
    assert shuffled == baseline


def test_permuted_values_do_change_the_score(
    predictor: Predictor, mock_request_body: dict[str, Any]
) -> None:
    """The inverse guard: same keys, permuted VALUES, must score differently.

    Without this, a `to_vector()` that ignored its input entirely (returning
    zeros, say) would pass every other test in this file.
    """
    baseline, _ = predictor.score_one(Transaction.model_validate(mock_request_body).to_vector())
    keys = list(FEATURE_ORDER)
    values = [mock_request_body[k] for k in keys]
    permuted = dict(zip(keys, list(reversed(values))))
    permuted_score, _ = predictor.score_one(Transaction.model_validate(permuted).to_vector())
    assert permuted_score != baseline


def test_vector_shape_and_dtype(mock_request_body: dict[str, Any]) -> None:
    """float32 is deliberate: sklearn's tree module casts to float32 internally,
    so supplying it avoids a per-request copy inside `check_array`."""
    vector = Transaction.model_validate(mock_request_body).to_vector()
    assert vector.shape == (N_FEATURES,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()
