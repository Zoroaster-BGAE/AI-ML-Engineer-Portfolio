"""Typed inference facade.

The estimator is untyped `Any` (scikit-learn ships no stubs). `Predictor` is
the boundary: `Any` goes in at construction, and only `float`/`List[float]`
come out, so `mypy --strict` holds for the rest of the codebase.

Measured characteristics of this artifact (introspected from the pickle):
    RandomForestClassifier, 15 trees, max_depth=5, 35-53 nodes per tree,
    30 features, criterion=entropy, classes_=[0., 1.]

MEASURED, not assumed (python:3.9-slim container, single core, 3,000 samples):

    RandomForest.predict_proba, 1 row        p50 2.517ms   p99 4.866ms
    single DecisionTree.predict_proba        p50 0.114ms
    single tree_.predict (no validation)     p50 0.007ms
    manual tree loop (fast path below)       p50 0.336ms   p99 0.592ms
    predict_proba, batch of 500              p50 2.965ms  (5.9us/row)

The profile is unambiguous: ~660 float comparisons of actual tree traversal
cost 7us, while the stock call costs 2,517us. Attribution, measured directly
rather than inferred:

    joblib Parallel dispatch (15 no-op tasks, n_jobs=1)   0.883ms   35%
    15x tree.predict_proba(check_input=False)             0.719ms   29%
      (of which raw tree_.predict traversal: 0.111ms)
    forest-level check_array (_validate_X_predict, once)  0.069ms    3%
    accumulation lock, np.zeros alloc, delayed() wrappers ~0.85ms   33%

NOTE: input validation is NOT the story. `_accumulate_prediction` calls
`predict(X, check_input=False)`, so `check_array` runs ONCE per prediction at
the forest level, not once per tree. The dominant cost is joblib's `Parallel`
machinery — which at n_jobs=1 buys nothing — followed by the per-tree Python
overhead of 15 `predict_proba` calls (proba slicing, normaliser, division).

THE FAST PATH
-------------
`_predict_proba_fast` reproduces sklearn's arithmetic exactly (per-tree leaf
value, normalise, sum, divide by tree count) while skipping the redundant
re-validation of an input that Pydantic already validated and that we
constructed ourselves as a C-contiguous float32 array.

This is only acceptable because it is *proven*, not assumed:
  * `verify_equivalence()` runs at startup against the stock path over random
    probes and must produce bitwise-identical output (`np.array_equal`);
  * if it ever does not, the service logs an error and silently falls back to
    stock sklearn — correctness wins over latency, always;
  * `FRAUD_API_USE_FAST_PATH=false` disables it in production without a
    rebuild, so the optimisation can be rolled back in one deploy.

Even at the stock 2.5ms, inference is ~1% of the 200ms p99 budget, so the
headline conclusion is unchanged: **the engineering effort belongs in tail
control, not in making the forest faster.** The fast path matters for a
different reason — at 0.34ms the handler can stay inline on the event loop
(see the concurrency note in main.py), whereas 2.5ms of inline blocking would
have forced a threadpool hop.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from core.logging import get_logger
from ml.loader import ModelArtifact
from schemas.response import Decision

logger = get_logger(__name__)


class Predictor:
    """Stateless, thread-safe wrapper around a fitted classifier.

    Thread-safety: `predict_proba` only reads the fitted trees, so concurrent
    calls from the ASGI threadpool are safe. Nothing here mutates the estimator
    after `load_artifact` returns.
    """

    __slots__ = (
        "_estimator",
        "_trees",
        "_n_trees",
        "_positive_index",
        "_version",
        "_review",
        "_block",
        "_fast_path",
    )

    #: Random probes used to prove fast-path equivalence at startup.
    _VERIFY_PROBES = 256

    def __init__(
        self,
        artifact: ModelArtifact,
        *,
        review_threshold: float,
        block_threshold: float,
        use_fast_path: bool = True,
    ) -> None:
        if not 0.0 <= review_threshold <= block_threshold <= 1.0:
            raise ValueError(
                f"Thresholds must satisfy 0 <= review ({review_threshold}) "
                f"<= block ({block_threshold}) <= 1"
            )
        # Bind hot attributes directly: one less attribute lookup per request
        # than going through the dataclass on every call.
        self._estimator = artifact.estimator
        self._trees = tuple(tree.tree_ for tree in artifact.estimator.estimators_)
        self._n_trees = float(len(self._trees))
        self._positive_index = artifact.positive_class_index
        self._version = artifact.version
        self._review = review_threshold
        self._block = block_threshold
        self._fast_path = use_fast_path and self.verify_equivalence()

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def fast_path_enabled(self) -> bool:
        return self._fast_path

    # -- inference ----------------------------------------------------------

    def _predict_proba_fast(self, matrix: NDArray[np.float32]) -> NDArray[np.float64]:
        """Reimplementation of `RandomForestClassifier.predict_proba`.

        Identical arithmetic, identical accumulation order, no re-validation:
        per-tree leaf values -> normalise to a distribution -> sum across trees
        -> divide by the tree count. Proven equal to the stock path by
        `verify_equivalence()` at startup and by the unit suite.
        """
        trees = self._trees
        accumulator: NDArray[np.float64] = trees[0].predict(matrix).astype(np.float64, copy=True)
        normaliser = accumulator.sum(axis=1, keepdims=True)
        normaliser[normaliser == 0.0] = 1.0
        accumulator /= normaliser

        for tree in trees[1:]:
            proba = tree.predict(matrix)
            normaliser = proba.sum(axis=1, keepdims=True)
            normaliser[normaliser == 0.0] = 1.0
            accumulator += proba / normaliser

        accumulator /= self._n_trees
        return accumulator

    def _predict_proba(self, matrix: NDArray[np.float32]) -> NDArray[np.float64]:
        if self._fast_path:
            return self._predict_proba_fast(matrix)
        return self._estimator.predict_proba(matrix)  # type: ignore[no-any-return]

    def verify_equivalence(self, probes: int = _VERIFY_PROBES) -> bool:
        """Prove the fast path is bitwise-identical to stock sklearn.

        Called at construction, before the readiness probe can pass.

        HONEST LIMITS OF THIS CHECK. The probes are Gaussian, not drawn from
        the production distribution, and they do not achieve full coverage:
        measured, the 259 probes reach 192 of the 336 leaves across the 15
        trees (57%). So this is a smoke test, not a proof — a transcription
        error confined to one of the 144 unreached leaves would pass.

        The actual guarantee is structural, not statistical: the fast path
        reads the same `tree_` buffers, applies the same operations, in the
        same order, at the same dtype. IEEE 754 is deterministic, so identical
        inputs through identical operations must produce identical bits. The
        probes exist to catch a *transcription* mistake, and they are the
        weaker half of the argument.

        (Curiously, Gaussian probes cover more leaves than realistic rows do:
        256 production-shaped rows reach only 71 leaves, 21%. Realistic data
        concentrates in the dense region of the tree.)
        """
        rng = np.random.default_rng(seed=1_234)
        sample = np.vstack(
            [
                rng.standard_normal((probes, self._estimator.n_features_in_)),
                np.zeros((1, self._estimator.n_features_in_)),
                np.full((1, self._estimator.n_features_in_), 1e6),
                np.full((1, self._estimator.n_features_in_), -1e6),
            ]
        ).astype(np.float32)

        expected = self._estimator.predict_proba(sample)
        actual = self._predict_proba_fast(sample)
        if np.array_equal(expected, actual):
            return True

        # Correctness beats latency: fall back to stock sklearn, loudly.
        max_delta = float(np.max(np.abs(expected - actual)))
        logger.error(
            "predictor.fast_path.rejected",
            max_abs_delta=max_delta,
            detail="Fast path diverged from sklearn; falling back to predict_proba.",
        )
        return False

    def score_one(self, vector: NDArray[np.float32]) -> tuple[float, float]:
        """Score a single transaction. Returns `(fraud_score, inference_seconds)`.

        `reshape(1, -1)` is a view, not a copy. The caller supplies float32
        (see `schemas.request.MODEL_DTYPE`) because sklearn's tree module casts
        inputs to `float32` internally — handing it the right dtype means no
        converted copy is allocated per request.
        """
        started = time.perf_counter()
        proba = self._predict_proba(vector.reshape(1, -1))
        elapsed = time.perf_counter() - started
        return float(proba[0, self._positive_index]), elapsed

    def score_many(self, matrix: NDArray[np.float32]) -> tuple[list[float], float]:
        """Score an (n_rows, 30) matrix in one vectorised call.

        The fixed per-call overhead amortises across rows — measured at
        5.9us/row for a 500-row batch versus 2,517us for a single-row stock
        call. That is why the batch endpoint exists.
        """
        started = time.perf_counter()
        proba = self._predict_proba(matrix)
        elapsed = time.perf_counter() - started
        return proba[:, self._positive_index].astype(float).tolist(), elapsed

    def decide(self, score: float) -> Decision:
        """Apply the configured decision policy to a score.

        Policy, not modelling: thresholds come from configuration so fraud ops
        can retune the precision/recall trade-off without a redeploy.
        """
        if score >= self._block:
            return Decision.BLOCK
        if score >= self._review:
            return Decision.REVIEW
        return Decision.ALLOW


_predictor: Optional[Predictor] = None


def set_predictor(predictor: Predictor) -> None:
    """Install the process-wide singleton (called from the lifespan hook)."""
    global _predictor
    _predictor = predictor


def get_predictor() -> Predictor:
    """Return the singleton predictor built at startup."""
    if _predictor is None:
        raise RuntimeError("Predictor not initialised; service is not ready.")
    return _predictor


def reset() -> None:
    """Clear the singleton. Test-only."""
    global _predictor
    _predictor = None
