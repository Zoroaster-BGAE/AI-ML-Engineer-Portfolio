"""Model artifact loading: integrity, validation, thread pinning, warm-up.

Everything expensive or dangerous happens exactly once, at process start,
before the readiness probe is allowed to return 200. The request path does
nothing but `predict_proba`.

Startup sequence
----------------
1. Read the artifact and verify its SHA-256 against the build-injected digest.
   Unpickling is arbitrary code execution; this is the control that makes it
   safe. A mismatch aborts startup.
2. Unpickle, translating the notorious cross-version failure into an
   actionable error (see `_UNPICKLE_HINT`).
3. Validate the artifact against the code's assumptions: feature count,
   presence of `predict_proba`, and the position of the positive class.
4. Pin the estimator to a single thread. A 15-tree forest dispatched across a
   joblib pool costs more in dispatch than it saves in traversal, and extra
   BLAS/OpenMP threads on a 1-vCPU Fargate task cause context-switch thrash
   that shows up directly in the p99.
5. Warm up. The first `predict_proba` in a process pays for lazy imports,
   first-touch page faults and allocator growth — routinely 100-300ms. That
   cost is paid here, not by the first customer.
"""

from __future__ import annotations

import hashlib
import hmac
import pickle  # — see integrity verification below
import pickletools
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import sklearn

from core.logging import get_logger
from ml.feature_order import N_FEATURES, assert_matches_artifact

logger = get_logger(__name__)

_UNPICKLE_HINT = (
    "Failed to unpickle the model artifact. The overwhelmingly likely cause is a "
    "scikit-learn version mismatch: this artifact was written by 1.0.2, and sklearn "
    ">=1.3 added the 'missing_go_to_left' field to the Cython Node struct "
    "(56 -> 64 bytes), which makes the tree node array dtype incompatible. "
    "Pin scikit-learn==1.0.2 and numpy<2.0 (see requirements.txt)."
)


class ModelIntegrityError(RuntimeError):
    """Artifact digest did not match the expected value."""


class ModelValidationError(RuntimeError):
    """Artifact loaded but does not satisfy the service's assumptions."""


class ModelNotLoadedError(RuntimeError):
    """A request reached the model before startup completed."""


@dataclass(frozen=True)
class ModelArtifact:
    """An immutable, validated, warm model ready to serve."""

    estimator: Any  # sklearn ships no stubs; `Any` is quarantined to this module
    version: str
    sha256: str
    positive_class_index: int
    n_features: int
    sklearn_version: str
    n_estimators: int
    max_depth: Optional[int]
    load_seconds: float


_artifact: Optional[ModelArtifact] = None
_load_lock = threading.Lock()


def compute_sha256(path: Path) -> str:
    """Stream the artifact and return its hex digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_integrity(path: Path, expected_sha256: Optional[str]) -> str:
    actual = compute_sha256(path)
    if expected_sha256:
        # Constant-time compare: cheap, and the right habit for digest checks.
        if not hmac.compare_digest(actual.lower(), expected_sha256.lower()):
            raise ModelIntegrityError(
                f"Artifact digest mismatch for {path}. "
                f"expected={expected_sha256[:16]}... actual={actual[:16]}... "
                "Refusing to unpickle an unverified artifact."
            )
        logger.info("model.integrity.verified", sha256=actual[:16])
    else:
        # Acceptable locally; in CI/CD the digest is always injected.
        logger.warning(
            "model.integrity.unverified",
            sha256=actual[:16],
            detail="FRAUD_API_MODEL_SHA256 not set; set it in the task definition.",
        )
    return actual


_STRING_OPCODES = frozenset(
    {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "SHORT_BINSTRING", "BINSTRING"}
)


def read_artifact_sklearn_version(path: Path) -> str:
    """Read the training-time scikit-learn version from the pickle STREAM.

    Why not just `estimator._sklearn_version`? Because it is not there after
    loading: `sklearn.base.BaseEstimator.__setstate__` *pops* `_sklearn_version`
    out of the state dict (it uses it to emit its own mismatch warning) before
    the attribute is ever set on the instance. Reading it back off the object
    always yields "unknown".

    So we walk the opcode stream with `pickletools` — which parses but never
    *executes* the pickle — and take the value that follows the
    `_sklearn_version` key. This runs before `pickle.load`, so a version
    mismatch can abort startup without ever deserialising anything.
    """
    try:
        with path.open("rb") as handle:
            key_seen = False
            for opcode, arg, _pos in pickletools.genops(handle):
                if opcode.name in _STRING_OPCODES and isinstance(arg, str):
                    if key_seen:
                        return arg
                    key_seen = arg == "_sklearn_version"
    except Exception as exc:  # — diagnostics only, never fatal here
        logger.warning("model.version.unreadable", error=str(exc))
    return "unknown"


def _unpickle(path: Path) -> Any:
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with path.open("rb") as handle:
                estimator = pickle.load(handle)  # — digest-verified above
        for warning in caught:
            # sklearn raises its own "Trying to unpickle estimator from version X
            # when using version Y" here. Surface it as an error-level log: it is
            # never benign for a model whose output drives payment decisions.
            logger.error("model.unpickle.warning", message=str(warning.message))
        return estimator
    except (ValueError, AttributeError, ModuleNotFoundError, pickle.UnpicklingError) as exc:
        raise ModelValidationError(f"{_UNPICKLE_HINT} Original error: {exc}") from exc


def _resolve_positive_class_index(estimator: Any) -> int:
    """Find the column of `predict_proba` that corresponds to class 'fraud'.

    The artifact's `classes_` is `array([0., 1.])` — FLOATS, not ints. Column 1
    happens to be correct today, but hardcoding `[:, 1]` is a latent defect: a
    retrain with a different label encoding would silently invert the score.
    Resolve it once, assert it, and stamp it on the artifact.
    """
    classes = np.asarray(getattr(estimator, "classes_", []))
    matches = np.flatnonzero(np.isclose(classes.astype(float), 1.0))
    if matches.size != 1:
        raise ModelValidationError(
            f"Expected exactly one positive class (1) in classes_={classes!r}; "
            f"found {matches.size}."
        )
    return int(matches[0])


def _pin_to_single_thread(estimator: Any) -> None:
    """Force serial inference.

    With 15 trees of depth 5 (~44 nodes each), joblib's dispatch overhead is an
    order of magnitude larger than the traversal itself. Parallelism here is a
    pure latency regression, and unbounded worker threads on a 1-vCPU task
    cause scheduler thrash that lands in the tail. The OMP_/OPENBLAS_/MKL_
    env vars in the Dockerfile cover the native side; this covers joblib.
    """
    if hasattr(estimator, "n_jobs"):
        estimator.n_jobs = 1
    if hasattr(estimator, "verbose"):
        estimator.verbose = 0


def _warm_up(estimator: Any, positive_index: int, iterations: int) -> float:
    """Execute real predictions so the first customer request hits a hot path."""
    rng = np.random.default_rng(seed=42)
    sample = rng.standard_normal((1, N_FEATURES)).astype(np.float32)

    started = time.perf_counter()
    for _ in range(max(1, iterations)):
        proba = estimator.predict_proba(sample)
    elapsed = time.perf_counter() - started

    # Prove the output shape/range before declaring readiness.
    score = float(proba[0, positive_index])
    if not 0.0 <= score <= 1.0:
        raise ModelValidationError(f"Warm-up produced an out-of-range score: {score}")
    return elapsed


def load_artifact(
    model_path: str,
    *,
    expected_sha256: Optional[str] = None,
    version: str = "unknown",
    strict_version_check: bool = True,
    warmup_iterations: int = 25,
) -> ModelArtifact:
    """Load, verify, validate, pin and warm the model. Raises on any failure."""
    started = time.perf_counter()
    path = Path(model_path).resolve()
    if not path.is_file():
        raise ModelValidationError(f"Model artifact not found at {path}")

    sha256 = _verify_integrity(path, expected_sha256)

    # Version gate runs BEFORE deserialisation: parse the stream, compare, and
    # refuse to execute the pickle at all if the runtime is wrong.
    artifact_sklearn = read_artifact_sklearn_version(path)
    if artifact_sklearn != sklearn.__version__:
        message = (
            f"Artifact was written by scikit-learn {artifact_sklearn} but the runtime "
            f"has {sklearn.__version__}. Predictions are not guaranteed to be reproducible."
        )
        if strict_version_check:
            raise ModelValidationError(message + " Refusing to start (strict check enabled).")
        logger.error("model.version.mismatch", detail=message)

    estimator = _unpickle(path)
    if not hasattr(estimator, "predict_proba"):
        raise ModelValidationError(
            f"Loaded object {type(estimator)!r} has no predict_proba; not a classifier."
        )

    n_features_in = int(getattr(estimator, "n_features_in_", -1))
    assert_matches_artifact(n_features_in)

    positive_index = _resolve_positive_class_index(estimator)
    _pin_to_single_thread(estimator)
    warmup_seconds = _warm_up(estimator, positive_index, warmup_iterations)

    load_seconds = time.perf_counter() - started
    artifact = ModelArtifact(
        estimator=estimator,
        version=version,
        sha256=sha256,
        positive_class_index=positive_index,
        n_features=n_features_in,
        sklearn_version=artifact_sklearn,
        n_estimators=int(getattr(estimator, "n_estimators", 0)),
        max_depth=getattr(estimator, "max_depth", None),
        load_seconds=load_seconds,
    )

    logger.info(
        "model.loaded",
        version=version,
        sha256=sha256[:16],
        sklearn_version=artifact_sklearn,
        n_features=n_features_in,
        n_estimators=artifact.n_estimators,
        max_depth=artifact.max_depth,
        positive_class_index=positive_index,
        warmup_iterations=warmup_iterations,
        warmup_ms_per_call=round(warmup_seconds * 1000 / max(1, warmup_iterations), 4),
        load_ms=round(load_seconds * 1000, 2),
    )
    return artifact


def set_artifact(artifact: ModelArtifact) -> None:
    """Install the process-wide singleton (called from the lifespan hook)."""
    global _artifact
    with _load_lock:
        _artifact = artifact


def get_artifact() -> ModelArtifact:
    """Return the loaded artifact, or fail loudly.

    This should be unreachable in production: readiness gates on the model
    being present, so the ALB never routes to a task without one.
    """
    if _artifact is None:
        raise ModelNotLoadedError("Model artifact is not loaded; service is not ready.")
    return _artifact


def is_loaded() -> bool:
    """Readiness predicate."""
    return _artifact is not None


def reset() -> None:
    """Clear the singleton. Test-only."""
    global _artifact
    with _load_lock:
        _artifact = None
