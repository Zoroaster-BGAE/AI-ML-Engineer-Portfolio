"""12-factor configuration. Every knob is env-driven; nothing is hardcoded."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, sourced from the environment."""

    model_config = SettingsConfigDict(
        env_prefix="FRAUD_API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- Service ---------------------------------------------------------
    app_name: str = "fraud-scoring-service"
    environment: str = "local"
    host: str = "0.0.0.0"  # noqa: S104 — binds inside the container network namespace
    port: int = 8888  # service contract
    log_level: str = "INFO"
    log_json: bool = True

    # --- Model artifact --------------------------------------------------
    model_path: str = "models/fraud-model.pickle"
    #: Optional SHA-256 of the artifact, injected by the build pipeline. When
    #: set, the loader refuses to unpickle anything else. A pickle is arbitrary
    #: code execution on load — integrity is a security control, not hygiene.
    model_sha256: Optional[str] = None
    model_name: str = "fraud-model"
    #: Refuse to start if the running sklearn differs from the one that wrote
    #: the artifact. Better to fail the deploy than to serve silently-wrong
    #: numbers (or crash on the first request).
    strict_sklearn_version_check: bool = True
    #: Warm-up predictions executed during startup, before readiness flips true.
    #: Removes the first-request tax (lazy imports, page-ins, allocator warm-up).
    warmup_iterations: int = 25

    # --- Decision policy (owned by fraud ops, not by this repo) -----------
    review_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    block_threshold: float = Field(default=0.80, ge=0.0, le=1.0)

    #: Bypass sklearn's redundant per-tree input re-validation. Measured at
    #: 0.336ms vs 2.517ms p50 for the stock path, and proven bitwise-identical
    #: at startup by `Predictor.verify_equivalence`. Set false to fall back to
    #: stock `predict_proba` in production without a rebuild.
    use_fast_path: bool = True

    # --- Concurrency -----------------------------------------------------
    #: Batches at or below this size are scored inline on the event loop
    #: (cheaper than a threadpool hop). Larger batches are offloaded so a single
    #: caller cannot starve the loop for every other in-flight request.
    inline_batch_limit: int = 32
    #: Bounds concurrent threadpool offloads => bounds CPU oversubscription.
    max_concurrent_offload: int = 4
    #: How long a request will wait for an offload slot before we shed load.
    offload_acquire_timeout_s: float = 0.050

    # --- Feature store (phase 2; fields wired, client not yet implemented) --
    feature_store_enabled: bool = False
    feature_store_url: Optional[str] = None
    feature_store_timeout_ms: int = 15
    feature_store_cache_ttl_s: int = 300

    @property
    def model_version(self) -> str:
        """Human-readable version stamped onto every response and metric."""
        digest = (self.model_sha256 or "unpinned")[:12]
        return f"{self.model_name}@sklearn-1.0.2+{digest}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (cached: parsed once, not per request)."""
    return Settings()
