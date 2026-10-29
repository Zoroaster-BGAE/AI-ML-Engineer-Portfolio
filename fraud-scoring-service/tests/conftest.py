"""Shared fixtures.

The artifact is loaded once per test session — it is the expensive thing, and
loading it per test would make the suite unusable.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ml import loader
from ml import predictor as predictor_module
from ml.loader import ModelArtifact, load_artifact
from ml.predictor import Predictor

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fraud-model.pickle"
FIXTURES = ROOT / "tests" / "contract" / "fixtures"

REVIEW_THRESHOLD = 0.30
BLOCK_THRESHOLD = 0.80

#: Mirrors the shape produced by `Settings.model_version`, so tests assert
#: against the real stamp format rather than a placeholder.
TEST_MODEL_VERSION = "fraud-model@sklearn-1.0.2+test"


@pytest.fixture(scope="session")
def artifact() -> ModelArtifact:
    """The real model, loaded exactly as production loads it."""
    return load_artifact(
        str(MODEL_PATH),
        version=TEST_MODEL_VERSION,
        strict_version_check=True,
        warmup_iterations=3,
    )


@pytest.fixture(scope="session")
def predictor(artifact: ModelArtifact) -> Predictor:
    return Predictor(artifact, review_threshold=REVIEW_THRESHOLD, block_threshold=BLOCK_THRESHOLD)


@pytest.fixture(scope="session")
def mock_request_body() -> dict[str, Any]:
    """The reference payload written by `scripts/train_demo_model.py`."""
    return json.loads((FIXTURES / "mock-request-body.json").read_text(encoding="utf-8"))


@pytest.fixture()
def app_client(artifact: ModelArtifact, predictor: Predictor) -> Iterator[Any]:
    """TestClient over the assembled app with the singletons pre-installed.

    Installing them directly bypasses the lifespan hook, which keeps the
    integration tests fast while still exercising the real routes, middleware
    and exception handlers.
    """
    from fastapi.testclient import TestClient

    from main import app

    loader.set_artifact(artifact)
    predictor_module.set_predictor(predictor)
    # Note: no `with` block — entering the context manager would run the
    # lifespan hook and reload the artifact. The singletons are already
    # installed above, which is what the routes actually depend on.
    yield TestClient(app)
    loader.reset()
    predictor_module.reset()
