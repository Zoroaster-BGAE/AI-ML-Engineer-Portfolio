"""The local Cognito emulator must never be usable outside a developer machine."""
import pytest

from app.config import Settings

POOL = "eu-central-1_abc"


def _s(**kw) -> Settings:
    return Settings(cognito_user_pool_id=POOL, cognito_client_id="c", **kw)


def test_real_aws_is_the_default():
    s = _s()
    assert s.cognito_jwks_url == f"https://cognito-idp.eu-central-1.amazonaws.com/{POOL}/.well-known/jwks.json"
    s.assert_emulators_are_local()  # nothing to check


def test_emulator_moves_only_the_key_fetch_not_the_issuer():
    s = _s(cognito_endpoint="http://moto:5000/", database_url="postgresql+psycopg://p:p@postgres:5432/assistant")
    assert s.cognito_jwks_url == f"http://moto:5000/{POOL}/.well-known/jwks.json"
    assert s.cognito_issuer.startswith("https://cognito-idp.")   # tokens are still checked against the AWS issuer
    s.assert_emulators_are_local()


def test_emulator_refuses_a_remote_database():
    s = _s(cognito_endpoint="http://moto:5000",
           database_url="postgresql+psycopg://app_rw:x@db.example.internal:5432/assistant")
    with pytest.raises(RuntimeError, match="refusing to start"):
        s.assert_emulators_are_local()


def test_emulator_refuses_to_run_on_ecs(monkeypatch):
    monkeypatch.setenv("ECS_CONTAINER_METADATA_URI_V4", "http://169.254.170.2/v4/x")
    s = _s(cognito_endpoint="http://moto:5000", database_url="postgresql+psycopg://p:p@localhost/assistant")
    with pytest.raises(RuntimeError):
        s.assert_emulators_are_local()
