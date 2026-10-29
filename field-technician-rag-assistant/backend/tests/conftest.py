"""Test harness. Needs only the compose Postgres (docker compose up -d postgres); no Ollama, no Claude:
a scripted FakeLLM and a hash embedder stand in for both."""
from __future__ import annotations

import os
from uuid import NAMESPACE_URL, UUID, uuid5

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_session
from app.deps import get_embedder_dep, get_llm
from app.domain.models import Chunk, Client, Document, MachineLine
from app.domain.support import AppUser
from app.llm.embed import HashEmbedder
from app.llm.fake import FakeLLM

BASE_URL = make_url(os.environ.get("TEST_DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/assistant_test"))
EMB = HashEmbedder()


def pytest_configure(config):
    import tempfile

    settings.embedder = "hash"
    settings.media_store = "local"
    settings.media_local_dir = tempfile.mkdtemp(prefix="assistant-test-media-")
    settings.transcriber = "fake"


@pytest.fixture(scope="session")
def engine():
    admin = BASE_URL.set(database="postgres").render_as_string(hide_password=False).replace("+psycopg", "")
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{BASE_URL.database}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{BASE_URL.database}"')
    url = BASE_URL.render_as_string(hide_password=False)
    real, settings.database_url = settings.database_url, url
    try:
        cfg = Config(os.path.join(os.path.dirname(__file__), "..", "alembic.ini"))
        cfg.set_main_option("script_location", os.path.join(os.path.dirname(__file__), "..", "migrations"))
        command.upgrade(cfg, "head")
    finally:
        settings.database_url = real
    eng = create_engine(url)
    yield eng
    eng.dispose()


@pytest.fixture
def db(engine):
    with Session(engine, expire_on_commit=False) as s:
        yield s
        s.rollback()
    with engine.begin() as c:
        c.execute(text("TRUNCATE agent_run, retrieval_gap, chat_session, app_user, chunk, document, machine_line, client, "
                       "llm_batch RESTART IDENTITY CASCADE"))


def add_doc(db: Session, ml: MachineLine, title: str, pages: list[str], *, doc_type="operating_manual",
            component: str | None = None) -> Document:
    d = Document(client_id=ml.client_id, machine_line_id=ml.id, component=component, title=title, doc_type=doc_type,
                 media_type="application/pdf", storage_uri=f"/private/{title}.pdf", sha256=f"{ml.id}-{title}",
                 page_count=len(pages), ingest_status="indexed", languages=["en"], outline=[])
    db.add(d)
    db.flush()
    vecs = EMB.embed(pages)
    for i, (txt, v) in enumerate(zip(pages, vecs, strict=True), start=1):
        db.add(Chunk(document_id=d.id, machine_line_id=ml.id, client_id=ml.client_id, page=i, content=txt,
                     content_type="text", embedding=v))
    db.flush()
    return d


@pytest.fixture
def world(db):
    """Two clients; client 1 has machines A and B with IDENTICAL text, plus a 'line' with components."""
    c1, c2 = Client(slug="c1", name="Client One"), Client(slug="c2", name="Client Two")
    db.add_all([c1, c2])
    db.flush()
    mk = lambda c, slug, label, kind="machine": MachineLine(client_id=c.id, slug=slug, label=label, kind=kind)  # noqa: E731
    a, b, other = mk(c1, "a", "MACHINE A"), mk(c1, "b", "MACHINE B"), mk(c2, "x", "MACHINE X")
    line = mk(c1, "line", "WASH LINE", "line")
    db.add_all([a, b, other, line])
    db.flush()
    same = ["Alarm E4 means the drum motor thermal switch tripped. Isolate power, wait ten minutes, reset the switch.",
            "Lubricate the main bearing every 500 hours with the specified grease."]
    doc_a = add_doc(db, a, "A Manual", same)
    doc_b = add_doc(db, b, "B Manual", same)
    doc_x = add_doc(db, other, "X Manual", same)
    press = add_doc(db, line, "Press Manual", ["Press alarm E7 hydraulic pressure low. Check the pump and oil level."],
                    component="Press")
    dryer = add_doc(db, line, "Dryer Manual", ["Dryer alarm E7 burner flame failure. Check the gas valve and ignition."],
                    component="Dryers")
    db.commit()
    return dict(c1=c1, c2=c2, a=a, b=b, other=other, line=line, doc_a=doc_a, doc_b=doc_b, doc_x=doc_x, press=press,
                dryer=dryer)


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def api(engine, llm, db):
    """TestClient wired to the test DB with FakeLLM + hash embedder. Users: admin / tech."""
    from app.main import create_app

    db.add_all([AppUser(email="admin@t", role="admin"),
                AppUser(email="tech@t", role="technician")])
    db.commit()
    app = create_app()

    def _sess():
        with Session(engine, expire_on_commit=False) as s:
            yield s

    app.dependency_overrides[get_session] = _sess
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_embedder_dep] = lambda: EMB
    client = TestClient(app)  # no context manager: skips the lifespan (which would seed the real DB)
    client.fastapi = app
    return client


# Emails the tests sign in as, and the Cognito group each is treated as being in.
# The `api` fixture creates matching app_user rows, so the role in the token and
# the role on the row agree -- `get_current_user` reconciles them on every call,
# and a mismatch here would silently rewrite the row mid-test.
_TEST_GROUPS = {"admin@t": ["admin"], "tech@t": ["technician"]}


@pytest.fixture(autouse=True)
def _fake_cognito(monkeypatch):
    """Replace the two places that talk to Cognito, and nothing else.

    Only `verify_id_token` is faked, not `get_current_user`. An earlier version of
    this file overrode the dependency itself, which looked simpler and quietly
    removed the `is_active` and `access_expires` checks that live inside it -- so
    the two tests asserting that a disabled or expired account is refused passed
    without exercising anything. Faking one layer down keeps the real code path:
    just-in-time row creation, the role reconciliation, and both of those checks.

    What is deliberately not tested here is signature verification. Minting a
    token a real verifier accepts means either a real login or injecting a key set
    into PyJWT's JWKS client, and both test PyJWT. The audience check, the
    token_use check and the group mapping are verified against the live pool
    instead, which is the only place they can be verified honestly.
    """
    import app.auth.dependencies as deps
    from app.auth.cognito import TokenError

    def fake_verify(token: str) -> dict:
        email = token.removeprefix("test.")
        if email not in _TEST_GROUPS:
            raise TokenError("not a recognised test token")
        return {"sub": f"sub-{email}", "email": email, "cognito:groups": _TEST_GROUPS[email],
                "token_use": "id", "aud": "test-client"}

    monkeypatch.setattr(deps, "verify_id_token", fake_verify)


class _FakeCognitoAdmin:
    """In-memory stand-in for the pool, backed by the rows the test created.

    Username is a synthetic uuid, deliberately NOT the email -- because that is
    what the real pool does when it signs in by email, and a fake that returns the
    email as the username cannot tell the two lookup paths apart. An earlier
    version did exactly that, and the regression test for addressing a user by
    email passed against code that only worked by uuid.

    Enough to prove the admin routes answer, that both identifiers resolve, and
    that the self-lockout guard fires; not a Cognito emulator.
    """

    users: dict[str, dict] = {}

    def __init__(self):
        for email, groups in _TEST_GROUPS.items():
            self.users.setdefault(email, {"username": self._username(email), "email": email,
                                          "enabled": True, "status": "CONFIRMED",
                                          "groups": list(groups)})

    @staticmethod
    def _username(email: str) -> str:
        """A stable stand-in for the opaque uuid Cognito assigns."""
        return f"uuid-{uuid5(NAMESPACE_URL, email)}"

    def list_users(self):
        return list(self.users.values())

    def _by_username(self, username: str) -> dict:
        for u in self.users.values():
            if u["username"] == username:
                return u
        raise KeyError(username)

    def groups_of(self, username):
        try:
            return list(self._by_username(username)["groups"])
        except KeyError:
            return []

    def create_user(self, email, password, role):
        username = self._username(email)
        self.users[email] = {"username": username, "email": email, "enabled": True,
                             "status": "CONFIRMED", "groups": [role]}
        return username

    def set_role(self, username, role):
        self._by_username(username)["groups"] = [role]

    def set_enabled(self, username, enabled):
        self._by_username(username)["enabled"] = enabled

    def set_password(self, username, password):
        self._by_username(username)  # raises if unknown, as the real API would

    def delete_user(self, username):
        u = self._by_username(username)
        self.users.pop(u["email"], None)


@pytest.fixture(autouse=True)
def _fake_cognito_admin(monkeypatch):
    import app.api.admin_users as au

    _FakeCognitoAdmin.users = {}
    monkeypatch.setattr(au, "Admin", _FakeCognitoAdmin)


def login(api: TestClient, who: str) -> dict:
    """Bearer headers for a fake Cognito id token. See `_fake_cognito` for why.

    Still spelled `login()` returning headers, so every call site reads exactly as
    it did when this posted credentials to an endpoint that no longer exists.
    """
    assert who in _TEST_GROUPS, who
    return {"Authorization": f"Bearer test.{who}"}


def uid(x) -> UUID:
    return x.id if hasattr(x, "id") else x
