"""Auth, roles, scope validation, session immutability and the no-exposure contract."""
import json
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from app.agent.schemas import Clarify, Grade, Slots, Understanding
from app.domain.support import AppUser
from tests.conftest import login

pytestmark = pytest.mark.db


def _create_session(api, h, world, machine="a", client="c1"):
    return api.post("/api/sessions", headers=h,
                    json={"client_id": str(world[client].id), "machine_line_id": str(world[machine].id)})


# ---- FR-01 / FR-02 -------------------------------------------------------------------------------------
def _admin_operations(api):
    """(method, concrete path) for every /api/admin operation, read from the OpenAPI spec."""
    import re
    ops = []
    for path, item in api.fastapi.openapi()["paths"].items():
        if path.startswith("/api/admin"):
            concrete = re.sub(r"\{[^}]+\}", lambda _: str(uuid.uuid4()), path)
            ops += [(m.upper(), concrete) for m in item if m in {"get", "post", "put", "patch", "delete"}]
    return ops


def test_technician_gets_403_on_every_admin_route(api, world):
    h = login(api, "tech@t")
    ops = _admin_operations(api)
    assert len(ops) >= 10
    for method, path in ops:
        resp = api.request(method, path, headers=h, json={})
        assert resp.status_code == 403, f"{method} {path} -> {resp.status_code}"


def test_admin_routes_require_authentication(api):
    assert api.get("/api/admin/users").status_code == 401


def test_admin_can_read_the_admin_routes(api, world):
    h = login(api, "admin@t")
    for p in ("/api/admin/users", "/api/admin/library", "/api/admin/traces", "/api/admin/gaps"):
        assert api.get(p, headers=h).status_code == 200, p


def test_disabled_user_is_rejected_on_the_next_request_not_at_token_expiry(api, db):
    h = login(api, "tech@t")
    assert api.get("/api/auth/me", headers=h).status_code == 200
    db.execute(text("update app_user set is_active=false where email='tech@t'"))
    db.commit()
    assert api.get("/api/auth/me", headers=h).status_code == 401


def test_expired_access_is_rejected(api, db):
    h = login(api, "tech@t")
    u = db.query(AppUser).filter_by(email="tech@t").one()
    u.access_expires = datetime.now(UTC) - timedelta(minutes=1)
    db.commit()
    assert api.get("/api/auth/me", headers=h).status_code == 403


def test_admin_routes_address_users_by_email_or_by_pool_username(api, db):
    """Both identifiers must work, because only one of them is guessable.

    This pool signs in by email, so Cognito assigns an opaque uuid as Username and
    every Admin* API takes that uuid -- while an administrator, and the console's
    own listing, work in email addresses. Keying the routes on the uuid alone
    returned 404 for every request a person would actually write, which is how
    this was found: a PATCH by email against the deployed service.
    """
    h = login(api, "admin@t")
    r = api.post("/api/admin/users", headers=h,
                 json={"email": "byemail@t", "password": "temporary1", "role": "technician"})
    assert r.status_code == 201
    username = r.json()["username"]

    assert api.patch("/api/admin/users/byemail@t", headers=h, json={"role": "customer"}).status_code == 200
    assert api.patch(f"/api/admin/users/{username}", headers=h, json={"is_active": False}).status_code == 200
    # Case is not the administrator's problem either.
    assert api.patch("/api/admin/users/ByEmail@T", headers=h, json={"is_active": True}).status_code == 200
    assert api.patch("/api/admin/users/nobody@t", headers=h, json={"is_active": False}).status_code == 404


def test_admin_cannot_disable_their_own_account(api, db):
    # Addressed by Cognito username, not the local row id: user administration is
    # applied to the pool now, and the pool is what the id in the URL refers to.
    h = login(api, "admin@t")
    assert api.patch("/api/admin/users/admin@t", headers=h, json={"is_active": False}).status_code == 400


def test_creating_a_user_provisions_them_in_the_pool_and_not_locally(api, db):
    """Creating a user writes to Cognito only; the app_user row appears at first sign-in.

    This replaces a forced-password-change test. That flow belonged to an app that
    stored passwords, and this one does not -- resetting a password is Cognito's
    job, through an administrator or its own forgot-password flow.
    """
    h = login(api, "admin@t")
    r = api.post("/api/admin/users", headers=h,
                 json={"email": "New@T", "password": "temporary1", "role": "technician"})
    assert r.status_code == 201
    body = r.json()
    assert body["email"] == "new@t"
    assert body["role"] == "technician"
    # No local row yet: nothing about this user is known here until they call the API.
    assert body["has_local_row"] is False
    assert db.query(AppUser).filter_by(email="new@t").one_or_none() is None

    # ...and it lists from the pool, so the new account is visible immediately.
    assert "new@t" in [u["email"] for u in api.get("/api/admin/users", headers=h).json()]




# ---- FR-03 / FR-04 / FR-05 -----------------------------------------------------------------------------
def test_catalog_returns_labels_and_counts_only(api, world):
    h = login(api, "tech@t")
    clients = api.get("/api/clients", headers=h).json()
    assert {c["name"] for c in clients} == {"Client One", "Client Two"}
    ms = api.get(f"/api/clients/{world['c1'].id}/machines", headers=h).json()
    line = next(m for m in ms if m["label"] == "WASH LINE")
    assert line["kind"] == "line" and line["document_count"] == 2 and line["component_count"] == 2
    assert set(line) == {"id", "label", "kind", "year", "document_count", "component_count"}


def test_mismatched_client_machine_pair_is_422(api, world):
    h = login(api, "tech@t")
    assert _create_session(api, h, world, machine="a", client="c2").status_code == 422
    assert _create_session(api, h, world).status_code == 201


def test_no_api_can_change_the_scope_of_a_session(api, world):
    """FR-04: scope is written once at creation and no request can move it.

    This used to assert that PATCH, PUT and DELETE all 405. PATCH now exists, for
    pinning, so asserting the method is absent no longer tests the rule -- it
    tests that the rule happens to be enforced by there being no handler. The
    rule is checked directly instead: send a scope change through the method that
    does exist, and confirm the scope did not move.
    """
    h = login(api, "tech@t")
    created = _create_session(api, h, world).json()
    sid, scope_before = created["id"], created["machine_label"]
    other = str(world["b"].id)

    # Still no handler at all for these two.
    for method in ("PUT", "DELETE"):
        assert api.request(method, f"/api/sessions/{sid}", headers=h,
                           json={"machine_line_id": other}).status_code == 405

    # PATCH without the one field it accepts is a validation error, not a
    # silent no-op that returns 200 and looks like it worked.
    assert api.patch(f"/api/sessions/{sid}", headers=h, json={"machine_line_id": other}).status_code == 422

    # And a valid PATCH carrying a scope change ignores it.
    r = api.patch(f"/api/sessions/{sid}", headers=h, json={"pinned": True, "machine_line_id": other})
    assert r.status_code == 200
    assert r.json()["pinned"] is True
    assert r.json()["machine_label"] == scope_before

    assert api.get(f"/api/sessions/{sid}", headers=h).json()["machine_label"] == scope_before


def test_creating_chats_deletes_nothing_and_older_chats_page_in(api, world, db):
    """Creating a chat used to prune the oldest ones past history_max_sessions, which destroyed rated
    conversations before review. Retention is now the nightly job's; the sidebar pages instead."""
    from app.config import settings

    h = login(api, "tech@t")
    keep = _create_session(api, h, world).json()["id"]
    assert api.patch(f"/api/sessions/{keep}", headers=h, json={"pinned": True}).status_code == 200
    first_unpinned = _create_session(api, h, world).json()["id"]

    for _ in range(settings.history_max_sessions + 2):
        _create_session(api, h, world)

    page1 = [s["id"] for s in api.get("/api/sessions", headers=h).json()]
    assert page1[0] == keep, "pinned sessions should group first"
    assert len(page1) == settings.history_max_sessions
    older = [s["id"] for s in api.get(f"/api/sessions?offset={len(page1)}", headers=h).json()]
    assert first_unpinned in older, "an old chat was deleted instead of paged"
    assert db.execute(text("SELECT count(*) FROM chat_session")).scalar() == settings.history_max_sessions + 4



def test_foreign_scope_in_a_message_payload_is_ignored(api, world, db, llm):
    h = login(api, "tech@t")
    sid = _create_session(api, h, world).json()["id"]
    llm.script["understand"] = [Understanding(intent="smalltalk", slots=Slots(error_codes=[], component_hint="", symptom=""), wanted_doc_types=[], plan=[], standalone_query="hi", language="en",
                                              rationale="greeting")]
    body = {"text": "hello", "machine_line_id": str(world["b"].id), "client_id": str(world["c2"].id)}
    with api.stream("POST", f"/api/sessions/{sid}/messages", headers=h, json=body) as r:
        assert r.status_code == 200
        r.read()
    detail = api.get(f"/api/sessions/{sid}", headers=h).json()
    assert detail["machine_label"] == "MACHINE A"


def test_sessions_are_private_to_their_owner(api, world):
    ha, ht = login(api, "admin@t"), login(api, "tech@t")
    sid = _create_session(api, ht, world).json()["id"]
    assert api.get(f"/api/sessions/{sid}", headers=ha).status_code == 404


# ---- FR-14 no raw exposure -----------------------------------------------------------------------------
_FORBIDDEN = {"content", "storage_uri", "path", "file_path", "sha256", "embedding", "tsv", "outline", "serial_ref"}


def _props(schema, defs, seen=None):
    seen = seen or set()
    out = set()
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        if name in seen:
            return out
        return _props(defs[name], defs, seen | {name})
    for k in ("anyOf", "allOf", "oneOf"):
        for s in schema.get(k, []):
            out |= _props(s, defs, seen)
    if "items" in schema:
        out |= _props(schema["items"], defs, seen)
    for name, sub in schema.get("properties", {}).items():
        out.add(name)
        out |= _props(sub, defs, seen)
    return out


def test_no_technician_facing_response_schema_can_carry_manual_content_or_paths(api):
    spec = api.fastapi.openapi()
    defs = spec["components"]["schemas"]
    checked = 0
    for path, item in spec["paths"].items():
        if path.startswith("/api/admin") or path in ("/metrics", "/readyz", "/healthz"):
            continue
        for op in item.values():
            for resp in op.get("responses", {}).values():
                schema = resp.get("content", {}).get("application/json", {}).get("schema")
                if schema:
                    checked += 1
                    bad = _props(schema, defs) & _FORBIDDEN
                    assert not bad, f"{path} exposes {bad}"
    assert checked >= 5


def test_there_is_no_file_serving_route(api):
    # D-A (2026-09-29) lets technicians see cited passages, drawings and their own photos. Those come from
    # /api/messages/{id}/sources/{n}, /api/figures/{id} and /api/attachments/{id}, each access-checked; what
    # stays forbidden is a generic route that hands out manual files or storage paths.
    paths = list(api.fastapi.openapi()["paths"])
    assert not any(k in p for p in paths for k in ("download", "/files", "/static", "/raw", "/content", "/file"))
    doc_ops = api.fastapi.openapi()["paths"]
    for p, item in doc_ops.items():
        if "/documents/" in p:
            assert set(item) == {"patch"}, f"{p} must be retag-only, found {set(item)}"


def test_planted_canary_sentence_never_reaches_an_answer(api, world, db, llm):
    """A sentence copied from a manual must be dropped by the verbatim guard."""
    from tests.conftest import add_doc

    canary = ("CANARY the quick brown fox jumps over the lazy dog while the technician tightens all twelve terminal "
              "screws to the specified value using a calibrated torque driver.")
    add_doc(db, world["a"], "Canary Manual", [canary], doc_type="instructions")
    db.commit()
    h = login(api, "tech@t")
    sid = _create_session(api, h, world).json()["id"]
    docs = api.get(f"/api/sessions/{sid}", headers=h)  # noqa: F841
    llm.script["understand"] = [Understanding(intent="information", slots=Slots(error_codes=[], component_hint="", symptom=""), wanted_doc_types=[], plan=[], standalone_query="the quick brown fox jumps over the lazy dog technician tightens twelve terminal screws torque driver",
                                              language="en", rationale="r")]
    llm.script["assess_grade"] = [Grade(sufficient=True, missing=[], llm_confidence=.9, rationale="ok")]
    llm.text = f"Tighten the terminals [1]. {canary} Then close the cover [1]."
    events = _post(api, h, sid, "how do I tighten the terminal screws")
    answer = "".join(json.loads(e["data"]) for e in events if e["event"] == "token")
    assert "quick brown fox" not in answer
    assert "Tighten the terminals" in answer and "close the cover" in answer


def _post(api, h, sid, text_, option_index=None):
    body = {"text": text_} | ({"option_index": option_index} if option_index is not None else {})
    events = []
    with api.stream("POST", f"/api/sessions/{sid}/messages", headers=h, json=body) as r:
        assert r.status_code == 200
        cur = {}
        for line in r.iter_lines():
            if line.startswith("event:"):
                cur = {"event": line.split(":", 1)[1].strip()}
            elif line.startswith("data:"):
                cur["data"] = line.split(":", 1)[1].strip()
                events.append(cur)
    return events


def test_unknown_llm_script_ends_the_stream_with_a_clean_error(api, world):
    h = login(api, "tech@t")
    sid = _create_session(api, h, world).json()["id"]
    events = _post(api, h, sid, "something")  # FakeLLM has no script -> LLMError -> ERROR done
    last = json.loads(events[-1]["data"])
    assert events[-1]["event"] == "done" and last["action"] == "ERROR" and last["message"]


_ = (Clarify,)
