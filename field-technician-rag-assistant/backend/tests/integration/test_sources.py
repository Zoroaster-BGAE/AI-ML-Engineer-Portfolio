"""Feature 3: [n] in an answer opens the exact passage it cites -- for the owner and for reviewers only."""
import json

import pytest

from app.agent.nodes.synthesize import renumber, source_entries
from app.agent.schemas import Grade, Slots, Understanding
from app.retrieval.repository import Hit
from tests.conftest import login
from tests.integration.test_api import _create_session, _post

pytestmark = pytest.mark.db


def _hit(doc, page, cid, text="x"):
    from uuid import UUID
    return Hit(chunk_id=UUID(int=cid), document_id=UUID(int=doc), page=page, section_path="5.2 Drum", content=text,
               content_type="text", doc_title=f"Doc {doc}", doc_type="operating_manual", component=None)


def test_markers_are_renumbered_in_order_of_citation_and_merged_per_page():
    chunks = [_hit(1, 10, 1), _hit(2, 4, 2), _hit(1, 10, 3)]      # excerpts 1 and 3 are the same page
    sources, mapping = source_entries([2, 1, 3], chunks)
    assert [(s["n"], s["title"], s["page"], len(s["chunk_ids"])) for s in sources] == [(1, "Doc 2", 4, 1),
                                                                                       (2, "Doc 1", 10, 2)]
    assert renumber("Isolate [2]. Reset [1] [3]. Unknown [9].", mapping) == "Isolate [1]. Reset [2]. Unknown ."


def _answer(api, world, llm, who="tech@t"):
    h = login(api, who)
    sid = _create_session(api, h, world).json()["id"]
    llm.script["understand"] = [Understanding(intent="troubleshoot", slots=Slots(error_codes=["E4"], component_hint="",
                                                                                 symptom=""), wanted_doc_types=[],
                                              standalone_query="alarm E4 drum motor thermal switch", language="en",
                                              plan=[], rationale="r")]
    llm.script["assess_grade"] = [Grade(sufficient=True, missing=[], llm_confidence=.9, rationale="ok")]
    llm.text = "Isolate power first [1]. Then wait and reset it [1]."
    events = _post(api, h, sid, "Alarm E4 on the drum motor")
    done = json.loads(events[-1]["data"])
    return h, sid, done


def test_the_owner_opens_the_cited_passage(api, world, llm):
    h, _, done = _answer(api, world, llm)
    assert done["sources"][0]["n"] == 1 and "chunk_ids" not in done["sources"][0]   # ids stay server-side
    r = api.get(f"/api/messages/{done['message_id']}/sources/1", headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "A Manual" and body["page"] == 1
    assert "thermal switch" in body["excerpts"][0]


def test_reviewers_can_verify_but_other_users_and_customers_cannot(api, world, llm, db):
    from app.domain.support import AppUser

    _, _, done = _answer(api, world, llm)
    url = f"/api/messages/{done['message_id']}/sources/1"
    assert api.get(url, headers=login(api, "admin@t")).status_code == 200          # admin verifies
    import tests.conftest as c
    db.add_all([AppUser(email="other@t", role="technician"), AppUser(email="cust@t", role="customer")])
    db.commit()
    c._TEST_GROUPS.update({"other@t": ["technician"], "cust@t": ["customer"]})
    try:
        assert api.get(url, headers={"Authorization": "Bearer test.other@t"}).status_code == 404
        assert api.get(url, headers={"Authorization": "Bearer test.cust@t"}).status_code == 403
    finally:
        c._TEST_GROUPS.pop("other@t"), c._TEST_GROUPS.pop("cust@t")


def test_only_numbers_the_answer_cited_resolve(api, world, llm):
    h, _, done = _answer(api, world, llm)
    assert api.get(f"/api/messages/{done['message_id']}/sources/2", headers=h).status_code == 404
