"""Milestone A: the pre-display safety review, star ratings, golden-example candidates, the reviewer queue,
retention and the nightly summaries. FakeLLM + hash embedder against real Postgres, as everywhere else."""
import json
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, text

from app.agent.safety import mentions_safety, tripwires
from app.agent.schemas import Grade, SafetyVerdict
from app.domain.support import (
    AppUser,
    ChatMessage,
    ChatSession,
    ConversationSummary,
    GoldenExample,
    LlmBatch,
    RetrievalGap,
    SafetyEvent,
)
from app.llm.base import LLMError
from app.llm.fake import FakeLLM
from app.quality import retention, summaries
from tests.agent.test_scenarios import GOOD, U, done, start, text_of, turn
from tests.conftest import login

pytestmark = pytest.mark.db

DRAFT = "Before you start: isolate power [1]. Wait ten minutes, then reset the thermal switch [1]."


def verdict(v, warnings=(), issues=()):
    return SafetyVerdict(verdict=v, issues=list(issues), warnings=list(warnings), rationale="scripted")


def answering_llm(review=None, text=DRAFT):
    script = {"understand": [U(codes=["E4"], query="alarm E4 drum motor thermal switch", route=["D1"])],
              "assess_grade": [GOOD]}
    if review is not None:
        script["safety_review"] = [review]
    return FakeLLM(script, text=text)


# ---- tripwires --------------------------------------------------------------------------------------------
def test_tripwires_catch_bypass_and_live_work_in_greek_and_english_with_or_without_accents():
    assert tripwires("Can I bypass the door interlock to test the drum?")
    assert tripwires("measure it while it is running")
    assert tripwires("Μπορώ να κάνω παράκαμψη του διακόπτη;")
    assert tripwires("ΓΕΦΥΡΩΣΤΕ ΤΙΣ ΕΠΑΦΕΣ")                     # upper case, no tonos
    assert tripwires("μέτρηση υπό τάση")


def test_tripwires_stay_quiet_on_ordinary_repair_text():
    assert tripwires(DRAFT) == []
    assert tripwires("Αλλάξτε το ρουλεμάν και ελέγξτε τη στάθμη λαδιού.") == []
    assert tripwires("Check the pump and the oil level, then restart the machine.") == []


def test_safety_comments_are_recognised():
    assert mentions_safety("This is dangerous, I nearly got a shock")
    assert mentions_safety("Επικίνδυνη οδηγία")
    assert not mentions_safety("Great, fixed it in five minutes")
    assert not mentions_safety(None)


# ---- the review in the turn -------------------------------------------------------------------------------
def test_every_answer_is_reviewed_before_any_text_is_released(db, world):
    user, s = start(db, world)
    llm = answering_llm()
    ev = turn(db, llm, user, s, "Alarm E4 on the drum motor")
    statuses = [e["data"]["key"] for e in ev if e["event"] == "status"]
    first_token = next(i for i, e in enumerate(ev) if e["event"] == "token")
    safety_status = next(i for i, e in enumerate(ev) if e["event"] == "status" and e["data"]["key"] == "safety")
    assert "safety" in statuses and safety_status < first_token
    assert [c["node"] for c in llm.calls][-2:] == ["synthesize", "safety_review"]
    assert done(ev)["action"] == "ANSWER" and done(ev)["message_id"]
    assert db.scalar(select(SafetyEvent)) is None        # clean approval, no keyword: nothing to review


def test_amended_answer_shows_the_warnings_above_the_steps_and_records_an_event(db, world):
    user, s = start(db, world)
    ev = turn(db, answering_llm(verdict("amend", ["Wear insulated gloves.", "Lock out the main switch."])),
              user, s, "Alarm E4 on the drum motor")
    shown = text_of(ev)
    assert shown.startswith("> **Safety first**")
    assert shown.index("Lock out the main switch.") < shown.index("isolate power")
    assert done(ev)["action"] == "ANSWER"
    e = db.scalar(select(SafetyEvent))
    assert e.kind == "amended" and e.detail["verdict"]["warnings"]


def test_blocked_answer_is_never_shown_and_becomes_a_hard_safety_candidate(db, world):
    user, s = start(db, world)
    unsafe = "Bridge out the door interlock with a jumper wire [1] and run the drum to test it."
    ev = turn(db, answering_llm(verdict("block", issues=["suggests defeating the interlock"]), text=unsafe),
              user, s, "the drum will not start with the door closed")
    shown = text_of(ev)
    assert "jumper" not in shown and "MACHINE_BRAND service" in shown
    assert done(ev)["action"] == "ESCALATE" and done(ev)["sources"] == []
    e = db.scalar(select(SafetyEvent))
    assert e.kind == "blocked" and "jumper" in e.detail["draft"] and e.detail["strict"] is True
    assert "safety review blocked" in db.scalar(select(RetrievalGap)).missing
    g = db.scalar(select(GoldenExample))
    assert g.kind == "hard" and g.safety and g.status == "candidate" and g.snapshot["answer"] == shown


def test_a_draft_that_trips_a_keyword_is_flagged_even_when_approved(db, world):
    user, s = start(db, world)
    turn(db, answering_llm(text="Do not bypass the interlock. Isolate power first [1]."), user, s,
         "Alarm E4 on the drum motor")
    e = db.scalar(select(SafetyEvent))
    assert e.kind == "tripwire" and e.detail["draft_hits"] == ["bypass"] and e.detail["strict"] is True


def test_a_failed_review_fails_closed(db, world):
    user, s = start(db, world)

    def boom(_):
        raise LLMError("reviewer down")

    ev = turn(db, answering_llm(boom), user, s, "Alarm E4 on the drum motor")
    assert done(ev)["action"] == "ERROR"
    assert "isolate power" not in text_of(ev)            # the unreviewed draft never went out
    assert db.scalar(select(ChatMessage).where(ChatMessage.role == "assistant")) is None


# ---- ratings and candidates -------------------------------------------------------------------------------
def _answered(api, world, llm):
    from tests.integration.test_api import _create_session, _post

    h = login(api, "tech@t")
    sid = _create_session(api, h, world).json()["id"]
    llm.script.update({"understand": [U(codes=["E4"], query="alarm E4 drum motor thermal switch", route=["D1"])],
                       "assess_grade": [Grade(sufficient=True, missing=[], llm_confidence=.9, rationale="ok")]})
    llm.text = DRAFT
    events = _post(api, h, sid, "Alarm E4 on the drum motor")
    return h, sid, json.loads(events[-1]["data"])["message_id"]


def rate(api, h, sid, stars, comment=None):
    return api.put(f"/api/sessions/{sid}/rating", headers=h, json={"stars": stars, "comment": comment})


def test_five_stars_on_the_conversation_nominate_a_positive_candidate(api, world, db, llm):
    h, sid, _ = _answered(api, world, llm)
    r = rate(api, h, sid, 5)
    assert r.status_code == 200 and r.json() == {"stars": 5, "comment": None}
    g = db.scalar(select(GoldenExample))
    assert g.kind == "positive" and str(g.session_id) == sid and "isolate power" in g.reference_answer
    assert g.snapshot["question"] == "Alarm E4 on the drum motor" and len(g.snapshot["answers"]) == 1
    assert g.snapshot["evidence"] and "content" not in json.dumps(g.snapshot["evidence"])   # references, not text
    detail = api.get(f"/api/sessions/{sid}", headers=h).json()
    assert detail["rating"] == {"stars": 5, "comment": None} and detail["has_answers"] is True
    assert "rating" not in detail["messages"][-1]                     # no per-answer rating any more


def test_re_rating_moves_or_drops_an_unreviewed_candidate(api, world, db, llm):
    h, sid, _ = _answered(api, world, llm)
    rate(api, h, sid, 5)
    rate(api, h, sid, 3, "ok")
    assert db.scalar(select(GoldenExample)) is None
    rate(api, h, sid, 1, "wrong switch")
    db.expire_all()
    g = db.scalar(select(GoldenExample))
    assert g.kind == "hard" and g.stars == 1 and g.comment == "wrong switch" and g.reference_answer is None
    assert db.get(ChatSession, g.session_id).rating == 1


def test_a_safety_comment_goes_to_the_safety_queue(api, world, db, llm):
    h, sid, _ = _answered(api, world, llm)
    rate(api, h, sid, 2, "Dangerous, no lockout step")
    e = db.scalar(select(SafetyEvent).where(SafetyEvent.kind == "feedback"))
    assert e and e.detail["comment"] == "Dangerous, no lockout step" and str(e.session_id) == sid


def test_only_your_own_answered_conversation_can_be_rated(api, world, db, llm):
    from tests.integration.test_api import _create_session

    h, sid, _ = _answered(api, world, llm)
    assert rate(api, login(api, "admin@t"), sid, 5).status_code == 404
    assert rate(api, h, sid, 6).status_code == 422
    empty = _create_session(api, h, world).json()["id"]
    assert rate(api, h, empty, 4).status_code == 422                  # nothing answered yet


# ---- the reviewer queue -----------------------------------------------------------------------------------
def test_review_is_for_admins_and_flagged_reviewers_and_hard_cases_need_the_lead_technician(api, world, db, llm):
    h, sid, _ = _answered(api, world, llm)
    rate(api, h, sid, 1, "wrong")
    ha = login(api, "admin@t")
    assert api.get("/api/review/examples", headers=h).status_code == 403

    [ex] = api.get("/api/review/examples", headers=ha).json()
    assert ex["kind"] == "hard"
    api.patch(f"/api/review/examples/{ex['id']}", headers=ha, json={"failure_cause": "answer_error"})
    r = api.patch(f"/api/review/examples/{ex['id']}", headers=ha, json={"status": "approved"})
    assert r.status_code == 403                                   # admin without the reviewer flag

    tech = db.scalar(select(AppUser).where(AppUser.email == "tech@t"))
    assert api.patch(f"/api/admin/users/{tech.email}", headers=ha,
                     json={"is_reviewer": True}).json()["is_reviewer"] is True
    assert api.get("/api/auth/me", headers=h).json()["can_review"] is True
    r = api.patch(f"/api/review/examples/{ex['id']}", headers=h, json={"status": "approved"})
    assert r.status_code == 422                                   # no reference answer yet
    api.patch(f"/api/review/examples/{ex['id']}", headers=h,
              json={"reference_answer": "Reset the thermal switch after isolating."})
    r = api.patch(f"/api/review/examples/{ex['id']}", headers=h, json={"status": "approved"})
    assert r.status_code == 200 and r.json()["reviewed_by"] == "tech@t"
    assert r.json()["failure_cause"] == "answer_error"

    assert api.post("/api/review/golden-sets", headers=h).status_code == 403   # publishing is admin-only
    g = api.post("/api/review/golden-sets", headers=ha).json()
    assert g["version"] == 1 and g["example_count"] == 1
    frozen = api.get("/api/review/golden-sets/1", headers=h).json()
    assert frozen["examples"][0]["reference_answer"] == "Reset the thermal switch after isolating."


def test_safety_events_can_be_resolved_with_a_note(api, world, db, llm):
    h, sid, _ = _answered(api, world, llm)
    rate(api, h, sid, 2, "unsafe")
    ha = login(api, "admin@t")
    [e] = api.get("/api/review/safety-events", headers=ha).json()
    r = api.patch(f"/api/review/safety-events/{e['id']}", headers=ha, json={"resolved": True, "note": "prompt fixed"})
    assert r.json()["resolved"] and r.json()["note"] == "prompt fixed"
    assert api.get("/api/review/safety-events", headers=ha).json() == []


# ---- retention --------------------------------------------------------------------------------------------
def test_retention_keeps_rated_and_pinned_chats_longer_and_detaches_what_outlives_them(db, world):
    user = AppUser(email="r@t", role="technician")
    db.add(user)
    db.flush()
    m = world["a"]
    now = datetime.now(UTC)

    def chat(days_old, pinned=False, stars=None):
        s = ChatSession(user_id=user.id, client_id=m.client_id, machine_line_id=m.id, working_memory={},
                        pinned=pinned, updated_at=now - timedelta(days=days_old), rating=stars,
                        rated_at=now - timedelta(days=days_old) if stars else None)
        db.add(s)
        db.flush()
        db.add(ChatMessage(session_id=s.id, role="assistant", content="answer", meta={"action": "ANSWER"}))
        db.flush()
        return s.id

    fresh, stale, pinned_stale = chat(10), chat(100), chat(100, pinned=True)
    rated_stale, rated_ancient, pinned_ancient = chat(200, stars=4), chat(400, stars=5), chat(400, pinned=True)
    db.add(ConversationSummary(session_id=rated_ancient, machine_line_id=m.id, stars_min=5, stars_max=5,
                               summary={"problem": "x"}, feedback_through=now))
    db.commit()

    assert retention.apply(db) == {"unrated_deleted": 1, "rated_or_pinned_deleted": 2}
    db.expire_all()
    left = set(db.scalars(select(ChatSession.id)))
    assert left == {fresh, pinned_stale, rated_stale}
    assert db.scalar(select(ConversationSummary)).session_id is None      # the summary outlives its chat
    assert stale not in left and pinned_ancient not in left


# ---- nightly summaries ------------------------------------------------------------------------------------
class FakeBatches:
    def __init__(self):
        self.submitted: list[list[dict]] = []
        self.results: dict[str, dict | None] = {}

    def submit(self, requests):
        self.submitted.append(requests)
        bid = f"batch_{len(self.submitted)}"
        self.results[bid] = None
        return bid

    def poll(self, batch_id):
        return self.results[batch_id]


SUMMARY = {"language": "en", "component": "drum motor", "error_codes": ["E4"], "problem": "Alarm E4.",
           "steps_tried": ["reset the thermal switch"], "resolution": "unknown", "outcome": "unknown",
           "cited_documents": [{"title": "A Manual", "page": 1}],
           "technician_feedback": "Rated 5. Call me on +30 694 123 4567 or tech@example.com",
           "answer_quality_notes": "Correct."}


def test_rated_conversations_are_summarised_by_batch_and_resubmitted_after_a_new_rating(api, world, db, llm):
    h, sid, _ = _answered(api, world, llm)
    rate(api, h, sid, 5)
    client = FakeBatches()

    assert summaries.submit(db, client) == 1
    req = client.submitted[0][0]
    assert req["params"]["output_config"]["format"]["type"] == "json_schema"
    assert "CONVERSATION RATED 5/5" in req["params"]["messages"][0]["content"]
    assert summaries.submit(db, client) == 0                     # already in flight
    assert summaries.collect(db, client) == 0                    # still processing

    client.results["batch_1"] = {req["custom_id"]: json.dumps(SUMMARY)}
    assert summaries.collect(db, client) == 1
    row = db.scalar(select(ConversationSummary))
    assert str(row.session_id) == sid and row.stars_max == 5
    assert "[phone]" in row.summary["technician_feedback"] and "[email]" in row.summary["technician_feedback"]
    assert db.scalar(select(LlmBatch)).status == "collected"
    assert summaries.pending(db) == []

    rate(api, h, sid, 2, "actually it did not work")
    assert [p[0] for p in summaries.pending(db)] == [row.session_id]


def test_a_failed_summary_request_is_retried_in_the_next_batch(api, world, db, llm):
    h, sid, _ = _answered(api, world, llm)
    rate(api, h, sid, 4)
    client = FakeBatches()
    summaries.submit(db, client)
    client.results["batch_1"] = {client.submitted[0][0]["custom_id"]: None}
    assert summaries.collect(db, client) == 0
    assert summaries.submit(db, client) == 1


def test_llm_batch_table_is_cleaned_between_tests(db):
    assert db.execute(text("SELECT count(*) FROM llm_batch")).scalar() == 0
