"""Agent scenarios from PRD section 9, driven by a scripted FakeLLM against real Postgres + pgvector."""
import json

import pytest
from sqlalchemy import select

from app.agent.graph import Agent
from app.agent.schemas import (
    Clarify,
    Grade,
    PlanItem,
    Reformulation,
    RetrieveDecision,
    Slots,
    Understanding,
)
from app.config import settings
from app.domain.support import AgentRun, AgentStep, AppUser, ChatSession, RetrievalGap, RetrievalHit
from app.llm.fake import FakeLLM
from tests.conftest import EMB

pytestmark = pytest.mark.db


def plan(*aliases, reason="r"):
    return [PlanItem(document_id=a, section_hints=[], reason=reason) for a in aliases]


def U(intent="troubleshoot", codes=(), symptom=None, comp=None, query="q", lang="en", types=(), route=()):
    """Understanding + routing plan (one merged model call). `route` = catalog aliases (or raw ids) to plan."""
    return Understanding(intent=intent, slots=Slots(error_codes=list(codes), symptom=symptom or "",
                                                    component_hint=comp or ""),
                         wanted_doc_types=list(types), standalone_query=query, language=lang, plan=plan(*route),
                         rationale="scripted")


GOOD = Grade(sufficient=True, missing=[], llm_confidence=.9, rationale="covers it")
FINISH = RetrieveDecision(actions=[], rationale="enough")


def start(db, world, machine="a"):
    user = AppUser(email="t@t", password_hash="x", role="technician")
    db.add(user)
    db.flush()
    m = world[machine]
    s = ChatSession(user_id=user.id, client_id=m.client_id, machine_line_id=m.id, working_memory={})
    db.add(s)
    db.commit()
    return user, s


def turn(db, llm, user, session, text, option_index=None):
    events = list(Agent(db, llm, EMB).run_turn(session, user, text, option_index))
    db.refresh(session)
    return events


def kinds(events):
    return [e["event"] for e in events]


def text_of(events):
    return "".join(e["data"] for e in events if e["event"] == "token")


def done(events):
    return events[-1]["data"]


def test_clear_question_is_answered_with_sources_and_a_full_trace(db, world):
    user, s = start(db, world)
    llm = FakeLLM({"understand": [U(codes=["E4"], query="alarm E4 drum motor thermal switch", route=["D1"])],
                   "assess_grade": [GOOD]},
                  text="Before you start: isolate power [1]. Wait ten minutes, then reset the switch [1].")
    ev = turn(db, llm, user, s, "Alarm E4 on the drum motor")
    assert kinds(ev)[-1] == "done" and "token" in kinds(ev) and "status" in kinds(ev)
    d = done(ev)
    assert d["action"] == "ANSWER"
    assert d["sources"] == [{"n": 1, "title": "A Manual", "doc_type": "Operating manual", "page": 1, "section": None,
                             "figure_id": None}]
    # the marker stays, renumbered to the source list, so it can open the cited passage
    assert "isolate power [1]" in text_of(ev) and "[1] [1]" not in text_of(ev)

    run = db.scalar(select(AgentRun))
    # understand, assess_grade, synthesize, safety_review
    assert run.final_action == "ANSWER" and run.llm_calls == 4 and run.latency_ms is not None
    nodes = [st.node for st in db.scalars(select(AgentStep).order_by(AgentStep.seq))]
    assert nodes[:3] == ["understand", "route", "retrieve"] and "assess" in nodes and "synthesize" in nodes
    assert nodes.index("safety_review") > nodes.index("synthesize")    # the answer is judged after it is written
    hits = db.scalars(select(RetrievalHit)).all()
    assert hits and {h.document_id for h in hits} == {world["doc_a"].id}      # scope held all the way through
    assert any(h.selected for h in hits) and any(h.vector_distance is not None for h in hits)
    from app.agent.prompts.v1 import PROMPT_VERSION
    assert all(st.prompt_version == PROMPT_VERSION for st in db.scalars(select(AgentStep)) if st.llm_model and st.node != "synthesize")


def test_ambiguous_component_asks_with_real_tags_then_a_reply_routes_and_answers(db, world):
    user, s = start(db, world, "line")
    llm = FakeLLM({
        "understand": [U(codes=["E7"], query="alarm E7", route=["D1", "D2"]),
                       U(codes=["E7"], query="alarm E7 press", route=["D2"])],
        "assess_grade": [GOOD],
        "clarify": [Clarify(question="Which part of the line is showing E7?", options=[])]},
        text="Check the pump and the oil level [1].")
    ev = turn(db, llm, user, s, "The line stops with alarm E7")
    assert "question" in kinds(ev) and kinds(ev)[-1] == "done"
    q = next(e["data"] for e in ev if e["event"] == "question")
    labels = [o["label"] for o in q["options"]]
    assert set(labels) >= {"Press", "Dryers", "Not sure"}                    # chips are real component tags
    assert all("effect" not in o for o in q["options"])                       # routing effects stay server-side
    assert done(ev)["action"] == "ASK" and s.pending_question and s.working_memory["asked"] == ["component"]

    press_idx = labels.index("Press")
    ev2 = turn(db, llm, user, s, "Press", option_index=press_idx)
    assert done(ev2)["action"] == "ANSWER"
    assert done(ev2)["sources"][0]["title"] == "Press Manual"
    assert s.pending_question is None
    assert not [e for e in ev2 if e["event"] == "question"]                  # the same question is never asked twice
    seen_by_model = [c["user"] for c in llm.calls if c["node"] == "understand"][-1]
    assert "Press Manual" in seen_by_model and "Dryer Manual" not in seen_by_model  # chip effect narrowed routing


def test_budget_spent_escalates_honestly_and_writes_a_gap_without_calling_synthesis(db, world):
    user, s = start(db, world)
    llm = FakeLLM({"understand": [U("information", query="zebra migration patterns in antarctica", route=["D1"])],
                   "expand": [Reformulation(queries=["penguin habitat"], widen="all_types", rationale="try wider")]})
    ev = turn(db, llm, user, s, "zebra migration patterns in antarctica")
    d = done(ev)
    assert d["action"] == "ESCALATE" and d["sources"] == []
    assert "could not find" in text_of(ev).lower()
    expands = [c["node"] for c in llm.calls].count("expand")
    assert expands == settings.max_expand_loops and expands <= 2              # the loop budget is honoured; a 3rd never runs
    assert "synthesize" not in [c["node"] for c in llm.calls]                  # no invented answer
    gap = db.scalar(select(RetrievalGap))
    assert gap and "zebra" in gap.question and gap.machine_line_id == world["a"].id


def test_route_plan_naming_a_foreign_document_is_dropped_and_logged(db, world):
    user, s = start(db, world)
    foreign = str(world["doc_b"].id)
    llm = FakeLLM({"understand": [U(codes=["E4"], query="alarm E4 drum motor", route=[foreign, "D1"])],
                   "assess_grade": [GOOD]}, text="Reset the switch [1].")
    ev = turn(db, llm, user, s, "Alarm E4")
    assert done(ev)["action"] == "ANSWER"
    step = db.scalar(select(AgentStep).where(AgentStep.node == "route"))
    assert step.output["dropped_ids"] == [foreign] and step.output["resolved"] == ["D1"]
    assert {h.document_id for h in db.scalars(select(RetrievalHit))} == {world["doc_a"].id}


def test_missing_slot_interviews_before_any_retrieval(db, world):
    user, s = start(db, world)
    llm = FakeLLM({"understand": [U("troubleshoot", query="machine problem")],
                   "clarify": [Clarify(question="What is the machine doing?",
                                       options=["Alarm code on the panel", "Does not start", "Stops mid-cycle"])]})
    ev = turn(db, llm, user, s, "it is broken")
    q = next(e["data"] for e in ev if e["event"] == "question")
    assert [o["label"] for o in q["options"]] == ["Alarm code on the panel", "Does not start", "Stops mid-cycle", "Not sure"]
    assert not db.scalars(select(RetrievalHit)).all() and "route" not in [c["node"] for c in llm.calls]


def test_at_most_two_interview_turns_per_issue(db, world):
    user, s = start(db, world, "line")
    llm = FakeLLM({"understand": [U(codes=["E7"], query="alarm E7", route=["D1", "D2"])],
                   "assess_grade": [GOOD],
                   "clarify": [Clarify(question="Which part?", options=[])]}, text="Answer text [1].")
    turn(db, llm, user, s, "E7 alarm")                                            # ask #1 (component)
    labels = [o["label"] for o in s.pending_question["options"]]
    ev = turn(db, llm, user, s, "Not sure", option_index=labels.index("Not sure"))  # no narrowing -> next facet or answer
    ev = turn(db, llm, user, s, "still not sure") if done(ev)["action"] == "ASK" else ev
    assert s.working_memory.get("ask_turns", 0) <= 2
    assert done(ev)["action"] in {"ANSWER", "ASK"}


def test_smalltalk_gets_a_greeting_and_no_retrieval(db, world):
    user, s = start(db, world)
    llm = FakeLLM({"understand": [U("smalltalk", query="hi")]})
    ev = turn(db, llm, user, s, "hello")
    assert done(ev)["action"] == "ANSWER" and "MACHINE A" in text_of(ev)
    assert [c["node"] for c in llm.calls] == ["understand"] and not s.working_memory.get("issue")


def test_answers_follow_the_technicians_language(db, world):
    user, s = start(db, world)
    llm = FakeLLM({"understand": [U("smalltalk", query="γεια", lang="el")]})
    assert "Γεια σας" in text_of(turn(db, llm, user, s, "γεια σας"))


def test_llm_outage_ends_the_turn_cleanly_and_records_the_error(db, world):
    user, s = start(db, world)
    ev = turn(db, FakeLLM(), user, s, "anything")                                 # no scripted nodes -> LLMError
    d = done(ev)
    assert d["action"] == "ERROR" and d["message"]
    run = db.scalar(select(AgentRun))
    assert run.final_action == "ERROR"
    assert db.scalar(select(AgentStep).where(AgentStep.error.is_not(None)))


def test_session_reopens_with_messages_and_pending_question(db, world):
    user, s = start(db, world, "line")
    llm = FakeLLM({"understand": [U(codes=["E7"], query="alarm E7", route=["D1", "D2"])],
                   "assess_grade": [GOOD], "clarify": [Clarify(question="Which part?", options=[])]})
    turn(db, llm, user, s, "E7")
    db.expire_all()
    s2 = db.get(ChatSession, s.id)
    assert [m.role for m in s2.messages] == ["user", "assistant"] and s2.pending_question["question"] == "Which part?"
    assert json.dumps(s2.pending_question)  # persisted as JSON
