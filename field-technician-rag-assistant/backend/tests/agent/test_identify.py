"""Feature 1: a chat starts without a machine, the Questioner agent identifies it, the scope is locked once, and
the question asked before the machine was known is answered by the RAG flow in that scope."""
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.agent.graph import Agent
from app.agent.schemas import MachineGuess
from app.domain.support import AgentRun, AppUser, ChatSession
from app.llm.fake import FakeLLM
from tests.agent.test_scenarios import GOOD, U, done, kinds, text_of
from tests.conftest import EMB, login

pytestmark = pytest.mark.db


def alias_of(label: str):
    """The catalog numbers machines by client then label; read the alias the model would see."""
    def pick(user: str) -> str:
        line = next(ln for ln in user.splitlines() if f"| {label} |" in ln)
        return line.split(" | ")[0]
    return pick


def guess(machine=None, candidates=(), question="", has_issue=True, lang="en"):
    def build(user: str) -> MachineGuess:
        return MachineGuess(machine=alias_of(machine)(user) if machine else "",
                            candidates=[alias_of(c)(user) for c in candidates], question=question,
                            has_issue=has_issue, language=lang, rationale="scripted")
    return build


def unscoped(db):
    user = AppUser(email="t@t", role="technician")
    db.add(user)
    db.flush()
    s = ChatSession(user_id=user.id, working_memory={})
    db.add(s)
    db.commit()
    return user, s


def turn(db, llm, user, s, msg, option_index=None):
    ev = list(Agent(db, llm, EMB).run_turn(s, user, msg, option_index))
    db.refresh(s)
    return ev


def test_a_named_machine_is_locked_and_the_original_question_answered_in_the_same_turn(db, world):
    user, s = unscoped(db)
    llm = FakeLLM({"identify": [guess("MACHINE A")],
                   "understand": [U(codes=["E4"], query="alarm E4 drum motor", route=["D1"])],
                   "assess_grade": [GOOD]}, text="Before you start: isolate power [1]. Reset the switch [1].")
    ev = turn(db, llm, user, s, "Alarm E4 on MACHINE A at Client One")
    assert kinds(ev).count("scope") == 1 and next(e for e in ev if e["event"] == "scope")["data"] == {
        "machine_label": "MACHINE A"}
    assert done(ev)["action"] == "ANSWER" and "isolate power" in text_of(ev)
    assert s.machine_line_id == world["a"].id and s.client_id == world["c1"].id
    understand_input = next(c for c in llm.calls if c["node"] == "understand")["user"]
    assert "Alarm E4 on MACHINE A" in understand_input
    run = db.query(AgentRun).one()                          # one traced run per turn, scope filled in on lock
    assert run.machine_line_id == world["a"].id


def test_ambiguous_machine_asks_with_chips_and_a_tap_locks_it_without_asking_the_model_again(db, world):
    user, s = unscoped(db)
    llm = FakeLLM({"identify": [guess(None, ["MACHINE A", "MACHINE B"], "Which machine at Client One?")],
                   "understand": [U(codes=["E4"], query="alarm E4 drum motor", route=["D1"])],
                   "assess_grade": [GOOD]}, text="Reset the switch [1].")
    ev = turn(db, llm, user, s, "Alarm E4 on the drum motor")
    q = next(e for e in ev if e["event"] == "question")["data"]
    assert done(ev)["action"] == "ASK" and s.machine_line_id is None
    assert [o["label"] for o in q["options"]] == ["MACHINE A · Client One", "MACHINE B · Client One",
                                                  "None of these"]

    ev = turn(db, llm, user, s, "MACHINE B · Client One", option_index=1)
    assert s.machine_line_id == world["b"].id and done(ev)["action"] == "ANSWER"
    assert [c["node"] for c in llm.calls].count("identify") == 1
    assert "Alarm E4 on the drum motor" in next(c for c in llm.calls if c["node"] == "understand")["user"]


def test_naming_the_machine_without_a_problem_locks_it_and_asks_what_is_wrong(db, world):
    user, s = unscoped(db)
    llm = FakeLLM({"identify": [guess(None, question="Which machine are you working on?", has_issue=False),
                                guess("WASH LINE", has_issue=False)]})
    ev = turn(db, llm, user, s, "hello")
    assert done(ev)["action"] == "ASK" and next(e for e in ev if e["event"] == "question")["data"]["options"] == []
    ev = turn(db, llm, user, s, "the wash line")
    assert s.machine_line_id == world["line"].id
    assert done(ev)["action"] == "ASK" and "WASH LINE" in text_of(ev)
    assert "understand" not in [c["node"] for c in llm.calls]


def test_an_alias_outside_the_catalog_is_never_locked(db, world):
    user, s = unscoped(db)
    llm = FakeLLM({"identify": [MachineGuess(machine="M99", candidates=["M42"], question="Which one?",
                                             has_issue=True, language="en", rationale="made up")]})
    ev = turn(db, llm, user, s, "Alarm on the dryer")
    assert s.machine_line_id is None and done(ev)["action"] == "ASK"


def test_scope_can_be_set_once_and_never_changed_or_cleared(db, world):
    _, s = unscoped(db)
    db.execute(text("UPDATE chat_session SET client_id=:c, machine_line_id=:m WHERE id=:s"),
               {"c": world["c1"].id, "m": world["a"].id, "s": s.id})
    db.commit()
    for new in ({"c": world["c1"].id, "m": world["b"].id}, {"c": None, "m": None}):
        with pytest.raises(DBAPIError):
            db.execute(text("UPDATE chat_session SET client_id=:c, machine_line_id=:m WHERE id=:s"), {**new, "s": s.id})
        db.rollback()


def test_half_a_scope_is_refused(db, world):
    _, s = unscoped(db)
    with pytest.raises(DBAPIError):
        db.execute(text("UPDATE chat_session SET machine_line_id=:m WHERE id=:s"), {"m": world["a"].id, "s": s.id})
    db.rollback()


def test_new_chat_needs_no_machine_and_does_not_pile_up_empty_chats(api, world):
    h = login(api, "tech@t")
    first = api.post("/api/sessions", headers=h)
    assert first.status_code == 201 and first.json()["machine_label"] is None
    assert api.post("/api/sessions", headers=h, json={}).json()["id"] == first.json()["id"]
    detail = api.get(f"/api/sessions/{first.json()['id']}", headers=h).json()
    assert detail["machine_label"] is None and detail["client_name"] is None
    assert [x["id"] for x in api.get("/api/sessions", headers=h).json()] == [first.json()["id"]]
