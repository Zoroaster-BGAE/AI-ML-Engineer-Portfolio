"""Golden-example candidates, written at rating time as a plain snapshot: no model call, no cost.

The technician rates the whole conversation: 5 stars -> positive candidate, 1-2 stars -> hard candidate,
3-4 stars -> none (the summary still records it). A blocked answer is a hard, safety candidate the moment it
happens, on that one message. Only an unreviewed candidate follows a changed rating; once a reviewer has
decided, the example is theirs and a re-rating does not touch it.
"""
from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.models import MachineLine
from app.domain.support import AgentRun, AgentStep, ChatMessage, ChatSession, GoldenExample, RetrievalHit

_TURNS = 40
ANSWER_ACTIONS = ("ANSWER", "ESCALATE")


def kind_for(stars: int) -> str | None:
    if stars == 5:
        return "positive"
    if stars <= 2:
        return "hard"
    return None


def _answer_record(db: Session, m: ChatMessage) -> dict:
    """One answer with how it was produced. Manual text is never copied: evidence is by reference."""
    meta = m.meta or {}
    out: dict = {"answer": m.content, "action": meta.get("action"), "sources": meta.get("sources") or [],
                 "safety_verdict": meta.get("safety")}
    run = db.get(AgentRun, m.run_id) if m.run_id else None
    if run is None:
        return out
    steps = db.scalars(select(AgentStep).where(AgentStep.run_id == run.id).order_by(AgentStep.seq)).all()
    by_node = {s.node: s for s in steps}
    out["question"] = run.question
    out["run"] = {"id": str(run.id), "final_action": run.final_action, "latency_ms": run.latency_ms,
                  "models": sorted({s.llm_model for s in steps if s.llm_model}),
                  "prompt_versions": sorted({s.prompt_version for s in steps if s.prompt_version})}
    if (u := by_node.get("understand")) is not None and u.output:
        out["understanding"] = {k: u.output.get(k) for k in ("intent", "slots", "standalone_query", "language",
                                                           "wanted_doc_types", "plan")}
    if (r := by_node.get("safety_review")) is not None and r.output:
        out["safety_review"] = r.output
    hits = db.execute(select(RetrievalHit).join(AgentStep, AgentStep.id == RetrievalHit.step_id)
                      .where(AgentStep.run_id == run.id, RetrievalHit.selected.is_(True))
                      .order_by(RetrievalHit.rank)).scalars().all()
    out["evidence"] = [{"document_id": str(h.document_id), "doc_label": h.doc_label, "page": h.page,
                        "chunk_id": str(h.chunk_id), "rank": h.rank} for h in hits]
    return out


def conversation_snapshot(db: Session, session: ChatSession, upto: ChatMessage | None = None) -> dict:
    """The conversation as it stood, copied so it outlives retention and the trace purge. The top-level
    question / answer / sources / evidence are the last answer's, which is what the review screen shows first;
    `answers` has every answer with its own evidence, for an eval replay."""
    q = select(ChatMessage).where(ChatMessage.session_id == session.id)
    if upto is not None:
        q = q.where(ChatMessage.created_at <= upto.created_at)
    msgs = db.scalars(q.order_by(ChatMessage.created_at)).all()[-_TURNS:]
    machine = db.get(MachineLine, session.machine_line_id) if session.machine_line_id else None
    answers = [_answer_record(db, m) for m in msgs
               if m.role == "assistant" and (m.meta or {}).get("action") in ANSWER_ACTIONS]
    last = answers[-1] if answers else {"answer": "", "sources": []}
    first_q = next((m.content for m in msgs if m.role == "user" and m.content), None)
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "machine": machine.label if machine else None,
        "machine_line_id": str(machine.id) if machine else None,
        "session_id": str(session.id),
        "turns": [{"role": m.role, "text": m.content} for m in msgs],
        "answers": answers,
        **{k: v for k, v in last.items() if k != "question"},
        "question": first_q,
    }


def nominate_from_session(db: Session, session: ChatSession) -> GoldenExample | None:
    existing = db.scalar(select(GoldenExample).where(GoldenExample.session_id == session.id,
                                                     GoldenExample.source_message_id.is_(None)))
    kind = kind_for(session.rating)
    if existing is not None:
        if existing.status != "candidate":
            return existing
        if kind is None:
            db.delete(existing)
            return None
        existing.kind, existing.stars, existing.comment = kind, session.rating, session.rating_comment
        existing.snapshot = conversation_snapshot(db, session)
        if kind == "positive" and not existing.reference_answer:
            existing.reference_answer = existing.snapshot.get("answer") or None
        existing.updated_at = datetime.now(UTC)
        return existing
    if kind is None:
        return None
    snap = conversation_snapshot(db, session)
    ex = GoldenExample(kind=kind, session_id=session.id, machine_line_id=session.machine_line_id,
                       stars=session.rating, comment=session.rating_comment, snapshot=snap,
                       reference_answer=(snap.get("answer") or None) if kind == "positive" else None)
    db.add(ex)
    return ex


def nominate_blocked(db: Session, message_id: UUID) -> None:
    message = db.get(ChatMessage, message_id)
    if message is None or db.scalar(select(GoldenExample.id).where(GoldenExample.source_message_id == message_id)):
        return
    session = db.get(ChatSession, message.session_id)
    db.add(GoldenExample(kind="hard", safety=True, session_id=session.id, source_message_id=message.id,
                         machine_line_id=session.machine_line_id,
                         snapshot=conversation_snapshot(db, session, upto=message)))
    db.commit()
