"""Golden-example review and the safety queue (design §3.6, §5.4). Admins and flagged reviewers only.

The queue is deliberately short: two people review (the developer and the lead technician), so it serves a
weekly batch ordered by value -- safety first, then hard cases, then machines with the fewest approved
examples -- instead of an ever-growing list. Hard and safety examples need a flagged reviewer to approve.
"""
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth.dependencies import require_admin, require_reviewer
from app.db import get_session
from app.domain.models import MachineLine
from app.domain.support import AppUser, GoldenExample, GoldenSet, SafetyEvent

router = APIRouter(prefix="/api/review", tags=["review"], dependencies=[Depends(require_reviewer)])

FailureCause = Literal["retrieval_miss", "wrong_machine", "answer_error", "manual_missing", "safety", "other"]


class ExampleOut(BaseModel):
    id: UUID
    status: str
    kind: str
    safety: bool
    machine: str | None
    stars: int | None
    comment: str | None
    snapshot: dict
    reference_answer: str | None
    rubric_notes: str | None
    failure_cause: str | None
    reviewed_by: str | None
    reviewed_at: datetime | None
    created_at: datetime


class ExamplePatch(BaseModel):
    status: Literal["approved", "rejected", "candidate"] | None = None
    reference_answer: str | None = Field(default=None, max_length=8000)
    rubric_notes: str | None = Field(default=None, max_length=4000)
    failure_cause: FailureCause | None = None


class SafetyEventOut(BaseModel):
    id: UUID
    kind: str
    machine: str | None
    question: str | None
    detail: dict
    resolved: bool
    note: str | None
    created_at: datetime


class SafetyPatch(BaseModel):
    resolved: bool
    note: str | None = Field(default=None, max_length=2000)


class GoldenSetOut(BaseModel):
    version: int
    example_count: int
    created_at: datetime


def _labels(db: Session) -> dict:
    return {m.id: m.label for m in db.scalars(select(MachineLine))}


def _emails(db: Session, ids) -> dict:
    ids = {i for i in ids if i}
    return {u.id: u.email for u in db.scalars(select(AppUser).where(AppUser.id.in_(ids)))} if ids else {}


def _ex_out(e: GoldenExample, labels: dict, emails: dict) -> ExampleOut:
    return ExampleOut(id=e.id, status=e.status, kind=e.kind, safety=e.safety, machine=labels.get(e.machine_line_id),
                      stars=e.stars, comment=e.comment, snapshot=e.snapshot, reference_answer=e.reference_answer,
                      rubric_notes=e.rubric_notes, failure_cause=e.failure_cause,
                      reviewed_by=emails.get(e.reviewed_by), reviewed_at=e.reviewed_at, created_at=e.created_at)


@router.get("/examples", response_model=list[ExampleOut])
def list_examples(status: Literal["candidate", "approved", "rejected"] = "candidate",
                  limit: int = Query(10, ge=1, le=200), db: Session = Depends(get_session)):
    rows = list(db.scalars(select(GoldenExample).where(GoldenExample.status == status)))
    if status == "candidate":
        approved = dict(db.execute(select(GoldenExample.machine_line_id, func.count())
                                   .where(GoldenExample.status == "approved")
                                   .group_by(GoldenExample.machine_line_id)).all())
        rows.sort(key=lambda e: (not e.safety, e.kind != "hard", approved.get(e.machine_line_id, 0), e.created_at))
    else:
        rows.sort(key=lambda e: e.reviewed_at or e.created_at, reverse=True)
    rows = rows[:limit]
    labels, emails = _labels(db), _emails(db, [e.reviewed_by for e in rows])
    return [_ex_out(e, labels, emails) for e in rows]


@router.patch("/examples/{example_id}", response_model=ExampleOut)
def patch_example(example_id: UUID, body: ExamplePatch, user: AppUser = Depends(require_reviewer),
                  db: Session = Depends(get_session)):
    e = db.get(GoldenExample, example_id)
    if e is None:
        raise HTTPException(404, "Example not found")
    for f in ("reference_answer", "rubric_notes", "failure_cause"):
        if f in body.model_fields_set:
            setattr(e, f, getattr(body, f))
    if body.status is not None and body.status != e.status:
        if body.status == "approved":
            if (e.kind == "hard" or e.safety) and not user.is_reviewer:
                raise HTTPException(403, "Hard and safety examples need the lead technician's approval")
            if not (e.reference_answer or "").strip():
                raise HTTPException(422, "Write a reference answer before approving")
        e.status = body.status
        e.reviewed_by = user.id if body.status != "candidate" else None
        e.reviewed_at = datetime.now(UTC) if body.status != "candidate" else None
    e.updated_at = datetime.now(UTC)
    db.commit()
    return _ex_out(e, _labels(db), _emails(db, [e.reviewed_by]))


@router.get("/safety-events", response_model=list[SafetyEventOut])
def list_safety_events(resolved: bool = False, limit: int = Query(50, ge=1, le=500),
                       db: Session = Depends(get_session)):
    rows = db.scalars(select(SafetyEvent).where(SafetyEvent.resolved.is_(resolved))
                      .order_by(SafetyEvent.created_at.desc()).limit(limit)).all()
    labels = _labels(db)
    return [SafetyEventOut(id=r.id, kind=r.kind, machine=labels.get(r.machine_line_id), question=r.question,
                           detail=r.detail, resolved=r.resolved, note=r.note, created_at=r.created_at) for r in rows]


@router.patch("/safety-events/{event_id}", response_model=SafetyEventOut)
def patch_safety_event(event_id: UUID, body: SafetyPatch, user: AppUser = Depends(require_reviewer),
                       db: Session = Depends(get_session)):
    r = db.get(SafetyEvent, event_id)
    if r is None:
        raise HTTPException(404, "Safety event not found")
    r.resolved, r.note = body.resolved, body.note if body.note is not None else r.note
    r.resolved_by = user.id if body.resolved else None
    r.resolved_at = datetime.now(UTC) if body.resolved else None
    db.commit()
    return SafetyEventOut(id=r.id, kind=r.kind, machine=_labels(db).get(r.machine_line_id), question=r.question,
                          detail=r.detail, resolved=r.resolved, note=r.note, created_at=r.created_at)


@router.get("/golden-sets", response_model=list[GoldenSetOut])
def list_golden_sets(db: Session = Depends(get_session)):
    return [GoldenSetOut(version=g.version, example_count=g.example_count, created_at=g.created_at)
            for g in db.scalars(select(GoldenSet).order_by(GoldenSet.version.desc()))]


@router.get("/golden-sets/{version}")
def export_golden_set(version: int, db: Session = Depends(get_session)) -> dict:
    g = db.get(GoldenSet, version)
    if g is None:
        raise HTTPException(404, "Golden set not found")
    return {"version": g.version, "created_at": g.created_at.isoformat(), "examples": g.examples}


@router.post("/golden-sets", response_model=GoldenSetOut, status_code=201)
def publish_golden_set(user: AppUser = Depends(require_admin), db: Session = Depends(get_session)):
    """Freeze every approved example into the next immutable version. Admins only."""
    approved = db.scalars(select(GoldenExample).where(GoldenExample.status == "approved")
                          .order_by(GoldenExample.created_at)).all()
    if not approved:
        raise HTTPException(422, "No approved examples to publish")
    version = (db.scalar(select(func.max(GoldenSet.version))) or 0) + 1
    labels = _labels(db)
    frozen = [{"id": str(e.id), "kind": e.kind, "safety": e.safety, "machine": labels.get(e.machine_line_id),
               "stars": e.stars, "comment": e.comment, "reference_answer": e.reference_answer,
               "rubric_notes": e.rubric_notes, "failure_cause": e.failure_cause, "snapshot": e.snapshot}
              for e in approved]
    g = GoldenSet(version=version, created_by=user.id, example_count=len(frozen), examples=frozen)
    db.add(g)
    db.commit()
    db.refresh(g)
    return GoldenSetOut(version=g.version, example_count=g.example_count, created_at=g.created_at)
