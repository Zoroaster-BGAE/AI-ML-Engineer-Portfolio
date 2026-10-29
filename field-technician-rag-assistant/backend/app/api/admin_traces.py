"""Trace viewer + retrieval gaps. Admin-only: trace text includes prompts, reasoning and chunk text."""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.auth.dependencies import require_admin
from app.db import get_session
from app.domain.models import MachineLine
from app.domain.support import AgentRun, AgentStep, AppUser, RetrievalGap

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


def _run_row(r: AgentRun, email: str | None, label: str | None) -> dict:
    return {"id": str(r.id), "started_at": r.started_at, "final_action": r.final_action, "question": r.question,
            "user": email, "machine": label, "latency_ms": r.latency_ms, "llm_calls": r.llm_calls,
            "tokens_in": r.tokens_in, "tokens_out": r.tokens_out}


@router.get("/traces")
def list_runs(limit: int = Query(50, le=200), offset: int = 0, action: str | None = None,
              db: Session = Depends(get_session)):
    q = (select(AgentRun, AppUser.email, MachineLine.label).join(AppUser, AppUser.id == AgentRun.user_id)
         .outerjoin(MachineLine, MachineLine.id == AgentRun.machine_line_id)  # null while identifying the machine
         .order_by(AgentRun.started_at.desc()))
    if action:
        q = q.where(AgentRun.final_action == action.upper())
    return [_run_row(*row) for row in db.execute(q.limit(limit).offset(offset)).all()]


def _step(s: AgentStep) -> dict:
    return {"seq": s.seq, "node": s.node, "model": s.llm_model, "prompt_version": s.prompt_version,
            "latency_ms": s.latency_ms, "tokens_in": s.tokens_in, "tokens_out": s.tokens_out,
            "rationale": s.rationale, "reasoning": s.reasoning_raw, "input": s.input, "output": s.output,
            "error": s.error,
            "hits": [{"rank": h.rank, "doc": h.doc_label, "page": h.page, "vec": h.vector_distance, "fts": h.fts_rank,
                      "rrf": h.rrf_score, "selected": h.selected, "chunk_id": str(h.chunk_id)} for h in s.hits]}


@router.get("/traces/{run_id}")
def run_detail(run_id: UUID, db: Session = Depends(get_session)):
    r = db.scalar(select(AgentRun).where(AgentRun.id == run_id)
                  .options(selectinload(AgentRun.steps).selectinload(AgentStep.hits)))
    if r is None:
        raise HTTPException(404, "Run not found (it may be past the retention window)")
    user = db.get(AppUser, r.user_id)
    ml = db.get(MachineLine, r.machine_line_id) if r.machine_line_id else None
    return {**_run_row(r, user.email if user else None, ml.label if ml else None), "steps": [_step(s) for s in r.steps]}


@router.get("/gaps")
def gaps(resolved: bool | None = None, db: Session = Depends(get_session)):
    q = select(RetrievalGap, MachineLine.label).join(MachineLine, MachineLine.id == RetrievalGap.machine_line_id)
    if resolved is not None:
        q = q.where(RetrievalGap.resolved == resolved)
    return [{"id": str(g.id), "run_id": str(g.run_id) if g.run_id else None, "machine": label, "question": g.question,
             "missing": g.missing, "resolved": g.resolved, "created_at": g.created_at}
            for g, label in db.execute(q.order_by(RetrievalGap.created_at.desc()).limit(200)).all()]


@router.patch("/gaps/{gap_id}", status_code=204)
def resolve_gap(gap_id: UUID, resolved: bool = True, db: Session = Depends(get_session)):
    g = db.get(RetrievalGap, gap_id)
    if g is None:
        raise HTTPException(404, "Gap not found")
    g.resolved = resolved
    db.commit()
