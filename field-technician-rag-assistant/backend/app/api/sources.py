"""GET /api/messages/{id}/sources/{n}: the exact manual passage behind reference [n] of an answer.

Readable by the chat's owner, and by admins and flagged reviewers verifying an answer. The `customer` role is
refused (D-B: client roles get no library access until it is defined). Only chunks the answer cited are
served -- the ids come from the message the server wrote, never from the request.
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import SourceExcerptOut
from app.auth.dependencies import get_current_user
from app.db import get_session
from app.domain.models import Chunk
from app.domain.support import AppUser, ChatMessage, ChatSession

router = APIRouter(prefix="/api/messages", tags=["sources"])


@router.get("/{message_id}/sources/{n}", response_model=SourceExcerptOut)
def source_excerpt(message_id: UUID, n: int, user: AppUser = Depends(get_current_user),
                   db: Session = Depends(get_session)) -> SourceExcerptOut:
    if user.role == "customer":
        raise HTTPException(403, "Manual excerpts are not available for this account")
    m = db.get(ChatMessage, message_id)
    s = db.get(ChatSession, m.session_id) if m else None
    if m is None or s is None or (s.user_id != user.id and not user.can_review):
        raise HTTPException(404, "Message not found")
    src = next((x for x in (m.meta or {}).get("sources") or [] if x.get("n") == n), None)
    if src is None or not src.get("chunk_ids"):
        raise HTTPException(404, "This reference has no stored excerpt")
    ids = [UUID(c) for c in src["chunk_ids"]]
    rows = {c.id: c for c in db.scalars(select(Chunk).where(Chunk.id.in_(ids)))}
    excerpts = [rows[i].content for i in ids if i in rows]
    if not excerpts:  # the document was re-indexed since the answer was written
        raise HTTPException(404, "The manual has been updated since this answer; the passage is no longer stored")
    return SourceExcerptOut(n=n, title=src["title"], doc_type=src.get("doc_type"), page=src["page"],
                            section=src.get("section"), excerpts=excerpts, figure_id=src.get("figure_id"))
