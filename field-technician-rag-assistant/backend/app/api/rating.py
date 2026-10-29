"""PUT /api/sessions/{id}/rating: 1-5 stars and an optional comment on the whole conversation.

Asked for when the technician finishes a chat or leaves it. A rating without a comment is complete. Saving
one keeps the golden-example candidate in step (a database write, no model call) and routes a comment that
talks about danger to the reviewers' safety queue.
"""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent.safety import mentions_safety
from app.api.schemas import SessionRatingOut
from app.auth.dependencies import get_current_user
from app.db import get_session
from app.domain.support import AppUser, ChatMessage, ChatSession, SafetyEvent
from app.quality.nominate import nominate_from_session

router = APIRouter(prefix="/api/sessions", tags=["rating"])

ANSWER_ACTIONS = ("ANSWER", "ESCALATE")


class RatingIn(BaseModel):
    stars: int = Field(ge=1, le=5)
    comment: str | None = Field(default=None, max_length=1000)


def has_answers(db: Session, session_id: UUID) -> bool:
    return db.scalar(select(func.count()).select_from(ChatMessage).where(
        ChatMessage.session_id == session_id, ChatMessage.role == "assistant",
        ChatMessage.meta["action"].astext.in_(ANSWER_ACTIONS))) > 0


@router.put("/{session_id}/rating", response_model=SessionRatingOut)
def rate_session(session_id: UUID, body: RatingIn, user: AppUser = Depends(get_current_user),
                 db: Session = Depends(get_session)) -> SessionRatingOut:
    s = db.get(ChatSession, session_id)
    if s is None or s.user_id != user.id:
        raise HTTPException(404, "Session not found")
    if not has_answers(db, s.id):
        raise HTTPException(422, "There is nothing to rate in this conversation yet")
    comment = (body.comment or "").strip() or None
    # The database clock, as for every other timestamp: the summary watermark compares against it.
    s.rating, s.rating_comment, s.rated_at = body.stars, comment, func.now()
    db.flush()
    db.refresh(s, attribute_names=["rated_at"])
    nominate_from_session(db, s)
    if mentions_safety(comment):
        last = db.scalar(select(ChatMessage).where(ChatMessage.session_id == s.id, ChatMessage.role == "assistant")
                         .order_by(ChatMessage.created_at.desc()).limit(1))
        db.add(SafetyEvent(kind="feedback", run_id=last.run_id if last else None, session_id=s.id,
                           message_id=last.id if last else None, user_id=user.id, machine_line_id=s.machine_line_id,
                           detail={"stars": body.stars, "comment": comment,
                                   "answer": (last.content if last else "")[:12000]}))
    db.commit()
    return SessionRatingOut(stars=s.rating, comment=s.rating_comment)
