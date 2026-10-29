from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.api.rating import ANSWER_ACTIONS, has_answers
from app.api.schemas import (
    AttachmentOut,
    FigureRefOut,
    MessageOut,
    QuestionOut,
    SessionCreate,
    SessionDetailOut,
    SessionOut,
    SessionRatingOut,
    SourceOut,
)
from app.auth.dependencies import get_current_user
from app.config import settings
from app.db import get_session
from app.domain.models import Client, MachineLine
from app.domain.support import AppUser, Attachment, ChatMessage, ChatSession

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

_ANSWERED = exists().where(ChatMessage.session_id == ChatSession.id, ChatMessage.role == "assistant",
                           ChatMessage.meta["action"].astext.in_(ANSWER_ACTIONS))


def _out(s: ChatSession, ml: MachineLine | None, answered: bool = False) -> dict:
    return dict(id=s.id, title=s.title, machine_label=ml.label if ml else None, kind=ml.kind if ml else None,
                pinned=s.pinned, created_at=s.created_at, updated_at=s.updated_at, has_answers=answered,
                rating=SessionRatingOut(stars=s.rating, comment=s.rating_comment) if s.rating else None)




@router.post("", response_model=SessionOut, status_code=201)
def create_session(body: SessionCreate | None = None, user: AppUser = Depends(get_current_user),
                   db: Session = Depends(get_session)):
    """No body: an unscoped chat, and the Questioner agent finds the machine. With client + machine: FR-05, the
    pair is validated server-side and a machine that belongs to another client is a 422."""
    body = body or SessionCreate()
    # Nothing is deleted here any more. Creating a chat used to prune the user's oldest ones, which destroyed
    # rated conversations before they could be reviewed. Retention is now the nightly job's (app.quality.retention).
    if body.machine_line_id is None and body.client_id is None:
        # "New chat" pressed twice, or a reload with no chat open, must not pile up empty chats.
        latest = db.scalar(select(ChatSession).where(ChatSession.user_id == user.id)
                           .order_by(ChatSession.updated_at.desc()).limit(1))
        if latest is not None and latest.machine_line_id is None and not latest.messages:
            return _out(latest, None)
        s = ChatSession(user_id=user.id, working_memory={})
        db.add(s)
        db.commit()
        db.refresh(s)
        return _out(s, None)
    ml = db.scalar(select(MachineLine).where(MachineLine.id == body.machine_line_id,
                                             MachineLine.client_id == body.client_id))
    client = db.get(Client, body.client_id) if body.client_id else None
    if ml is None or client is None or not client.is_active:
        raise HTTPException(422, "The machine/line does not belong to the selected client")
    s = ChatSession(user_id=user.id, client_id=ml.client_id, machine_line_id=ml.id, working_memory={})
    db.add(s)
    db.commit()
    db.refresh(s)
    return _out(s, ml)


@router.get("", response_model=list[SessionOut])
def list_sessions(offset: int = Query(0, ge=0), user: AppUser = Depends(get_current_user),
                  db: Session = Depends(get_session)):
    """Pinned first, then most recent. A page of `history_max_sessions`; `offset` pages back to older chats."""
    rows = db.execute(select(ChatSession, MachineLine, _ANSWERED.label("answered"))
                      .outerjoin(MachineLine, MachineLine.id == ChatSession.machine_line_id)
                      .where(ChatSession.user_id == user.id)
                      .order_by(ChatSession.pinned.desc(), ChatSession.updated_at.desc())
                      .offset(offset).limit(settings.history_max_sessions)).all()
    return [_out(s, ml, answered) for s, ml, answered in rows]


class SessionPatch(BaseModel):
    pinned: bool


@router.patch("/{session_id}", response_model=SessionOut)
def patch_session(session_id: UUID, body: SessionPatch, user: AppUser = Depends(get_current_user),
                  db: Session = Depends(get_session)):
    """Pin or unpin. Deliberately does not touch updated_at -- pinning is not a
    visit, and bumping it would reorder the list under the user's cursor."""
    s = db.get(ChatSession, session_id)
    if s is None or s.user_id != user.id:
        raise HTTPException(404, "Session not found")
    s.pinned = body.pinned
    db.commit()
    db.refresh(s)
    return _out(s, db.get(MachineLine, s.machine_line_id) if s.machine_line_id else None, has_answers(db, s.id))


@router.get("/{session_id}", response_model=SessionDetailOut)
def get_session_detail(session_id: UUID, user: AppUser = Depends(get_current_user), db: Session = Depends(get_session)):
    s = db.get(ChatSession, session_id)
    if s is None or s.user_id != user.id:
        raise HTTPException(404, "Session not found")
    ml = db.get(MachineLine, s.machine_line_id) if s.machine_line_id else None
    client = db.get(Client, s.client_id) if s.client_id else None
    attached: dict = {}
    for a in db.scalars(select(Attachment).where(Attachment.session_id == s.id, Attachment.message_id.is_not(None))
                        .order_by(Attachment.created_at)):
        attached.setdefault(a.message_id, []).append(AttachmentOut(id=a.id, kind=a.kind))
    msgs = [MessageOut(id=m.id, role=m.role, text=m.content, created_at=m.created_at,
                       sources=[SourceOut(**x) for x in ((m.meta or {}).get("sources") or [])],
                       attachments=attached.get(m.id, []),
                       figures=[FigureRefOut(**f) for f in ((m.meta or {}).get("figures") or [])]) for m in s.messages]
    pq = s.pending_question
    pending = QuestionOut(question=pq["question"], options=[{"label": o["label"]} for o in pq["options"]]) if pq else None
    answered = any(m.role == "assistant" and (m.meta or {}).get("action") in ANSWER_ACTIONS for m in s.messages)
    return SessionDetailOut(**_out(s, ml, answered), client_name=client.name if client else None, messages=msgs,
                            pending_question=pending)
