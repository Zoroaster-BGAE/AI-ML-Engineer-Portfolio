"""Serve a technician's own photo back to them (thumbnails in the chat history). Reviewers and admins may see
any, for reviewing a conversation. Bytes are streamed through the API: no storage key or URL leaves the server."""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.db import get_session
from app.domain.support import AppUser, Attachment, ChatSession
from app.media.storage import get_storage

router = APIRouter(prefix="/api/attachments", tags=["media"])


@router.get("/{attachment_id}")
async def get_attachment(attachment_id: UUID, user: AppUser = Depends(get_current_user),
                         db: Session = Depends(get_session)) -> Response:
    a = db.get(Attachment, attachment_id)
    s = db.get(ChatSession, a.session_id) if a else None
    if a is None or s is None or a.kind != "image" or (s.user_id != user.id and not user.can_review):
        raise HTTPException(404, "Not found")
    data = await run_in_threadpool(get_storage().get, a.storage_key)
    return Response(data, media_type=a.content_type, headers={"Cache-Control": "private, max-age=3600"})
