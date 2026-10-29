"""POST /api/sessions/{id}/messages -> SSE events: status | scope | token | question | done.

The body is JSON ({text, option_index}) or multipart/form-data (text, option_index, up to 4 `images`, and an
`audio_id` from POST /audio). Scope comes from the session row; nothing in the payload can set it.

POST /api/sessions/{id}/audio stores and transcribes a voice note and returns the transcript, which the
technician confirms or corrects in the composer before sending it as the message text.
"""
import json
from collections.abc import Iterator
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError
from sqlalchemy.orm import Session
from sse_starlette.sse import EventSourceResponse
from starlette.datastructures import UploadFile as StarletteUpload

from app.agent.graph import Agent
from app.api.schemas import MessageIn
from app.auth.dependencies import get_current_user
from app.db import get_session
from app.deps import get_embedder_dep, get_llm
from app.domain.support import AppUser, Attachment, ChatMessage, ChatSession
from app.llm.base import ImageInput, LLMClient
from app.llm.embed import Embedder
from app.media import images as img
from app.media.storage import get_storage
from app.media.transcribe import AUDIO_TYPES, MAX_AUDIO_BYTES, TranscriptionFailed, get_transcriber

router = APIRouter(prefix="/api/sessions", tags=["chat"])

MAX_IMAGES = 4
PHOTO_ONLY = "(The technician sent a photo without text.)"


def _own_session(db: Session, session_id: UUID, user: AppUser) -> ChatSession:
    s = db.get(ChatSession, session_id)
    if s is None or s.user_id != user.id:
        raise HTTPException(404, "Session not found")
    return s


async def _read_body(request: Request) -> tuple[MessageIn, list[bytes], UUID | None]:
    ctype = request.headers.get("content-type", "")
    if not ctype.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
        try:
            return MessageIn.model_validate(await request.json()), [], None
        except (ValidationError, ValueError) as exc:
            raise HTTPException(422, "Invalid message") from exc
    form = await request.form(max_files=MAX_IMAGES + 1, max_fields=10)
    raw_images = []
    for f in form.getlist("images"):
        if isinstance(f, StarletteUpload):
            raw_images.append(await f.read(img.MAX_BYTES + 1))
    if len(raw_images) > MAX_IMAGES:
        raise HTTPException(422, f"At most {MAX_IMAGES} photos per message")
    typed = str(form.get("text") or "").strip()
    if not typed and raw_images:
        typed = PHOTO_ONLY
    try:
        opt = form.get("option_index")
        body = MessageIn(text=typed, option_index=int(opt) if opt not in (None, "") else None)
        audio = form.get("audio_id")
        audio_id = UUID(str(audio)) if audio else None
    except (ValidationError, ValueError) as exc:
        raise HTTPException(422, "Invalid message") from exc
    if body.option_index is not None and (raw_images or audio_id):
        raise HTTPException(422, "A chip answer cannot carry photos or a voice note")
    return body, raw_images, audio_id


@router.post("/{session_id}/messages")
async def post_message(session_id: UUID, request: Request, user: AppUser = Depends(get_current_user),
                       db: Session = Depends(get_session), llm: LLMClient = Depends(get_llm),
                       embedder: Embedder = Depends(get_embedder_dep)):
    _own_session(db, session_id, user)
    body, raw_images, audio_id = await _read_body(request)
    try:
        cleaned = [await run_in_threadpool(img.clean, r) for r in raw_images]
    except img.ImageRejected as exc:
        raise HTTPException(415, str(exc)) from exc

    user_message_id = None
    if cleaned or audio_id:
        # The message is stored here, not by the agent, because its attachments must point at it.
        audio = db.get(Attachment, audio_id) if audio_id else None
        if audio_id and (audio is None or audio.session_id != session_id or audio.user_id != user.id
                         or audio.kind != "audio" or audio.message_id is not None):
            raise HTTPException(422, "Unknown voice note")
        # A photo sent without words is stored as an empty message (the chat shows the photo); the agent is told.
        msg = ChatMessage(session_id=session_id, role="user", content="" if body.text == PHOTO_ONLY else body.text)
        db.add(msg)
        db.flush()
        storage = get_storage()
        for c in cleaned:
            key = f"events/{session_id}/{uuid4()}.jpg"
            await run_in_threadpool(storage.put, key, c.data, c.media_type)
            db.add(Attachment(session_id=session_id, message_id=msg.id, user_id=user.id, kind="image",
                              content_type=c.media_type, storage_key=key, size_bytes=len(c.data),
                              width=c.width, height=c.height))
        if audio is not None:
            audio.message_id = msg.id
        db.commit()
        user_message_id = msg.id
    images = [ImageInput(c.media_type, c.data) for c in cleaned]
    bind = db.get_bind()

    def stream() -> Iterator[dict]:
        # a dedicated DB session: the request-scoped one may be closed before the stream finishes
        with Session(bind, expire_on_commit=False) as work:
            sess, usr = work.get(ChatSession, session_id), work.get(AppUser, user.id)
            for ev in Agent(work, llm, embedder).run_turn(sess, usr, body.text, body.option_index, images=images,
                                                          user_message_id=user_message_id):
                yield {"event": ev["event"], "data": json.dumps(ev["data"], ensure_ascii=False)}

    return EventSourceResponse(stream(), headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sniff_audio(data: bytes) -> bool:
    head = data[:12]
    return (head.startswith(b"\x1a\x45\xdf\xa3") or head.startswith(b"OggS") or head[4:8] == b"ftyp"
            or head.startswith(b"ID3") or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
            or (head.startswith(b"RIFF") and head[8:12] == b"WAVE"))


@router.post("/{session_id}/audio")
async def post_audio(session_id: UUID, file: UploadFile = File(...), language: str | None = Form(None),
                     user: AppUser = Depends(get_current_user), db: Session = Depends(get_session)) -> dict:
    _own_session(db, session_id, user)
    ctype = (file.content_type or "").split(";")[0].strip().lower()
    ext = AUDIO_TYPES.get(ctype)
    data = await file.read(MAX_AUDIO_BYTES + 1)
    if ext is None or not _sniff_audio(data):
        raise HTTPException(415, "Unsupported audio type")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Voice note is larger than 10 MB")
    key = f"events/{session_id}/{uuid4()}.{ext}"
    storage = get_storage()
    await run_in_threadpool(storage.put, key, data, ctype)
    try:
        t = await run_in_threadpool(get_transcriber().transcribe, storage.uri(key), ext,
                                    language if language in ("el", "en") else None)
    except TranscriptionFailed as exc:
        raise HTTPException(502, "The voice note could not be transcribed. Please type your message.") from exc
    att = Attachment(session_id=session_id, user_id=user.id, kind="audio", content_type=ctype, storage_key=key,
                     size_bytes=len(data), stt_text=t.text, language=t.language)
    db.add(att)
    db.commit()
    return {"audio_id": str(att.id), "transcript": t.text, "language": t.language}
