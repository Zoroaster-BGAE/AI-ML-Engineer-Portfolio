"""Supporting tables (PRD section 5): accounts, sessions, and the trace store. Not part of the hierarchy."""
from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    SmallInteger,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.domain.models import Base, _uuid_pk

_now = dict(server_default=func.now())


class AppUser(Base):
    __tablename__ = "app_user"
    __table_args__ = (CheckConstraint("role IN ('admin','technician','customer')", name="ck_app_user_role"),)

    id: Mapped[UUID] = mapped_column(**_uuid_pk)
    # The pool's stable id for this user. Nullable: rows that predate Cognito
    # are matched by email until their owner next signs in.
    cognito_sub: Mapped[str | None] = mapped_column(Text, unique=True)
    email: Mapped[str] = mapped_column(Text, unique=True)
    # No password is stored any more -- Cognito holds it. Kept nullable rather
    # than dropped so the column, and the migration, stay reversible.
    password_hash: Mapped[str | None] = mapped_column(Text)
    role: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"))
    # Vestigial: Cognito owns password lifecycle now. Kept rather than dropped so
    # this model and migration 0005 stay reversible; nothing reads it.
    must_change_password: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    access_expires: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Golden-example reviewer (design §5.4). Local, not a Cognito group: it is an app permission, and a
    # group would need an infra change. Hard and safety examples need a flagged reviewer to approve.
    is_reviewer: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), **_now)

    @property
    def can_review(self) -> bool:
        return self.role == "admin" or self.is_reviewer


class ChatSession(Base):
    """A chat may start without a machine: the Questioner agent identifies it and writes the scope once.
    From then on `machine_line_id` / `client_id` are immutable (FR-04, enforced by a trigger, migration 0008)."""

    __tablename__ = "chat_session"
    __table_args__ = (
        ForeignKeyConstraint(["machine_line_id", "client_id"], ["machine_line.id", "machine_line.client_id"]),
        Index("ix_chat_session_user", "user_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(**_uuid_pk)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("app_user.id"))
    client_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    machine_line_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    working_memory: Mapped[dict] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"))
    pending_question: Mapped[dict | None] = mapped_column(JSONB)
    title: Mapped[str | None] = mapped_column(Text)
    # Pinned chats sort first and are exempt from the 90-day expiry of unrated chats
    # (app.quality.retention): a technician pins the chat about a machine they will
    # come back to, and it has to still be there.
    pinned: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    # The technician's rating of the whole conversation (1-5, CHECK in migration 0011) and optional comment.
    rating: Mapped[int | None] = mapped_column(SmallInteger)
    rating_comment: Mapped[str | None] = mapped_column(Text)
    rated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), **_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), **_now)

    messages: Mapped[list[ChatMessage]] = relationship(back_populates="session", order_by="ChatMessage.created_at",
                                                       cascade="all, delete-orphan")


class ChatMessage(Base):
    __tablename__ = "chat_message"
    __table_args__ = (CheckConstraint("role IN ('user','assistant')", name="ck_chat_message_role"),)

    id: Mapped[UUID] = mapped_column(**_uuid_pk)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("chat_session.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict | None] = mapped_column(JSONB)   # assistant: {sources:[{title,page}], question:{...}}
    run_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), **_now)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class AgentRun(Base):
    __tablename__ = "agent_run"
    __table_args__ = (
        CheckConstraint("final_action IN ('ANSWER','ASK','ESCALATE','ERROR')", name="ck_agent_run_action"),
        Index("ix_agent_run_started", "started_at"),
    )

    id: Mapped[UUID] = mapped_column(**_uuid_pk)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("chat_session.id", ondelete="CASCADE"))
    message_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    user_id: Mapped[UUID] = mapped_column(ForeignKey("app_user.id"))
    # Null for a turn that ran before the machine was identified.
    client_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    machine_line_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    question: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), **_now)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    final_action: Mapped[str | None] = mapped_column(Text)
    llm_calls: Mapped[int] = mapped_column(Integer, default=0)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int | None] = mapped_column(Integer)

    steps: Mapped[list[AgentStep]] = relationship(back_populates="run", order_by="AgentStep.seq",
                                                  cascade="all, delete-orphan")


class AgentStep(Base):
    __tablename__ = "agent_step"

    id: Mapped[UUID] = mapped_column(**_uuid_pk)
    run_id: Mapped[UUID] = mapped_column(ForeignKey("agent_run.id", ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    node: Mapped[str] = mapped_column(Text)
    llm_model: Mapped[str | None] = mapped_column(Text)
    prompt_version: Mapped[str | None] = mapped_column(Text)
    input: Mapped[dict | None] = mapped_column(JSONB)
    output: Mapped[dict | None] = mapped_column(JSONB)
    rationale: Mapped[str | None] = mapped_column(Text)
    reasoning_raw: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    tokens_in: Mapped[int | None] = mapped_column(Integer)
    tokens_out: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), **_now)

    run: Mapped[AgentRun] = relationship(back_populates="steps")
    hits: Mapped[list[RetrievalHit]] = relationship(back_populates="step", order_by="RetrievalHit.rank",
                                                    cascade="all, delete-orphan")


class RetrievalHit(Base):
    __tablename__ = "retrieval_hit"

    id: Mapped[UUID] = mapped_column(**_uuid_pk)
    step_id: Mapped[UUID] = mapped_column(ForeignKey("agent_step.id", ondelete="CASCADE"), index=True)
    chunk_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True))
    document_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True))
    page: Mapped[int | None] = mapped_column(Integer)
    doc_label: Mapped[str | None] = mapped_column(Text)
    rank: Mapped[int] = mapped_column(Integer)
    vector_distance: Mapped[float | None] = mapped_column(Float)
    fts_rank: Mapped[float | None] = mapped_column(Float)
    rrf_score: Mapped[float | None] = mapped_column(Float)
    selected: Mapped[bool] = mapped_column(Boolean, default=False)

    step: Mapped[AgentStep] = relationship(back_populates="hits")


class RetrievalGap(Base):
    """Kept past the trace retention window: it is the admin's list of what the library fails to cover."""

    __tablename__ = "retrieval_gap"

    id: Mapped[UUID] = mapped_column(**_uuid_pk)
    run_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    machine_line_id: Mapped[UUID] = mapped_column(ForeignKey("machine_line.id"))
    question: Mapped[str] = mapped_column(Text)
    missing: Mapped[str | None] = mapped_column(Text)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), **_now)


class Attachment(Base):
    """A photo or voice note a technician sent. Bytes live in storage (S3 `events/`); this is the index.
    A voice note exists before its message does (it is transcribed, then confirmed), hence `message_id` is null
    until it is sent with one. `stt_text` is what Transcribe heard; the message text is what the technician sent."""

    __tablename__ = "attachment"
    __table_args__ = (CheckConstraint("kind IN ('image','audio')", name="ck_attachment_kind"),)

    id: Mapped[UUID] = mapped_column(**_uuid_pk)
    session_id: Mapped[UUID] = mapped_column(ForeignKey("chat_session.id", ondelete="CASCADE"), index=True)
    message_id: Mapped[UUID | None] = mapped_column(ForeignKey("chat_message.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("app_user.id"))
    kind: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(Text)
    storage_key: Mapped[str] = mapped_column(Text)   # private; never in an API schema
    size_bytes: Mapped[int] = mapped_column(Integer)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    stt_text: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), **_now)


# ---- quality loop (design §5): ratings, safety events, golden examples, summaries -----------------------------

class SafetyEvent(Base):
    """Something the lead technician should look at. Kept after the conversation is gone (no cascade)."""

    __tablename__ = "safety_event"
    __table_args__ = (
        CheckConstraint("kind IN ('blocked','amended','tripwire','feedback')", name="ck_safety_event_kind"),
        Index("ix_safety_event_open", "resolved", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(**_uuid_pk)
    kind: Mapped[str] = mapped_column(Text)
    run_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True))
    session_id: Mapped[UUID | None] = mapped_column(ForeignKey("chat_session.id", ondelete="SET NULL"))
    message_id: Mapped[UUID | None] = mapped_column(ForeignKey("chat_message.id", ondelete="SET NULL"))
    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("app_user.id"))
    machine_line_id: Mapped[UUID | None] = mapped_column(ForeignKey("machine_line.id"))
    question: Mapped[str | None] = mapped_column(Text)
    detail: Mapped[dict] = mapped_column(JSONB, default=dict, server_default=text("'{}'::jsonb"))
    resolved: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    resolved_by: Mapped[UUID | None] = mapped_column(ForeignKey("app_user.id"))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), **_now)


class GoldenExample(Base):
    """A snapshot, not a pointer: it must outlive the 90-day / 12-month conversation retention and the 30-day
    trace purge, so it copies what it needs (turns, answer, evidence references) at nomination time."""

    __tablename__ = "golden_example"
    __table_args__ = (
        CheckConstraint("status IN ('candidate','approved','rejected')", name="ck_golden_example_status"),
        CheckConstraint("kind IN ('positive','hard')", name="ck_golden_example_kind"),
        Index("ix_golden_example_status", "status", "created_at"),
        Index("uq_golden_example_message", "source_message_id", unique=True,
              postgresql_where=text("source_message_id IS NOT NULL")),
    )

    id: Mapped[UUID] = mapped_column(**_uuid_pk)
    status: Mapped[str] = mapped_column(Text, default="candidate", server_default="candidate")
    kind: Mapped[str] = mapped_column(Text)
    safety: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    # A rated conversation (session_id) or one blocked answer (source_message_id, safety cases).
    session_id: Mapped[UUID | None] = mapped_column(ForeignKey("chat_session.id", ondelete="SET NULL"))
    source_message_id: Mapped[UUID | None] = mapped_column(ForeignKey("chat_message.id", ondelete="SET NULL"))
    machine_line_id: Mapped[UUID | None] = mapped_column(ForeignKey("machine_line.id"))
    stars: Mapped[int | None] = mapped_column(Integer)
    comment: Mapped[str | None] = mapped_column(Text)
    snapshot: Mapped[dict] = mapped_column(JSONB)
    reference_answer: Mapped[str | None] = mapped_column(Text)
    rubric_notes: Mapped[str | None] = mapped_column(Text)
    failure_cause: Mapped[str | None] = mapped_column(Text)
    reviewed_by: Mapped[UUID | None] = mapped_column(ForeignKey("app_user.id"))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), **_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), **_now)


class GoldenSet(Base):
    """An immutable, versioned freeze of the approved examples; what an eval run names."""

    __tablename__ = "golden_set"

    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_by: Mapped[UUID | None] = mapped_column(ForeignKey("app_user.id"))
    example_count: Mapped[int] = mapped_column(Integer)
    examples: Mapped[list] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), **_now)


class ConversationSummary(Base):
    """Structured summary of a rated conversation. Kept indefinitely; its session is not (SET NULL)."""

    __tablename__ = "conversation_summary"

    id: Mapped[UUID] = mapped_column(**_uuid_pk)
    session_id: Mapped[UUID | None] = mapped_column(ForeignKey("chat_session.id", ondelete="SET NULL"),
                                                    unique=True)
    machine_line_id: Mapped[UUID | None] = mapped_column(ForeignKey("machine_line.id"))
    stars_min: Mapped[int] = mapped_column(Integer)
    stars_max: Mapped[int] = mapped_column(Integer)
    summary: Mapped[dict] = mapped_column(JSONB)
    model: Mapped[str | None] = mapped_column(Text)
    prompt_version: Mapped[str | None] = mapped_column(Text)
    feedback_through: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # latest rating summarised
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), **_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), **_now)


class LlmBatch(Base):
    """A Message Batches job submitted by the nightly run and collected by a later one."""

    __tablename__ = "llm_batch"
    __table_args__ = (CheckConstraint("status IN ('submitted','collected','failed')", name="ck_llm_batch_status"),)

    id: Mapped[UUID] = mapped_column(**_uuid_pk)
    provider_id: Mapped[str] = mapped_column(Text, unique=True)
    kind: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, default="submitted", server_default="submitted")
    items: Mapped[dict] = mapped_column(JSONB)  # custom_id -> what it is for (session id, feedback_through)
    error: Mapped[str | None] = mapped_column(Text)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), **_now)
    collected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
