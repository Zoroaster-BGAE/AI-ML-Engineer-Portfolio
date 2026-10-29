"""The quality loop: star ratings, safety events, golden examples and sets, conversation summaries, and the
batch jobs that produce the summaries. Plus the reviewer flag on app_user.

Tables are written out here rather than built from the live models (as 0003/0004 do), so this revision keeps
producing the same schema however the models change later. `is_reviewer` is checked first for the reason
0005 explains: revision 0003 builds app_user from the live model, so a fresh database already has it.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0007_quality_loop"
down_revision = "0006_pinned_sessions"
branch_labels = None
depends_on = None

_uuid = dict(type_=UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()"))
_ts = dict(nullable=False, server_default=sa.func.now())


def _has_column(table: str, name: str) -> bool:
    return name in {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    if not _has_column("app_user", "is_reviewer"):
        op.add_column("app_user", sa.Column("is_reviewer", sa.Boolean(), nullable=False,
                                            server_default=sa.text("false")))

    op.create_table(
        "message_feedback",
        sa.Column("id", **_uuid),
        sa.Column("message_id", UUID(as_uuid=True), sa.ForeignKey("chat_message.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("chat_session.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("app_user.id"), nullable=False),
        sa.Column("stars", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), **_ts),
        sa.Column("updated_at", sa.DateTime(timezone=True), **_ts),
        sa.CheckConstraint("stars BETWEEN 1 AND 5", name="ck_message_feedback_stars"),
    )
    op.create_index("uq_message_feedback_user_message", "message_feedback", ["message_id", "user_id"], unique=True)
    op.create_index("ix_message_feedback_session_id", "message_feedback", ["session_id"])

    op.create_table(
        "safety_event",
        sa.Column("id", **_uuid),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("run_id", UUID(as_uuid=True)),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("chat_session.id", ondelete="SET NULL")),
        sa.Column("message_id", UUID(as_uuid=True), sa.ForeignKey("chat_message.id", ondelete="SET NULL")),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("app_user.id")),
        sa.Column("machine_line_id", UUID(as_uuid=True), sa.ForeignKey("machine_line.id")),
        sa.Column("question", sa.Text()),
        sa.Column("detail", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("resolved", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("resolved_by", UUID(as_uuid=True), sa.ForeignKey("app_user.id")),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        sa.Column("note", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), **_ts),
        sa.CheckConstraint("kind IN ('blocked','amended','tripwire','feedback')", name="ck_safety_event_kind"),
    )
    op.create_index("ix_safety_event_open", "safety_event", ["resolved", "created_at"])

    op.create_table(
        "golden_example",
        sa.Column("id", **_uuid),
        sa.Column("status", sa.Text(), nullable=False, server_default="candidate"),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("safety", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("source_message_id", UUID(as_uuid=True), sa.ForeignKey("chat_message.id", ondelete="SET NULL")),
        sa.Column("machine_line_id", UUID(as_uuid=True), sa.ForeignKey("machine_line.id")),
        sa.Column("stars", sa.Integer()),
        sa.Column("comment", sa.Text()),
        sa.Column("snapshot", JSONB(), nullable=False),
        sa.Column("reference_answer", sa.Text()),
        sa.Column("rubric_notes", sa.Text()),
        sa.Column("failure_cause", sa.Text()),
        sa.Column("reviewed_by", UUID(as_uuid=True), sa.ForeignKey("app_user.id")),
        sa.Column("reviewed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), **_ts),
        sa.Column("updated_at", sa.DateTime(timezone=True), **_ts),
        sa.CheckConstraint("status IN ('candidate','approved','rejected')", name="ck_golden_example_status"),
        sa.CheckConstraint("kind IN ('positive','hard')", name="ck_golden_example_kind"),
    )
    op.create_index("ix_golden_example_status", "golden_example", ["status", "created_at"])
    op.create_index("uq_golden_example_message", "golden_example", ["source_message_id"], unique=True,
                    postgresql_where=sa.text("source_message_id IS NOT NULL"))

    op.create_table(
        "golden_set",
        sa.Column("version", sa.Integer(), primary_key=True, autoincrement=False),
        sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("app_user.id")),
        sa.Column("example_count", sa.Integer(), nullable=False),
        sa.Column("examples", JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), **_ts),
    )

    op.create_table(
        "conversation_summary",
        sa.Column("id", **_uuid),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("chat_session.id", ondelete="SET NULL"),
                  unique=True),
        sa.Column("machine_line_id", UUID(as_uuid=True), sa.ForeignKey("machine_line.id")),
        sa.Column("stars_min", sa.Integer(), nullable=False),
        sa.Column("stars_max", sa.Integer(), nullable=False),
        sa.Column("summary", JSONB(), nullable=False),
        sa.Column("model", sa.Text()),
        sa.Column("prompt_version", sa.Text()),
        sa.Column("feedback_through", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), **_ts),
        sa.Column("updated_at", sa.DateTime(timezone=True), **_ts),
    )

    op.create_table(
        "llm_batch",
        sa.Column("id", **_uuid),
        sa.Column("provider_id", sa.Text(), nullable=False, unique=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False, server_default="submitted"),
        sa.Column("items", JSONB(), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("submitted_at", sa.DateTime(timezone=True), **_ts),
        sa.Column("collected_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("status IN ('submitted','collected','failed')", name="ck_llm_batch_status"),
    )


def downgrade() -> None:
    for t in ("llm_batch", "conversation_summary", "golden_set", "golden_example", "safety_event",
              "message_feedback"):
        op.drop_table(t)
    if _has_column("app_user", "is_reviewer"):
        op.drop_column("app_user", "is_reviewer")
