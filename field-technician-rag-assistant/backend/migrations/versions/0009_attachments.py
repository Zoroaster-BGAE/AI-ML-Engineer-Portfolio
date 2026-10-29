"""Photos and voice notes sent with a message (the bytes are in S3 `events/`; this table indexes them)."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0009_attachments"
down_revision = "0008_unscoped_sessions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "attachment",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("chat_session.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("message_id", UUID(as_uuid=True), sa.ForeignKey("chat_message.id", ondelete="CASCADE")),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("app_user.id"), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("content_type", sa.Text(), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer()),
        sa.Column("height", sa.Integer()),
        sa.Column("stt_text", sa.Text()),
        sa.Column("language", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("kind IN ('image','audio')", name="ck_attachment_kind"),
    )
    op.create_index("ix_attachment_session_id", "attachment", ["session_id"])
    op.create_index("ix_attachment_message_id", "attachment", ["message_id"])


def downgrade() -> None:
    op.drop_table("attachment")
