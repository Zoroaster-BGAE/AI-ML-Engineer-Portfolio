"""Ratings move from single answers to the whole conversation (1-5 stars and an optional comment on
chat_session). Golden examples nominated by a rating now point at their conversation.

Per-answer ratings existed only in development builds (0007 never reached production); any that exist are
carried over as their conversation's rating (the latest one wins) before message_feedback is dropped.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0011_session_rating"
down_revision = "0010_figures"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    # Checked first: revision 0003 builds chat_session from the live model (see 0005), so a fresh database
    # already has these columns.
    have = _columns("chat_session")
    for name, type_ in (("rating", sa.SmallInteger()), ("rating_comment", sa.Text()),
                        ("rated_at", sa.DateTime(timezone=True))):
        if name not in have:
            op.add_column("chat_session", sa.Column(name, type_))
    op.create_check_constraint("ck_chat_session_rating", "chat_session", "rating BETWEEN 1 AND 5")

    op.add_column("golden_example", sa.Column("session_id", UUID(as_uuid=True),
                                              sa.ForeignKey("chat_session.id", ondelete="SET NULL")))
    op.create_index("uq_golden_example_session", "golden_example", ["session_id"], unique=True,
                    postgresql_where=sa.text("session_id IS NOT NULL AND source_message_id IS NULL"))

    op.execute("""
        UPDATE chat_session s SET rating = f.stars, rating_comment = f.comment, rated_at = f.updated_at
        FROM (SELECT DISTINCT ON (session_id) session_id, stars, comment, updated_at
              FROM message_feedback ORDER BY session_id, updated_at DESC) f
        WHERE f.session_id = s.id
    """)
    op.drop_table("message_feedback")


def downgrade() -> None:
    op.create_table(
        "message_feedback",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("message_id", UUID(as_uuid=True), sa.ForeignKey("chat_message.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("chat_session.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("app_user.id"), nullable=False),
        sa.Column("stars", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("stars BETWEEN 1 AND 5", name="ck_message_feedback_stars"),
    )
    op.create_index("uq_message_feedback_user_message", "message_feedback", ["message_id", "user_id"], unique=True)
    op.create_index("ix_message_feedback_session_id", "message_feedback", ["session_id"])
    op.drop_index("uq_golden_example_session", "golden_example")
    op.drop_column("golden_example", "session_id")
    op.drop_constraint("ck_chat_session_rating", "chat_session", type_="check")
    for col in ("rated_at", "rating_comment", "rating"):
        op.drop_column("chat_session", col)
