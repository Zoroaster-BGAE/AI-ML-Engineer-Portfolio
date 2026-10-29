"""Let a technician pin a chat so it survives history pruning.

Pinning is not only a sort order. `create_session` deletes everything past
`history_max_sessions` to keep the sidebar finite, so without an exemption a
pinned chat would be promoted to the top of the list and then silently deleted by
the next few new chats -- which is worse than not offering pinning at all.

Not null with a false default, so every existing row is unpinned and no
backfill is needed.
"""
from alembic import op
import sqlalchemy as sa

revision = "0006_pinned_sessions"
down_revision = "0005_cognito_identity"
branch_labels = None
depends_on = None


def _has_column(name: str) -> bool:
    return name in {c["name"] for c in sa.inspect(op.get_bind()).get_columns("chat_session")}


def upgrade() -> None:
    # Checked for the same reason as revision 0005: the early revisions build
    # their tables from the live models, so a database created from scratch today
    # already has this column by the time this runs.
    if not _has_column("pinned"):
        op.add_column("chat_session",
                      sa.Column("pinned", sa.Boolean(), nullable=False, server_default=sa.text("false")))


def downgrade() -> None:
    if _has_column("pinned"):
        op.drop_column("chat_session", "pinned")
