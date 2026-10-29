"""Chats can start without a machine; the Questioner agent sets the scope once.

- chat_session.client_id / machine_line_id become nullable, and must be both set or both empty.
- The immutability trigger now allows exactly one change: empty -> set. After that the scope is fixed, as before.
  The old trigger compared with `<>`, which is NULL (not true) when either side is NULL -- it would have let a
  scope be cleared again. IS DISTINCT FROM closes that.
- agent_run's scope columns become nullable, so an identification turn is traced like any other.
"""
import sqlalchemy as sa
from alembic import op

revision = "0008_unscoped_sessions"
down_revision = "0007_quality_loop"
branch_labels = None
depends_on = None

_LOCK_ONCE = """
CREATE OR REPLACE FUNCTION chat_session_scope_immutable() RETURNS trigger AS $$
BEGIN
  IF OLD.machine_line_id IS NOT NULL AND (NEW.machine_line_id IS DISTINCT FROM OLD.machine_line_id
                                          OR NEW.client_id IS DISTINCT FROM OLD.client_id) THEN
    RAISE EXCEPTION 'chat_session scope is immutable once set';
  END IF;
  RETURN NEW;
END $$ LANGUAGE plpgsql
"""

_ORIGINAL = """
CREATE OR REPLACE FUNCTION chat_session_scope_immutable() RETURNS trigger AS $$
BEGIN
  IF NEW.client_id <> OLD.client_id OR NEW.machine_line_id <> OLD.machine_line_id THEN
    RAISE EXCEPTION 'chat_session scope is immutable';
  END IF;
  RETURN NEW;
END $$ LANGUAGE plpgsql
"""


def _has_check(table: str, name: str) -> bool:
    return name in {c["name"] for c in sa.inspect(op.get_bind()).get_check_constraints(table)}


def upgrade() -> None:
    for col in ("client_id", "machine_line_id"):
        op.alter_column("chat_session", col, nullable=True)
        op.alter_column("agent_run", col, nullable=True)
    if not _has_check("chat_session", "ck_chat_session_scope_pair"):
        op.create_check_constraint("ck_chat_session_scope_pair", "chat_session",
                                   "(client_id IS NULL) = (machine_line_id IS NULL)")
    op.execute(_LOCK_ONCE)


def downgrade() -> None:
    op.execute(_ORIGINAL)
    op.drop_constraint("ck_chat_session_scope_pair", "chat_session", type_="check")
    # Chats that never identified a machine cannot exist in the old schema.
    op.execute("DELETE FROM agent_run WHERE machine_line_id IS NULL")
    op.execute("DELETE FROM chat_session WHERE machine_line_id IS NULL")
    for col in ("client_id", "machine_line_id"):
        op.alter_column("agent_run", col, nullable=False)
        op.alter_column("chat_session", col, nullable=False)
