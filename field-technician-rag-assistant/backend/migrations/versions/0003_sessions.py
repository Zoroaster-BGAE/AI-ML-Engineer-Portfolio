"""app_user, chat_session, chat_message."""
from alembic import op

import app.domain.support  # noqa: F401
from app.domain.models import Base

revision = "0003_sessions"
down_revision = "0002_chunks_fts"
branch_labels = None
depends_on = None

_T = ["app_user", "chat_session", "chat_message"]


def upgrade() -> None:
    Base.metadata.create_all(op.get_bind(), tables=[Base.metadata.tables[t] for t in _T])
    # FR-04: the scope of a session can never change once written.
    op.execute("""
        CREATE FUNCTION chat_session_scope_immutable() RETURNS trigger AS $$
        BEGIN
          IF NEW.client_id <> OLD.client_id OR NEW.machine_line_id <> OLD.machine_line_id THEN
            RAISE EXCEPTION 'chat_session scope is immutable';
          END IF;
          RETURN NEW;
        END $$ LANGUAGE plpgsql
    """)
    op.execute("""CREATE TRIGGER trg_chat_session_scope_immutable BEFORE UPDATE ON chat_session
                  FOR EACH ROW EXECUTE FUNCTION chat_session_scope_immutable()""")


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_chat_session_scope_immutable ON chat_session")
    op.execute("DROP FUNCTION IF EXISTS chat_session_scope_immutable()")
    for t in reversed(_T):
        op.drop_table(t)
