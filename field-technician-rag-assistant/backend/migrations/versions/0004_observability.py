"""agent_run, agent_step, retrieval_hit, retrieval_gap."""
from alembic import op

import app.domain.support  # noqa: F401
from app.domain.models import Base

revision = "0004_observability"
down_revision = "0003_sessions"
branch_labels = None
depends_on = None

_T = ["agent_run", "agent_step", "retrieval_hit", "retrieval_gap"]


def upgrade() -> None:
    Base.metadata.create_all(op.get_bind(), tables=[Base.metadata.tables[t] for t in _T])


def downgrade() -> None:
    for t in reversed(_T):
        op.drop_table(t)
