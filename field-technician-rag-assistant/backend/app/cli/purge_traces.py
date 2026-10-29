"""Delete trace rows older than the retention window (ADR 0001 / D6: 30 days, TRACE_RETENTION_DAYS).
agent_step / retrieval_hit go with their run (cascade). retrieval_gap rows are deliberately kept."""
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.config import settings
from app.domain.support import AgentRun


def purge(db: Session, days: int | None = None) -> int:
    cutoff = datetime.now(UTC) - timedelta(days=settings.trace_retention_days if days is None else days)
    n = db.execute(delete(AgentRun).where(AgentRun.started_at < cutoff)).rowcount
    db.commit()
    return n


if __name__ == "__main__":
    from app.db import SessionLocal

    with SessionLocal() as s:
        print(f"purged {purge(s)} runs")
