"""Conversation retention (design §5.1), run nightly. Measured from a chat's last activity (`updated_at`).

- unrated chats: 90 days; pinned ones are exempt until the 12-month limit
- rated chats (any star rating), and pinned chats: 12 months
Deleting a chat cascades in the database to its messages, ratings and agent runs. Summaries, golden examples,
safety events and retrieval gaps are kept: they are detached (SET NULL) or hold no foreign key at all.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.config import settings
from app.domain.support import ChatSession


def apply(db: Session, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    rated = ChatSession.rating.is_not(None)
    unrated = db.execute(delete(ChatSession).where(
        ChatSession.updated_at < now - timedelta(days=settings.retention_unrated_days),
        ChatSession.pinned.is_(False), ChatSession.rating.is_(None))).rowcount
    kept_longer = db.execute(delete(ChatSession).where(
        ChatSession.updated_at < now - timedelta(days=settings.retention_rated_days),
        rated | ChatSession.pinned.is_(True))).rowcount
    db.commit()
    return {"unrated_deleted": unrated, "rated_or_pinned_deleted": kept_longer}
