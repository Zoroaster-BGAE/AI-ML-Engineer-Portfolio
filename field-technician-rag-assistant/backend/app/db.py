from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=True)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one DB session per request."""
    with SessionLocal() as session:
        yield session
