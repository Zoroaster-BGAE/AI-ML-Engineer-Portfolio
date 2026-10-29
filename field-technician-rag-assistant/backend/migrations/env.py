from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

import app.domain.support  # noqa: F401  (registers the supporting tables on Base.metadata)
from app.config import settings
from app.domain.models import Base

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name, disable_existing_loggers=False)
target_metadata = Base.metadata


def run_migrations_online() -> None:
    engine = create_engine(settings.database_url)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
