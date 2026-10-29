"""chunk table with pgvector embedding + a GENERATED tsvector column (raw SQL on purpose:
avoids the REGCONFIG cast crash that SQLAlchemy's Computed() hit in the earlier MVP)."""
from alembic import op

from app.domain.models import Base

revision = "0002_chunks_fts"
down_revision = "0001_hierarchy"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind, tables=[Base.metadata.tables["chunk"]])
    op.execute("ALTER TABLE chunk DROP COLUMN tsv")
    op.execute("ALTER TABLE chunk ADD COLUMN tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED")
    op.execute("CREATE INDEX ix_chunk_tsv ON chunk USING gin (tsv)")


def downgrade() -> None:
    op.drop_table("chunk")
