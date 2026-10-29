"""Rendered drawing and scanned pages (the PNGs are in storage under derived/; this table indexes them)."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = "0010_figures"
down_revision = "0009_attachments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "figure",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", UUID(as_uuid=True), nullable=False),
        sa.Column("machine_line_id", UUID(as_uuid=True), nullable=False),
        sa.Column("client_id", UUID(as_uuid=True), nullable=False),
        sa.Column("page", sa.Integer(), nullable=False),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["document_id", "machine_line_id", "client_id"],
                                ["document.id", "document.machine_line_id", "document.client_id"],
                                ondelete="CASCADE"),
        sa.UniqueConstraint("document_id", "page"),
        sa.CheckConstraint("kind IN ('drawing','scanned')", name="ck_figure_kind"),
    )


def downgrade() -> None:
    op.drop_table("figure")
