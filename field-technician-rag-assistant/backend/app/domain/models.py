"""Three-tier hierarchy: Client -> MachineLine -> Document (+ Chunk as an index projection).

Composite FKs make an inconsistent hierarchy impossible (PRD section 5). D1 (locked): the first-level
folder is the machine/line; sub-folders are a `component` tag, not entities.
"""
from __future__ import annotations

from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    FetchedValue,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    SmallInteger,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.config import settings
from app.domain.enums import DocType, IngestStatus, MachineKind

EMBEDDING_DIM = settings.embed_dim  # must match the embedder; changing it needs a re-index


def _in(column: str, enum_cls) -> str:
    return f"{column} IN ({', '.join(repr(m.value) for m in enum_cls)})"


class Base(DeclarativeBase):
    pass


_uuid_pk = dict(type_=PGUUID(as_uuid=True), primary_key=True, default=uuid4,
                server_default=text("gen_random_uuid()"))


class Client(Base):
    __tablename__ = "client"

    id: Mapped[UUID] = mapped_column(**_uuid_pk)
    slug: Mapped[str] = mapped_column(Text, unique=True)      # ascii, stable: 'sunrise-laundry'
    name: Mapped[str] = mapped_column(Text)                   # verbatim folder name (Greek OK)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"))

    machine_lines: Mapped[list[MachineLine]] = relationship(back_populates="client")


class MachineLine(Base):
    __tablename__ = "machine_line"
    __table_args__ = (
        UniqueConstraint("client_id", "slug"),
        UniqueConstraint("id", "client_id"),                  # target of composite FKs
        CheckConstraint(_in("kind", MachineKind), name="ck_machine_line_kind"),
    )

    id: Mapped[UUID] = mapped_column(**_uuid_pk)
    client_id: Mapped[UUID] = mapped_column(ForeignKey("client.id"))
    slug: Mapped[str] = mapped_column(Text)
    label: Mapped[str] = mapped_column(Text)                  # 'FEEDER KLIQ PLUS 3300_3R0348 (2026)'
    kind: Mapped[str] = mapped_column(Text)
    model_label: Mapped[str | None] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(SmallInteger)
    serial_ref: Mapped[str | None] = mapped_column(Text)      # OPAQUE display text; never used for routing

    client: Mapped[Client] = relationship(back_populates="machine_lines")
    documents: Mapped[list[Document]] = relationship(back_populates="machine_line")


class Document(Base):
    __tablename__ = "document"
    __table_args__ = (
        ForeignKeyConstraint(["machine_line_id", "client_id"],
                             ["machine_line.id", "machine_line.client_id"]),
        UniqueConstraint("machine_line_id", "sha256"),        # idempotent onboarding
        UniqueConstraint("id", "machine_line_id", "client_id"),
        CheckConstraint(_in("doc_type", DocType), name="ck_document_doc_type"),
        CheckConstraint(_in("ingest_status", IngestStatus), name="ck_document_ingest_status"),
    )

    id: Mapped[UUID] = mapped_column(**_uuid_pk)
    client_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True))
    machine_line_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True))
    component: Mapped[str | None] = mapped_column(Text)      # sub-folder inside a line; NULL for machines
    title: Mapped[str] = mapped_column(Text)                  # display title, never the path
    doc_type: Mapped[str] = mapped_column(Text)
    languages: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, server_default=text("'{}'"))
    media_type: Mapped[str] = mapped_column(Text)
    storage_uri: Mapped[str] = mapped_column(Text)            # private; excluded from every API schema
    sha256: Mapped[str] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    outline: Mapped[list | None] = mapped_column(JSONB)       # [{id,title,level,page_from,page_to}] router catalog
    ingest_status: Mapped[str] = mapped_column(Text, default=IngestStatus.pending,
                                               server_default=IngestStatus.pending.value)
    ingest_note: Mapped[str | None] = mapped_column(Text)

    machine_line: Mapped[MachineLine] = relationship(back_populates="documents",
                                                     foreign_keys=[machine_line_id, client_id],
                                                     overlaps="documents")
    chunks: Mapped[list[Chunk]] = relationship(back_populates="document", cascade="all, delete-orphan",
                                               passive_deletes=True)


class Figure(Base):
    """A drawing or scanned page rendered to an image at ingest, shown to the technician as it is (D-A).
    The PNG lives in storage under derived/; this row says which page of which document it is."""

    __tablename__ = "figure"
    __table_args__ = (
        ForeignKeyConstraint(["document_id", "machine_line_id", "client_id"],
                             ["document.id", "document.machine_line_id", "document.client_id"],
                             ondelete="CASCADE"),
        UniqueConstraint("document_id", "page"),
        CheckConstraint("kind IN ('drawing','scanned')", name="ck_figure_kind"),
    )

    id: Mapped[UUID] = mapped_column(**_uuid_pk)
    document_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True))
    machine_line_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True))
    client_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True))
    page: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(Text)
    storage_key: Mapped[str] = mapped_column(Text)   # private; never in an API schema
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)


class Chunk(Base):
    """Index projection of Document (not a fourth tier). Only retrieval/repository.py may query it."""

    __tablename__ = "chunk"
    __table_args__ = (
        ForeignKeyConstraint(["document_id", "machine_line_id", "client_id"],
                             ["document.id", "document.machine_line_id", "document.client_id"],
                             ondelete="CASCADE"),
        Index("ix_chunk_scope_doc", "machine_line_id", "document_id"),
        Index("ix_chunk_tsv", "tsv", postgresql_using="gin"),
        Index("ix_chunk_embedding", "embedding", postgresql_using="hnsw",
              postgresql_ops={"embedding": "vector_cosine_ops"}),
    )

    id: Mapped[UUID] = mapped_column(**_uuid_pk)
    document_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True))
    machine_line_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True))
    client_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True))
    page: Mapped[int] = mapped_column(Integer)
    section_id: Mapped[str | None] = mapped_column(Text)
    section_path: Mapped[str | None] = mapped_column(Text)
    content_type: Mapped[str | None] = mapped_column(Text)  # procedure|alarm_list|table|wiring_sheet|safety|spec
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    # GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED lives in the Alembic migration as raw
    # SQL (sidesteps the REGCONFIG cast crash from the earlier MVP). FetchedValue keeps the ORM from
    # writing it; deferred keeps it out of default SELECTs.
    tsv: Mapped[str | None] = mapped_column(TSVECTOR, server_default=FetchedValue(), deferred=True)

    document: Mapped[Document] = relationship(back_populates="chunks",
                                              foreign_keys=[document_id, machine_line_id, client_id])
