"""Admin library view: clients -> machines/lines -> documents with tags and ingest status.
Retag only. There is no content viewer and no download (FR-18)."""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth.dependencies import require_admin
from app.db import get_session
from app.domain.enums import DocType
from app.domain.models import Chunk, Client, Document, MachineLine

router = APIRouter(prefix="/api/admin/library", tags=["admin"], dependencies=[Depends(require_admin)])


class DocOut(BaseModel):
    id: UUID
    title: str
    doc_type: str
    component: str | None
    languages: list[str]
    page_count: int | None
    ingest_status: str
    ingest_note: str | None
    chunk_count: int


class MachineTree(BaseModel):
    id: UUID
    label: str
    kind: str
    documents: list[DocOut]


class ClientTree(BaseModel):
    id: UUID
    name: str
    slug: str
    machines: list[MachineTree]


class Retag(BaseModel):
    doc_type: DocType | None = None
    component: str | None = None
    languages: list[str] | None = None


@router.get("", response_model=list[ClientTree])
def library(db: Session = Depends(get_session)):
    counts = dict(db.execute(select(Chunk.document_id, func.count()).group_by(Chunk.document_id)).all())
    out = []
    for c in db.scalars(select(Client).order_by(Client.name)):
        machines = []
        for m in db.scalars(select(MachineLine).where(MachineLine.client_id == c.id).order_by(MachineLine.label)):
            docs = db.scalars(select(Document).where(Document.machine_line_id == m.id)
                              .order_by(Document.component.nulls_first(), Document.title)).all()
            machines.append(MachineTree(id=m.id, label=m.label, kind=m.kind, documents=[
                DocOut(id=d.id, title=d.title, doc_type=d.doc_type, component=d.component, languages=d.languages or [],
                       page_count=d.page_count, ingest_status=d.ingest_status, ingest_note=d.ingest_note,
                       chunk_count=counts.get(d.id, 0)) for d in docs]))
        out.append(ClientTree(id=c.id, name=c.name, slug=c.slug, machines=machines))
    return out


@router.patch("/documents/{doc_id}", response_model=DocOut)
def retag(doc_id: UUID, body: Retag, db: Session = Depends(get_session)):
    d = db.get(Document, doc_id)
    if d is None:
        raise HTTPException(404, "Document not found")
    if body.doc_type:
        d.doc_type = body.doc_type.value
    if body.component is not None:
        d.component = body.component.strip() or None
    if body.languages is not None:
        d.languages = body.languages
    db.commit()
    n = db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == d.id))
    return DocOut(id=d.id, title=d.title, doc_type=d.doc_type, component=d.component, languages=d.languages or [],
                  page_count=d.page_count, ingest_status=d.ingest_status, ingest_note=d.ingest_note, chunk_count=n)
