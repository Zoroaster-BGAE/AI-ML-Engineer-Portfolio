"""Feeds the Manual Guide Choice modal. Labels and counts only: no document titles, no content."""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from app.api.schemas import ClientOut, MachineOut
from app.auth.dependencies import get_current_user
from app.db import get_session
from app.domain.enums import IngestStatus
from app.domain.models import Client, Document, MachineLine

router = APIRouter(prefix="/api/clients", tags=["catalog"], dependencies=[Depends(get_current_user)])


@router.get("", response_model=list[ClientOut])
def clients(db: Session = Depends(get_session)):
    return list(db.scalars(select(Client).where(Client.is_active).order_by(Client.name)))


@router.get("/{client_id}/machines", response_model=list[MachineOut])
def machines(client_id: UUID, db: Session = Depends(get_session)):
    client = db.get(Client, client_id)
    if client is None or not client.is_active:
        raise HTTPException(404, "Unknown client")
    indexed = Document.ingest_status == IngestStatus.indexed
    rows = db.execute(
        select(MachineLine, func.count(Document.id).filter(indexed), func.count(distinct(Document.component)).filter(indexed))
        .outerjoin(Document, Document.machine_line_id == MachineLine.id)
        .where(MachineLine.client_id == client_id).group_by(MachineLine.id).order_by(MachineLine.label)).all()
    return [MachineOut(id=m.id, label=m.label, kind=m.kind, year=m.year, document_count=n, component_count=c)
            for m, n, c in rows]
