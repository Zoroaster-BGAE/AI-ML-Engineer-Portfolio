"""GET /api/figures/{id}: a rendered drawing page, as it is (D-A). Technicians and admins may see the whole
library (D-D); the `customer` role may not (D-B). Bytes go through the API: no storage key leaves the server."""
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.db import get_session
from app.domain.models import Figure
from app.domain.support import AppUser
from app.media.storage import get_storage

router = APIRouter(prefix="/api/figures", tags=["figures"])


@router.get("/{figure_id}")
async def get_figure(figure_id: UUID, user: AppUser = Depends(get_current_user),
                     db: Session = Depends(get_session)) -> Response:
    if user.role == "customer":
        raise HTTPException(403, "Drawings are not available for this account")
    f = db.get(Figure, figure_id)
    if f is None:
        raise HTTPException(404, "Figure not found")
    data = await run_in_threadpool(get_storage().get, f.storage_key)
    return Response(data, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})
