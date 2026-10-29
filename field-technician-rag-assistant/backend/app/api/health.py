import httpx
from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_session

router = APIRouter(tags=["ops"])


@router.get("/healthz")
def healthz():
    return {"status": "ok"}


@router.get("/readyz")
def readyz(response: Response, db: Session = Depends(get_session)):
    checks: dict[str, str] = {}
    try:
        db.execute(text("select 1"))
        checks["postgres"] = "ok"
        ext = db.scalar(text("select extversion from pg_extension where extname='vector'"))
        checks["pgvector"] = f"ok ({ext})" if ext else "missing extension"
    except Exception as exc:  # noqa: BLE001
        checks["postgres"] = f"error: {type(exc).__name__}"
    checks["claude_api_key"] = "configured" if settings.claude_api_key.get_secret_value() else "MISSING"
    if settings.embedder == "ollama":
        try:
            tags = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3).json()
            names = [m.get("name", "") for m in tags.get("models", [])]
            checks["embeddings"] = "ok" if any(n.startswith(settings.embed_model) for n in names) \
                else f"model {settings.embed_model} not pulled"
        except Exception:  # noqa: BLE001
            checks["embeddings"] = "ollama unreachable"
    else:
        checks["embeddings"] = f"ok ({settings.embedder}: {settings.embed_model})" if settings.embedder == "fastembed"             else f"{settings.embedder} (test embedder)"
    ready = all(v.startswith(("ok", "configured", "hash")) for v in checks.values())
    response.status_code = 200 if ready else 503
    return {"ready": ready, "checks": checks}


@router.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
