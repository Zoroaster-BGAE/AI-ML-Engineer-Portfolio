import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import (
    admin_library,
    admin_traces,
    admin_users,
    auth,
    catalog,
    chat,
    figures,
    health,
    media,
    rating,
    review,
    sessions,
    sources,
)
from app.config import settings
from app.obs.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    settings.assert_emulators_are_local()
    # No user seeding. Accounts live in the Cognito pool, and a row in app_user
    # appears the first time someone signs in -- see app/auth/dependencies.py.
    if not settings.cognito_configured:
        get_logger().warning("cognito_not_configured",
                             detail="COGNITO_USER_POOL_ID/COGNITO_CLIENT_ID unset; every request will 401")
    # load the embedding model now, off the request path, so the first technician query is not the one that pays
    threading.Thread(target=_warm_embedder, daemon=True).start()
    yield


def _warm_embedder() -> None:
    try:
        from app.deps import get_embedder_dep

        get_embedder_dep().embed_query("warm up")
    except Exception as exc:  # noqa: BLE001 - /readyz reports embedder problems
        get_logger().warning("embedder_warmup_failed", error=str(exc)[:200])


def create_app() -> FastAPI:
    app = FastAPI(title="Field Technician AI Assistant", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins_list, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
    for r in (health.router, auth.router, catalog.router, sessions.router, chat.router, media.router, sources.router,
              figures.router,
              rating.router,
              review.router, admin_users.router, admin_library.router, admin_traces.router):
        app.include_router(r)
    _mount_spa(app)
    return app


def _mount_spa(app: FastAPI) -> None:
    """Serve the built frontend from this same container, if it is present.

    Only the production image has it: the root Dockerfile compiles the SPA and
    copies it to /app/static. Locally the directory does not exist and this is a
    no-op, because `docker compose` runs the frontend as a separate Vite dev
    server with hot reload.

    The frontend lives here rather than on a separate static host, so one
    container serves both the API and the single-page app.

    Two details that are easy to get wrong:

    * This runs *after* every router is registered. Mounted first, the catch-all
      at "/" swallows /api/*, /healthz, /readyz and /metrics, and the API
      silently starts returning index.html with a 200.
    * Client-side routes need _SPAStaticFiles below, not just html=True.
    """
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.is_dir():
        app.mount("/", _SPAStaticFiles(directory=static_dir, html=True), name="spa")


class _SPAStaticFiles(StaticFiles):
    """StaticFiles that serves index.html for client-side routes.

    `html=True` alone is not enough, and the gap is easy to miss: it serves
    index.html for *directories*, but a request for /sessions/42 matches no file
    and no directory, so it 404s. Every deep link and every page refresh away
    from "/" would break -- and only in the deployed image, since the Vite dev
    server handles this itself.

    The fallback is deliberately limited to paths with no file extension, which
    is the usual rule for single-page-app fallbacks. Falling
    back on *everything* would mean a missing /assets/index-abc.js returns
    index.html with a 200, and the browser reports it as a syntax error in a
    JavaScript file that is actually HTML -- a genuinely confusing hour. A
    missing asset must stay a 404.
    """

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404 and "." not in Path(path).name:
                return await super().get_response("index.html", scope)
            raise


app = create_app()
