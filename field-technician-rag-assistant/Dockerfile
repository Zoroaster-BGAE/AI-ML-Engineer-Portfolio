# Production image: the built SPA and the API in one container.
#
# This exists alongside backend/Dockerfile rather than replacing it. That one
# stays exactly as it was for `docker compose up`, where the frontend runs as a
# separate Vite dev server with hot reload -- nothing about local development
# changes. This one is only what CI builds and the deployment runs.
#
# Why they are bundled at all: with no separate static host or CDN in front, the
# compiled frontend files have to be served by the API container itself.
#
# The consequence to know about: a frontend change rebuilds this whole image and
# restarts the API. That is the accepted cost of the arrangement.
#
# Build from the repository root, not from backend/:
#     docker build -t assistant-backend .

# ---------------------------------------------------------------------------
# Stage 1 -- compile the SPA to static files
# ---------------------------------------------------------------------------
FROM node:22-alpine AS frontend

WORKDIR /build

# Lockfile first so a source-only change reuses the installed layer. `npm ci`
# rather than `npm install`: it installs exactly what the lockfile pins and
# fails if the two disagree, instead of quietly resolving something newer.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
# `npm run build` is `tsc --noEmit && vite build`, so a type error fails the
# image build here rather than shipping a broken bundle.
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 -- the application
# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/models

WORKDIR /app

# Dependencies in their own layer. The editable install needs the package
# directory to exist at install time, hence the placeholder that is then
# removed -- the real app/ is copied in below.
COPY backend/pyproject.toml ./
RUN mkdir app && touch app/__init__.py \
    && pip install --no-cache-dir -e . \
    && rm -rf app

# Bake the embedding model into the image. The container has no persistent volume, so
# without this every task start re-downloads ~220MB from Hugging Face before it
# can answer anything -- slow cold starts, and a deploy that fails outright if
# that download is unavailable. Doing it here makes the image bigger and the
# task start fast, which is the right trade for something that starts far more
# often than it is built.
#
# Keep the model name in step with `embed_model` in backend/app/config.py. If
# they drift, the build downloads one model and the running app downloads
# another at first request.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY backend/ ./

# The compiled SPA. app/main.py mounts this directory if it exists, so the same
# code runs unchanged locally where it does not.
COPY --from=frontend /build/dist ./static

# Builds DATABASE_URL from DB_HOST/PORT/NAME/USER + DB_PASSWORD, then runs
# migrations and starts the server. See the file for why the URL is assembled at
# start rather than stored anywhere.
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000

CMD ["docker-entrypoint.sh"]
