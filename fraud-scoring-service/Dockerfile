# =============================================================================
# Fraud Scoring Service — production image for AWS ECS Fargate
#
# Multi-stage: the build stage owns compilers and pip caches; the runtime stage
# ships only a virtualenv, the source and the model artifact. Nothing that can
# compile code survives into the image that faces the network.
#
# Base is python:3.9-slim, not 3.11/3.12: scikit-learn 1.0.2 publishes
# manylinux wheels for cp38/cp39/cp310 only. On a newer interpreter pip would
# fall back to a source build against a modern Cython and fail.
# =============================================================================

# --- Stage 1: build ----------------------------------------------------------
FROM python:3.9-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# Present as a safety net only — every pinned dependency resolves to a
# prebuilt manylinux wheel, so nothing should actually compile here.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build
COPY requirements.txt .
# In CI this becomes `pip install --require-hashes -r requirements.lock`, so a
# compromised or re-uploaded wheel cannot enter the image.
RUN pip install --no-cache-dir --upgrade pip==24.0 \
 && pip install --no-cache-dir -r requirements.txt

# --- Stage 2: runtime --------------------------------------------------------
FROM python:3.9-slim-bookworm AS runtime

# -----------------------------------------------------------------------------
# THREAD PINNING — the single highest-value environment setting in this file.
#
# NumPy/SciPy link against OpenBLAS, which by default spawns one thread per
# visible core. On a 1-vCPU Fargate task that means several threads fighting
# over one core, and the resulting context-switch thrash lands squarely in the
# p99. For a 15-tree, depth-5 forest there is nothing to parallelise anyway:
# joblib's dispatch overhead exceeds the traversal cost. One worker, one
# thread, many containers.
# -----------------------------------------------------------------------------
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app/src

# Application defaults; every one is overridable by the ECS task definition.
ENV FRAUD_API_PORT=8888 \
    FRAUD_API_MODEL_PATH=/app/models/fraud-model.pickle \
    FRAUD_API_LOG_LEVEL=INFO \
    FRAUD_API_LOG_JSON=true \
    FRAUD_API_ENVIRONMENT=production

# Digest of the model artifact, injected by the build pipeline:
#   docker build --build-arg MODEL_SHA256=$(sha256sum models/fraud-model.pickle | cut -d' ' -f1)
# The loader refuses to unpickle anything else. Unpickling is arbitrary code
# execution — this is the control that makes shipping a .pickle acceptable.
ARG MODEL_SHA256=""
ENV FRAUD_API_MODEL_SHA256=${MODEL_SHA256}

# Non-root, no login shell, no home-directory writes needed at runtime.
RUN groupadd --gid 10001 appuser \
 && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/

USER appuser
EXPOSE 8888

# Container-level health check. ECS also has its own; the ALB target group
# health check is what actually gates traffic (see infra/). No curl in slim,
# so probe with the interpreter that is already here.
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8888/health/ready', timeout=2).status == 200 else 1)"

# --workers 1          : one worker per container; scale out with tasks, not
#                        with in-process workers. Cleaner CPU accounting on
#                        Fargate, finer autoscaling granularity, small OOM
#                        blast radius, no in-container head-of-line blocking.
# --limit-concurrency  : hard cap on in-flight requests. Beyond it uvicorn
#                        returns 503 immediately instead of building an
#                        unbounded queue that turns into a latency cliff.
# --timeout-keep-alive : must exceed the ALB idle timeout (default 60s), or the
#                        server closes connections the ALB still believes are
#                        open — the classic source of sporadic 502s.
# --no-access-log      : access logging is handled by CorrelationMiddleware as
#                        structured JSON; uvicorn's would be a duplicate write
#                        on the hot path.
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8888", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--limit-concurrency", "256", \
     "--timeout-keep-alive", "65", \
     "--no-access-log"]
