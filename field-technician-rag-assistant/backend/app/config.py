"""Application settings. Values come from env vars, then from a local .env file.

The Claude key is read from `CLAUDE_API_KEY` (keep it in a git-ignored .env at the repo root). It is a
SecretStr, so it never shows up in logs, reprs or API responses.
"""
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[2]  # repo root (parent of backend/)
_LOCAL_DB_HOSTS = {"localhost", "127.0.0.1", "::1", "postgres"}  # "postgres" = the docker-compose service


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", str(_ROOT / ".env")),
        extra="ignore",
    )

    cors_origins: str = "http://localhost:5173"
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/assistant"

    # --- LLM: Claude API (reasoning, routing, grading, synthesis) ---
    claude_api_key: SecretStr = SecretStr("")
    claude_model: str = "claude-opus-5-5"
    claude_max_retries: int = 3
    claude_timeout_s: float = 120.0
    # A classifier decline is re-run on Anthropic's recommended fallback model instead of ending the turn.
    claude_fallbacks: bool = True
    # Up to 2.5x faster output at 2x price, used only where the technician is waiting on it (answer and
    # safety review). Research preview with its own rate limit: a 429 retries at standard speed.
    claude_fast_mode: bool = True
    # Effort per node, set explicitly: Opus 5.5 defaults to "medium" when unset.
    effort_nodes: str = "low"
    effort_synthesis: str = "low"
    effort_safety: str = "low"
    synthesis_max_tokens: int = 1500
    # The LLM retrieval controller costs one extra ~5 s call per turn, so it is off by default and replaced by a
    # deterministic read of the top hit's next page. Turn on to let the model request read_pages/search actions.
    retrieval_controller: bool = False

    # --- Embeddings: Claude has no embeddings API. Default = a small multilingual model run in-process on
    # CPU via fastembed/ONNX (Greek + German + English, ~5x faster than bge-m3 on CPU, no extra service).
    # Queries are also normalised to English technical terms by the `understand` node.
    embedder: str = "fastembed"  # fastembed | ollama | hash (hash = deterministic fake, for tests)
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_dim: int = 384
    ollama_base_url: str = "http://localhost:11434"  # only when EMBEDDER=ollama (then also set EMBED_MODEL/EMBED_DIM)

    # --- Retrieval / agent budgets (thresholds are calibrated by `make eval`) ---
    retrieval_candidate_k: int = 20
    rrf_k: int = 60
    context_chunks: int = 8
    max_tool_calls: int = 4
    read_pages_cap: int = 3
    max_expand_loops: int = 1   # PRD: at most 2. Measured: a 2nd loop rarely changes the grader verdict but costs ~12 s
    max_ask_turns: int = 2
    gate_max_distance: float = 0.62      # best cosine distance above this = library doesn't cover it
    gate_margin: float = 0.25            # top candidate's score share over the runner-up to skip an ASK
    verbatim_ngram: int = 14             # consecutive identical words that count as copying a manual

    # --- Auth ---
    # Cognito is the identity provider. The application issues no tokens and
    # stores no passwords; it verifies what Cognito signed. Both ids are public
    # -- the browser sends the client id in every login -- so they are plain
    # settings rather than secrets. The deployment sets them as environment variables.
    cognito_user_pool_id: str = ""
    cognito_client_id: str = ""
    cognito_region: str = "eu-central-1"
    # LOCAL DEVELOPMENT ONLY: a Cognito emulator (moto). `cognito_endpoint` is how this server reaches it
    # (signing keys, admin calls); `cognito_public_endpoint` is how the browser does (sign-in). Empty = real AWS.
    # The emulator issues tokens with the real AWS issuer, so only where the keys are fetched changes.
    cognito_endpoint: str = ""
    cognito_public_endpoint: str = ""

    @property
    def cognito_configured(self) -> bool:
        return bool(self.cognito_user_pool_id and self.cognito_client_id)

    @property
    def cognito_issuer(self) -> str:
        return f"https://cognito-idp.{self.cognito_region}.amazonaws.com/{self.cognito_user_pool_id}"

    @property
    def cognito_jwks_url(self) -> str:
        if self.cognito_endpoint:
            return f"{self.cognito_endpoint.rstrip('/')}/{self.cognito_user_pool_id}/.well-known/jwks.json"
        return f"{self.cognito_issuer}/.well-known/jwks.json"

    def assert_emulators_are_local(self) -> None:
        """Refuse to run against an emulator anywhere but a developer machine. An emulated identity provider
        accepts any password it was seeded with, so on a real deployment it would be an open door."""
        if not self.cognito_endpoint:
            return
        import os

        from sqlalchemy.engine import make_url

        db_host = make_url(self.database_url).host or ""
        if os.environ.get("ECS_CONTAINER_METADATA_URI_V4") or db_host not in _LOCAL_DB_HOSTS:
            raise RuntimeError("COGNITO_ENDPOINT points at an emulator; refusing to start outside local "
                               f"development (db host {db_host!r})")
    history_max_sessions: int = 30

    # --- Retention (design §5.1, D-N) ---
    trace_retention_days: int = 30  # ADR 0001 / D6
    retention_unrated_days: int = 90
    retention_rated_days: int = 365

    # --- Media (photos, voice, rendered drawing pages) ---
    # Photos and voice notes go to events/*; rendered drawing pages go to derived/*.
    media_store: str = "s3"             # s3 | local (tests)
    media_bucket: str = "example-documents-bucket"
    media_local_dir: str = "/tmp/assistant-media"
    transcriber: str = "aws"            # aws | fake (local development: the emulator has no Transcribe)

    # --- Corpus ---
    corpus_dir: str = "/corpus"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
