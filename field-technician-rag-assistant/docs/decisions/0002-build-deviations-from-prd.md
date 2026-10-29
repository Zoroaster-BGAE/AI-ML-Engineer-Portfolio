# ADR 0002: Where the MVP build deviates from the PRD, and why

Date: 2026-09-19 · Status: accepted · Driver: the product owner supplied a Claude API key and asked to build the
MVP around it; everything below follows from measurements taken while building.

| # | PRD said | Build does | Why |
|---|----------|------------|-----|
| 1 | LLM = local Ollama `aya-expanse:8b`; "no outbound calls" | **Claude API** (`claude-opus-5`, configurable via `CLAUDE_MODEL`) behind the same `LLMClient` protocol | Requested. Consequence: technician questions and retrieved manual excerpts now leave the machine for Anthropic. The PRD's local-only NFR no longer holds. |
| 2 | Embeddings = `bge-m3` on Ollama (1024-d) | **`paraphrase-multilingual-MiniLM-L12-v2`** in-process via fastembed/ONNX (384-d); Ollama stays as `EMBEDDER=ollama` | Claude has no embeddings API. `bge-m3` on this CPU-only machine measured ~0.7 s/chunk, i.e. ~10 h for ~50k chunks. The small model is ~5x faster, needs no extra service, and covers Greek/German/English. |
| 3 | Drawing pages are chunked and embedded | Pages with >= 300 line/rect objects become **one keyword-only chunk** (no embedding) | Label soup carries little semantic meaning and is the most expensive part to embed. Full-text plus document-level routing still finds them (PRD section 6, "sparse wiring text"). |
| 4 | `understand` and `route` are separate calls | **Merged into one model call**; `route` remains a code step that validates every proposed document id | Measured ~3-5 s per sequential structured call. PRD section 12 anticipated this merge "if measurements justify it". |
| 5 | `retrieve` is an LLM-driven search/read_pages loop | Default is **deterministic**: hybrid search, then read the top hit's next page. The LLM controller still exists behind `RETRIEVAL_CONTROLLER=true` | The controller cost one extra ~5 s call per turn. The `read_pages` and budget rules (4 tool calls, 3 pages) apply either way. |
| 6 | Structured output "constrained by JSON schema" | **Every LLM-facing field is required and non-nullable**; no Python defaults | Measured against the live model: schemas with nullable or optional (defaulted) properties made constrained decoding loop and truncate in most calls (2-4 of 6 valid), while all-required schemas returned valid JSON 6 of 6. The SDK turns a default into "optional". |
| 8 | Up to 2 EXPAND loops | Default **1** (`MAX_EXPAND_LOOPS`) | On the live corpus the grader's "insufficient" verdict on broad questions rarely changed after a reformulation, and each loop costs ~12 s. The PRD says "at most 2". |
| 7 | Chunk text for interview ambiguity | Facets are computed from **search-ranked hits only**; pages fetched by `read_pages` never vote | Found by a failing scenario test: a score boost on a neighbour page made the top document look unambiguous and suppressed the interview. |
| 9 | Seeded local accounts, bcrypt hashes and app-issued HS256 tokens; the PRD chose Cognito for revocation | **Cognito is the identity provider.** The browser authenticates against the pool directly; the API only verifies the id token it signs, stores no password, and reads the role from `cognito:groups`. `app_user` is a projection created on first sign-in, and user administration writes to the pool. | The PRD's choice, implemented late -- what existed was scaffolding from building the agent. Two measured consequences. Verification is offline, so disabling an account cannot reach a signature already issued: token lifetime was cut from Cognito's default 60 minutes to **5**, and a disabled user is stopped at the refresh, which is a call Cognito checks. And `access_expires` stays local, because Cognito can disable an account but not schedule it -- a subscription end date has no equivalent in the pool. |

## Not done / next
- OpenTelemetry spans and the Phoenix `obs` profile (structlog JSON logs, Prometheus `/metrics` and the Postgres trace
  tables are in).
- `make eval` golden-question set and threshold calibration (M5). `GATE_MAX_DISTANCE` (0.62) and `GATE_MARGIN` (0.25)
  are unmeasured starting values.
- Claude server-side refusal fallbacks (`fallbacks`) are not enabled; a refusal ends the turn with a clear message.
- Optional latency levers, all measured but not applied: `CLAUDE_MODEL` on a smaller model, prompt caching of the
  per-machine catalog, running `assess` in parallel with the start of synthesis.
