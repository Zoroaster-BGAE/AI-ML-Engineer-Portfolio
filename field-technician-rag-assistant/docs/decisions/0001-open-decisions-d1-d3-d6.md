# ADR 0001 — Open decisions D1, D2, D3, D6

Date: 2026-09-19 · Status: accepted (confirmed by the product owner)

| # | Decision | Outcome |
|---|----------|---------|
| D1 | Machine/Line granularity | First-level folder = machine/line. Sub-folders are a `document.component` tag. No `line_group` column. |
| D2 | Web-search fallback | Removed. Answers come only from the manuals; uncovered questions ESCALATE and write a `retrieval_gap`. No outbound network calls. |
| D3 | Citations | Technicians see title + page only. No excerpt, no file link. Citations must resolve to in-scope documents (FR-13). |
| D6 | Trace retention | 30 days, configurable (`TRACE_RETENTION_DAYS=30`). A purge job removes `agent_step` / `retrieval_hit` past the window; `retrieval_gap` rows are kept. |

Assumed, not yet confirmed: D4 (evolve the existing repo; for now code lives under `frontend/` and
`backend/` in this folder) and D5 (CLI-only onboarding, no browser upload).
