# Field Technician AI Assistant. On Windows without `make`, run the commands on the right directly.
CLIENT ?= sample_corpus/Acme Laundry Services
PY = .venv/Scripts/python

up:        ## whole stack in Docker: postgres + backend + frontend  (http://localhost:5173)
	docker compose up --build

down:
	docker compose down

migrate:   ## alembic upgrade head (the backend container also does this on start)
	cd backend && $(PY) -m alembic upgrade head

sample-corpus:  ## generate the synthetic demo manuals into ./sample_corpus (needs: pip install reportlab)
	python scripts/make_sample_corpus.py --out sample_corpus

onboard:   ## index a client folder. DRY=1 plan only, ONLY=text one machine, REINDEX=1 re-parse (renders drawings)
	cd backend && $(PY) -m app.cli.onboard "../$(CLIENT)" $(if $(DRY),--dry-run,) $(if $(ONLY),--only "$(ONLY)",) $(if $(REINDEX),--reindex,)

test:      ## no Claude, no embedding model: FakeLLM + hash embedder; needs `docker compose up -d postgres`
	cd backend && $(PY) -m pytest -q

lint:
	cd backend && $(PY) -m ruff check app tests

purge-traces:  ## delete traces older than TRACE_RETENTION_DAYS (30). Gaps are kept.
	cd backend && $(PY) -m app.cli.purge_traces

nightly:   ## retention (90 d unrated / 12 mo rated), trace purge, summary batches. SKIP=1 skips the Claude step
	cd backend && $(PY) -m app.cli.nightly $(if $(SKIP),--skip-summaries,)

.PHONY: up down migrate sample-corpus onboard test lint purge-traces nightly
