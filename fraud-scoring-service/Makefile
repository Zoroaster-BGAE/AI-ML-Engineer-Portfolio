IMAGE  ?= fraud-scoring-service
TAG    ?= 1.0.0
PORT   ?= 8888
MODEL  := models/fraud-model.pickle
SHA    = $(shell sha256sum $(MODEL) | cut -d' ' -f1)

.PHONY: help build run test lint typecheck golden load smoke clean demo-model

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

build: $(MODEL) ## Build the production image with the model digest baked in
	docker build --build-arg MODEL_SHA256=$(SHA) -t $(IMAGE):$(TAG) .

run: build ## Run the service on localhost:8888
	docker run --rm -p $(PORT):8888 --name $(IMAGE) $(IMAGE):$(TAG)

test: build ## Run the full suite inside the production image
	docker run --rm -v "$(PWD)/tests:/app/tests" -v "$(PWD)/pyproject.toml:/app/pyproject.toml" \
		-e PYTHONPATH=/app/src $(IMAGE):$(TAG) \
		sh -c "pip install -q pytest pytest-cov pytest-asyncio httpx && cd /app && python -m pytest"

golden: build ## Regenerate the golden fixture inside the production image
	docker run --rm -v "$(PWD)/tests:/app/tests" -v "$(PWD)/scripts:/app/scripts" \
		-e PYTHONPATH=/app/src $(IMAGE):$(TAG) python scripts/generate_golden.py

demo-model: ## Train the synthetic demo model in the pinned python:3.9 toolchain (writes models/ and the mock payloads)
	docker run --rm -v "$(PWD):/work" -w /work python:3.9-slim-bookworm \
		sh -c "pip install -q -r requirements.txt && python scripts/train_demo_model.py"

lint: ## PEP8 + import order + security rules
	ruff check src tests scripts && black --check src tests scripts

typecheck: ## mypy --strict
	mypy src

smoke: ## Hit the running service with the demo payload
	curl -s -X POST http://localhost:$(PORT)/fraud-score \
		-H 'Content-Type: application/json' \
		-d @mock-request-body.json | python -m json.tool

load: ## Load test the running service (see tests/load/README.md)
	locust -f tests/load/locustfile.py --headless -u 100 -r 20 -t 60s \
		--host http://localhost:$(PORT)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

$(MODEL):
	@echo "$(MODEL) not found - run 'make demo-model' first" >&2; exit 1
