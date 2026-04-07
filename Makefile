UV := uv run

lint:
	$(UV) ruff check src tests

format:
	$(UV) ruff format src tests

format-check:
	$(UV) ruff format --check src tests

typecheck:
	$(UV) mypy src

test:
	$(UV) pytest tests

test-unit:
	$(UV) pytest tests/unit

test-integration:
	$(UV) pytest tests --integration

check: lint typecheck test
