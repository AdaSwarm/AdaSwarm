install:
	uv sync --extra examples
test:
	uv run pytest
lint:
	uv run ruff check adaswarm examples tests
format:
	uv run ruff format adaswarm examples tests
typecheck:
	uv run mypy adaswarm
run: install
	uv run python examples/quickstart.py
build:
	uv build
publish: build
	uv publish