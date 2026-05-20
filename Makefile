.PHONY: install build publish lint format clean test

install:
	uv sync

build:
	uv build

publish: build
	uv publish

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

clean:
	rm -rf dist/ .ruff_cache/ .pytest_cache/ .venv/
