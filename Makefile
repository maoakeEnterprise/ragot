.PHONY: install run debug clean lint

install:
	uv sync

run:
	uv run python -m src