PYTHON = python
KATALOGI = src/ tests/

.PHONY: install data lint format test tune train clean all

install:
	uv sync

data:
	uv run dvc pull

lint:
	uv run ruff check $(KATALOGI)

format:
	uv run ruff format $(KATALOGI)
	uv run ruff check --fix $(KATALOGI)

test:
	uv run pytest tests/ -v

tune: data
	uv run -m mlops_projekt.tune

train: data
	uv run -m mlops_projekt.train

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

serve:
	uv run uvicorn mlops_projekt.api:app --host 127.0.0.1 --port 8000 --reload

all: lint test train