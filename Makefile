PYTHON = python
KATALOGI = src/ tests/

.PHONY: install data lint format test tune train clean all

install:
    pip install -r requirements.txt

data:
    dvc pull

lint:
    ruff check $(KATALOGI)

format:
    ruff format $(KATALOGI)
    ruff check --fix $(KATALOGI)

test:
    pytest tests/ -v

tune: data
    $(PYTHON) -m src.tune

train: data
    $(PYTHON) -m src.train

clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    rm -rf .pytest_cache .ruff_cache

all: lint test train