# echokv — developer convenience targets
.PHONY: install dev test test-all lint format bench clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# CPU-friendly tests (skip gpu/slow). CI runs this.
test:
	pytest -q -m "not gpu and not slow"

# everything, including model-downloading and GPU tests
test-all:
	pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests

# one-command reproduction: JSON + figures on a small model
bench:
	python -m echokv.benchmarks

clean:
	rm -rf build dist *.egg-info src/echokv/__pycache__ tests/__pycache__ .pytest_cache echokv_bench
