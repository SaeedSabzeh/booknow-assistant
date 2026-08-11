.PHONY: install test lint fmt typecheck run app clean

install:
	pip install -e ".[ui,dev]"

test:
	pytest --cov=booknow --cov-report=term-missing

lint:
	ruff check .

fmt:
	ruff check --fix . && ruff format .

typecheck:
	mypy src/booknow

run:
	python -m booknow.cli -v

app:
	python -m booknow.app

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage dist build
	find . -name __pycache__ -type d -exec rm -rf {} +
