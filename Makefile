.PHONY: install-dev format lint typecheck test test-cov security

install-dev:
	python -m pip install --upgrade pip
	python -m pip install -r requirements-dev.txt

format:
	ruff check --fix tests
	ruff format tests

lint:
	ruff check tests
	ruff format --check tests

typecheck:
	mypy

test:
	pytest

test-cov:
	pytest --cov=tests --cov-report=term-missing --cov-report=xml

security:
	bandit -q -r tests -s B101
	pip-audit -r requirements-dev.txt
