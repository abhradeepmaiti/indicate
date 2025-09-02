.PHONY: help install install-dev test lint format typecheck clean all

help:
	@echo "Available commands:"
	@echo "  install      Install package"
	@echo "  install-dev  Install with development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run flake8 linting"
	@echo "  format       Format code with black and isort"
	@echo "  typecheck    Run mypy type checking"
	@echo "  clean        Clean build artifacts"
	@echo "  all          Run format, lint, typecheck, and test"

install:
	pip install .

install-dev:
	pip install -e ".[dev]"

test:
	python3 -m pytest indicate/tests/ -v

lint:
	python3 -m flake8 indicate/

format:
	python3 -m black indicate/
	python3 -m isort indicate/

typecheck:
	python3 -m mypy indicate/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all: format lint typecheck test