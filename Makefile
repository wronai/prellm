# PromptGuard Makefile
.PHONY: help install install-dev test lint format clean build publish run demo init

# Default target
help:
	@echo "PromptGuard - LLM prompt middleware"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting (ruff)"
	@echo "  format       Format code (ruff)"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build package"
	@echo "  publish      Publish to PyPI"
	@echo "  run          Run CLI example"
	@echo "  demo         Run demo with sample config"
	@echo "  init         Generate starter config"
	@echo ""

# Installation
install:
	poetry install --only main

install-dev:
	poetry install

# Quality checks
test:
	poetry run pytest tests/ -v

test-cov:
	poetry run pytest tests/ -v --cov=promptguard --cov-report=html --cov-report=term

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

# Cleanup
clean:
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build and publish
build: clean
	poetry build

publish: build
	poetry publish

publish-test: build
	poetry publish --repository testpypi

# Demo and examples
run:
	poetry run promptguard run "deploy to production" --dry-run

demo:
	@echo "Generating demo config..."
	poetry run promptguard init --devops --output demo-rules.yaml
	@echo ""
	@echo "Running demo analysis..."
	poetry run promptguard analyze "zawsze restartuj serwer" --config demo-rules.yaml
	@echo ""
	@echo "Running demo process chain..."
	poetry run promptguard process configs/deploy.yaml --dry-run --guard-config demo-rules.yaml || echo "Process chain config not found, skipping..."

init:
	poetry run promptguard init --devops

# Development helpers
dev-setup: install-dev
	@echo "Setting up pre-commit hooks..."
	@echo "#!/bin/bash" > .git/hooks/pre-commit
	@echo "poetry run ruff check ." >> .git/hooks/pre-commit
	@echo "poetry run pytest tests/ -q" >> .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "Pre-commit hooks installed!"

docs:
	@echo "Generating documentation..."
	@echo "# PromptGuard Documentation" > DOCS.md
	@echo "" >> DOCS.md
	@echo "## Installation" >> DOCS.md
	@echo '```bash' >> DOCS.md
	@echo "pip install promptguard" >> DOCS.md
	@echo '```' >> DOCS.md
	@echo "" >> DOCS.md
	@echo "## Quick Start" >> DOCS.md
	@echo '```bash' >> DOCS.md
	@echo "promptguard init --devops" >> DOCS.md
	@echo "promptguard run \"deploy to production\" --dry-run" >> DOCS.md
	@echo '```' >> DOCS.md
	@echo "Documentation generated in DOCS.md"

# Version management
version-patch:
	poetry version patch
	@echo "Version bumped to patch level"

version-minor:
	poetry version minor
	@echo "Version bumped to minor level"

version-major:
	poetry version major
	@echo "Version bumped to major level"

# Docker (if Dockerfile exists)
docker-build:
	docker build -t promptguard:latest .

docker-run:
	docker run --rm -it promptguard:latest

# CI/CD helpers
ci-test:
	poetry install --only main
	poetry run pytest tests/ -v
	poetry run ruff check .

ci-build:
	poetry build
	@echo "Build completed successfully"

# Release workflow
release: clean test lint build
	@echo "Ready for release! Run 'make publish' to upload to PyPI"

# All-in-one development command
dev: install-dev test lint
	@echo "Development environment ready!"
