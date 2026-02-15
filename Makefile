# Prellm Makefile
.PHONY: help install install-dev test lint format clean build publish run demo init bump-patch check-bumpver check-build check-twine

# Check for Poetry availability
POETRY := $(shell command -v poetry 2>/dev/null)
ifeq ($(POETRY),)
RUN :=
PYTHON := python3
PIP := pip
else
RUN := poetry run
PYTHON := $(RUN) python
PIP :=
endif

# Colors for terminal output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

# Default target
help:
	@echo "Prellm - LLM prompt middleware"
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
	@if [ -z "$(POETRY)" ]; then \
		$(PIP) install -e .; \
	else \
		poetry install --only main; \
	fi

install-dev:
	@if [ -z "$(POETRY)" ]; then \
		$(PIP) install -e ".[dev]"; \
	else \
		poetry install; \
	fi

# Quality checks
test:
	$(RUN) python -m pytest tests/ -v

test-cov:
	$(RUN) python -m pytest tests/ -v --cov=prellm --cov-report=html --cov-report=term

lint:
	$(RUN) ruff check .

format:
	$(RUN) ruff format .

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
build: clean check-build ## Build package
	$(PYTHON) -m build

publish: bump-patch build check-twine ## Publish to PyPI (production)
	@echo "$(YELLOW)Publishing to PyPI...$(NC)"
	$(PYTHON) -m twine upload dist/*
	@echo "$(GREEN)Published to PyPI!$(NC)"
	@echo "Install with: pip install prellm"

publish-test: build ## Publish to Test PyPI
	@echo "$(YELLOW)Publishing to Test PyPI...$(NC)"
	$(PYTHON) -m twine upload --repository testpypi dist/*
	@echo "$(GREEN)Published to Test PyPI!$(NC)"

bump-patch: check-bumpver ## Bump patch version (updates pyproject.toml and prellm/__init__.py)
	$(PYTHON) -m bumpver update --patch

check-twine: ## Ensure twine is installed
	@$(PYTHON) -c "import twine" >/dev/null 2>&1 || ( \
	   echo "Missing twine. Installing twine..."; \
	   $(PYTHON) -m pip install "twine>=4.0.0" >/dev/null; \
	   $(PYTHON) -c "import twine" >/dev/null 2>&1 || (echo "Failed to install twine."; exit 1); \
	)

check-build: ## Ensure build is installed
	@$(PYTHON) -c "import build" >/dev/null 2>&1 || ( \
	   echo "Missing build. Installing build..."; \
	   $(PYTHON) -m pip install "build>=0.8.0" >/dev/null; \
	   $(PYTHON) -c "import build" >/dev/null 2>&1 || (echo "Failed to install build."; exit 1); \
	)

check-bumpver: ## Ensure bumpver is installed
	@$(PYTHON) -c "import bumpver" >/dev/null 2>&1 || ( \
	   echo "Missing bumpver. Installing bumpver..."; \
	   $(PYTHON) -m pip install "bumpver>=2023.1129" >/dev/null; \
	   $(PYTHON) -c "import bumpver" >/dev/null 2>&1 || ( \
		  echo "bumpver still missing. Installing project dev dependencies..."; \
		  $(PIP) install -e \".[dev]\"; \
		  $(PYTHON) -c "import bumpver" >/dev/null 2>&1 || (echo "Failed to install bumpver."; exit 1); \
	   ); \
	)

# Demo and examples
run:
	$(RUN) prellm run "deploy to production" --dry-run

demo:
	@echo "Generating demo config..."
	$(RUN) prellm init --devops
	@echo ""
	@echo "Running demo analysis..."
	$(RUN) prellm analyze "zawsze restartuj serwer" --config rules.yaml
	@echo ""
	@echo "Running demo process chain..."
	$(RUN) prellm process configs/deploy.yaml --dry-run --guard-config rules.yaml || echo "Process chain config not found, skipping..."

init:
	$(RUN) prellm init --devops

# Development helpers
dev-setup: install-dev
	@echo "Setting up pre-commit hooks..."
	@echo "#!/bin/bash" > .git/hooks/pre-commit
	@echo "$(RUN) ruff check ." >> .git/hooks/pre-commit
	@echo "$(RUN) python -m pytest tests/ -q" >> .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "Pre-commit hooks installed!"

docs:
	@echo "Generating documentation..."
	@echo "# Prellm Documentation" > DOCS.md
	@echo "" >> DOCS.md
	@echo "## Installation" >> DOCS.md
	@echo '```bash' >> DOCS.md
	@echo "pip install prellm" >> DOCS.md
	@echo '```' >> DOCS.md
	@echo "" >> DOCS.md
	@echo "## Quick Start" >> DOCS.md
	@echo '```bash' >> DOCS.md
	@echo "prellm init --devops" >> DOCS.md
	@echo "prellm run \"deploy to production\" --dry-run" >> DOCS.md
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
	docker build -t prellm:latest .

docker-run:
	docker run --rm -it prellm:latest

# CI/CD helpers
ci-test:
	ifeq ($(POETRY),)
		$(PIP) install -e .
	else
		poetry install --only main
	endif
	$(RUN) python -m pytest tests/ -v
	$(RUN) ruff check .

ci-build:
	$(PYTHON) -m build
	@echo "Build completed successfully"

# Release workflow
release: clean test lint build
	@echo "Ready for release! Run 'make publish' to upload to PyPI"

# All-in-one development command
dev: install-dev test lint
	@echo "Development environment ready!"

# Version management (alternative methods)
bump-minor: check-bumpver ## Bump minor version
	$(PYTHON) -m bumpver update --minor

bump-major: check-bumpver ## Bump major version
	$(PYTHON) -m bumpver update --major

set-version: ## Set specific version (usage: make set-version VERSION=1.2.3)
	@if [ -z "$(VERSION)" ]; then echo "Usage: make set-version VERSION=1.2.3"; exit 1; fi
	$(PYTHON) -m bumpver update --set-version $(VERSION)
	@echo "Version set to $(VERSION)"
