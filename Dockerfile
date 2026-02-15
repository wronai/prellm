FROM python:3.12-slim AS base

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml README.md ./
COPY promptguard/ promptguard/
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction && \
    pip uninstall -y poetry && \
    python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

COPY configs/ configs/

ENTRYPOINT ["promptguard"]
CMD ["--help"]
