FROM python:3.12-slim AS base
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY prellm/ prellm/
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction && \
    pip uninstall -y poetry

COPY configs/ configs/

# Default: prellm query "..." --small ollama/qwen2.5:3b --large gpt-4o-mini
# Override: docker run prellm/prellm query "Deploy app" --small phi3:mini --large claude-sonnet-4-20250514
ENTRYPOINT ["prellm"]
CMD ["--help"]
