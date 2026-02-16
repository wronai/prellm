FROM python:3.12-slim AS base
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies first (cached layer)
COPY pyproject.toml README.md ./
COPY prellm/ prellm/
COPY configs/ configs/
RUN pip install --no-cache-dir .

# Environment variables for server mode
ENV SMALL_MODEL="ollama/qwen2.5:3b"
ENV LARGE_MODEL="gpt-4o-mini"
ENV PRELLM_STRATEGY="classify"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Usage:
#   Server:  docker run -p 8080:8080 prellm/prellm serve
#   Query:   docker run prellm/prellm query "Deploy app" --small phi3:mini --large gpt-4o-mini
#   Help:    docker run prellm/prellm --help
ENTRYPOINT ["prellm"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
