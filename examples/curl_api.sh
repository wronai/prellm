#!/usr/bin/env bash
# preLLM API Examples — curl commands for the OpenAI-compatible server
#
# Start server first:
#   prellm serve --small ollama/qwen2.5:3b --large gpt-4o-mini --port 8080
#   # or
#   docker run -p 8080:8080 prellm/prellm serve --small ollama/qwen2.5:3b --large gpt-4o-mini

BASE_URL="${PRELLM_BASE_URL:-http://localhost:8080}"

echo "=== preLLM API Examples ==="
echo "Server: $BASE_URL"
echo ""

# ─────────────────────────────────────────────
# 1. Health check
# ─────────────────────────────────────────────
echo "--- Health Check ---"
curl -s "$BASE_URL/health" | jq .
echo ""

# ─────────────────────────────────────────────
# 2. List models
# ─────────────────────────────────────────────
echo "--- Available Models ---"
curl -s "$BASE_URL/v1/models" | jq .
echo ""

# ─────────────────────────────────────────────
# 3. Basic chat completion (OpenAI-compatible)
# ─────────────────────────────────────────────
echo "--- Basic Completion ---"
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "prellm:default",
    "messages": [
      {"role": "user", "content": "Explain Docker networking in 3 sentences"}
    ]
  }' | jq .
echo ""

# ─────────────────────────────────────────────
# 4. Code refactoring with user context
# ─────────────────────────────────────────────
echo "--- Code Refactoring (with context) ---"
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "prellm:qwen→claude",
    "messages": [
      {"role": "user", "content": "Znajdź hardcode w moim Python projekcie i zaproponuj refaktoryzację"}
    ],
    "prellm": {
      "user_context": "gdansk embedded python docker kubernetes",
      "strategy": "structure",
      "response_format": "yaml"
    }
  }' | jq .
echo ""

# ─────────────────────────────────────────────
# 5. K8s diagnostics with enrich strategy
# ─────────────────────────────────────────────
echo "--- K8s Diagnostics ---"
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "prellm:qwen→claude",
    "messages": [
      {"role": "user", "content": "Zdiagnozuj problem z Kubernetes podami - CrashLoopBackOff na namespace backend"}
    ],
    "prellm": {
      "user_context": {"cluster": "k8s-prod", "namespace": "backend", "node": "rpi-cluster"},
      "strategy": "enrich"
    }
  }' | jq .
echo ""

# ─────────────────────────────────────────────
# 6. With domain rules (safety check)
# ─────────────────────────────────────────────
echo "--- Production Deploy (domain rules) ---"
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "prellm:default",
    "messages": [
      {"role": "user", "content": "Deploy app to production"}
    ],
    "prellm": {
      "strategy": "structure",
      "domain_rules": [{
        "name": "prod_deploy",
        "keywords": ["deploy", "production"],
        "intent": "deploy",
        "required_fields": ["environment", "version", "rollback_plan"],
        "severity": "critical"
      }]
    }
  }' | jq .
echo ""

# ─────────────────────────────────────────────
# 7. Passthrough (no preprocessing)
# ─────────────────────────────────────────────
echo "--- Passthrough (direct) ---"
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "prellm:default",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "prellm": {
      "strategy": "passthrough"
    }
  }' | jq .
echo ""

# ─────────────────────────────────────────────
# 8. Streaming with stage progress
# ─────────────────────────────────────────────
echo "--- Streaming (with stages) ---"
curl -s -N -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "prellm:qwen→claude",
    "messages": [
      {"role": "user", "content": "Create a REST API for user management"}
    ],
    "stream": true,
    "prellm": {
      "show_stages": true,
      "strategy": "structure"
    }
  }'
echo ""

# ─────────────────────────────────────────────
# 9. Batch processing
# ─────────────────────────────────────────────
echo "--- Batch Processing ---"
curl -s -X POST "$BASE_URL/v1/batch" \
  -H "Content-Type: application/json" \
  -d '[
    {"query": "Refaktoryzuj hardcode w projekcie", "context": "python docker", "strategy": "structure"},
    {"query": "Zdiagnozuj K8s logi", "context": "rpi cluster", "strategy": "enrich"},
    {"query": "Kalkulacja leasingu camper van", "context": "PL automotive", "strategy": "classify"}
  ]' | jq .
echo ""

# ─────────────────────────────────────────────
# 10. Custom model pair
# ─────────────────────────────────────────────
echo "--- Custom Model Pair ---"
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "prellm:phi3->gpt-4o",
    "messages": [
      {"role": "user", "content": "Optimize this SQL query for PostgreSQL"}
    ],
    "max_tokens": 4096,
    "temperature": 0.3
  }' | jq .
echo ""

echo "=== Done ==="
