#!/usr/bin/env bash
# preLLM CLI Examples — shell commands for direct usage
#
# Install: pip install prellm

echo "=== preLLM CLI Examples ==="
echo ""

# ─────────────────────────────────────────────
# 1. Basic query (zero-config)
# ─────────────────────────────────────────────
echo "--- 1. Basic Query ---"
echo '$ prellm query "Explain Docker networking"'
prellm query "Explain Docker networking"
echo ""

# ─────────────────────────────────────────────
# 2. Custom models
# ─────────────────────────────────────────────
echo "--- 2. Custom Models ---"
echo '$ prellm query "Deploy app" --small ollama/qwen2.5:3b --large gpt-4o-mini'
prellm query "Deploy app to staging" \
  --small ollama/qwen2.5:3b \
  --large gpt-4o-mini
echo ""

# ─────────────────────────────────────────────
# 3. Structure strategy with JSON output
# ─────────────────────────────────────────────
echo "--- 3. Structure Strategy (JSON) ---"
echo '$ prellm query "Refaktoryzuj kod" --strategy structure --json'
prellm query "Refaktoryzuj kod z hardcode'ami" \
  --strategy structure \
  --json
echo ""

# ─────────────────────────────────────────────
# 4. With user context
# ─────────────────────────────────────────────
echo "--- 4. With Context ---"
echo '$ prellm query "K8s diagnostics" --context "gdansk_embedded_python" --strategy enrich'
prellm query "Zdiagnozuj problem z K8s podami" \
  --context "gdansk_embedded_python_docker_k8s" \
  --strategy enrich
echo ""

# ─────────────────────────────────────────────
# 5. With YAML config
# ─────────────────────────────────────────────
echo "--- 5. With Config File ---"
echo '$ prellm query "Deploy to prod" --config configs/prellm_config.yaml'
prellm query "Deploy to production" \
  --config configs/prellm_config.yaml
echo ""

# ─────────────────────────────────────────────
# 6. Decompose only (no large LLM call)
# ─────────────────────────────────────────────
echo "--- 6. Decompose Only ---"
echo '$ prellm decompose "Deploy app to prod" --strategy structure --json'
prellm decompose "Deploy app to production" \
  --strategy structure \
  --json
echo ""

# ─────────────────────────────────────────────
# 7. Start API server
# ─────────────────────────────────────────────
echo "--- 7. Start API Server ---"
echo '$ prellm serve --small ollama/qwen2.5:3b --large gpt-4o-mini --port 8080'
echo "(not running — use this command to start the server)"
echo ""

# ─────────────────────────────────────────────
# 8. Generate config
# ─────────────────────────────────────────────
echo "--- 8. Generate Config ---"
echo '$ prellm init --v2 --devops -o my_config.yaml'
prellm init --v2 --devops -o /tmp/prellm_example_config.yaml
echo ""

# ─────────────────────────────────────────────
# 9. v0.1 compat: analyze without LLM
# ─────────────────────────────────────────────
echo "--- 9. Analyze (v0.1 compat) ---"
echo '$ prellm analyze "Deploy to production"'
prellm analyze "Deploy to production"
echo ""

# ─────────────────────────────────────────────
# 10. Local models only (Ollama)
# ─────────────────────────────────────────────
echo "--- 10. Local Models (Ollama) ---"
echo '$ prellm query "debug docker" --small ollama/llama3.2 --large ollama/llama3:70b'
echo "(requires Ollama running locally)"
echo ""

echo "=== Done ==="
