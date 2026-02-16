# preLLM

**One function for small LLM preprocessing before large LLM execution.**
Like `litellm.completion()` — but with a smart preprocessing layer.

```python
from prellm import preprocess_and_execute

result = await preprocess_and_execute(
    query="Deploy app to production",
    small_llm="ollama/qwen2.5:3b",     # local, fast, cheap
    large_llm="anthropic/claude-sonnet-4-20250514",  # cloud, powerful
)
print(result.content)
```

## Install & Run in 60 Seconds

```bash
pip install prellm

# CLI — zero config
prellm query "Zdeployuj apkę na prod" --small ollama/qwen2.5:3b --large gpt-4o-mini

# With strategy
prellm query "Refaktoryzuj kod" --strategy structure --json

# Two-agent pipeline (v0.3)
prellm query "Deploy app" --pipeline dual_agent_full

# Docker
docker run prellm/prellm query "Deploy app" --small ollama/qwen2.5:3b --large gpt-4o-mini
```

### Interactive Configuration (repo)

```bash
make install-dev
make config         # guided wizard + diagnostics
source .env
prellm doctor --live
make examples       # runs all example scripts (real-time)
```

## How It Works

```text
User Query
  → Small LLM (≤24B, local)    → classify / structure / enrich    → optimized prompt
    Qwen2.5 / Phi3 / Gemma       PromptPipeline (YAML)
  → Large LLM (cloud)          → execute with full context        → validated response
    GPT-4 / Claude / Llama       ResponseValidator (YAML schema)
```

**Result:** 70–80% token savings + enterprise-quality output for the price of a small LLM call.

---

## Python API

### One Function — Two Execution Paths

```python
from prellm import preprocess_and_execute

# PATH A: Strategy-based (v0.2, default)
result = await preprocess_and_execute(
    query="Deploy app to production",
    small_llm="ollama/qwen2.5:3b",
    large_llm="anthropic/claude-sonnet-4-20250514",
    strategy="structure",                 # classify|structure|split|enrich|passthrough
    user_context="gdansk_embedded_python",
)

# PATH B: Pipeline-based two-agent (v0.3)
result = await preprocess_and_execute(
    query="Deploy app to production",
    small_llm="ollama/qwen2.5:3b",
    large_llm="anthropic/claude-sonnet-4-20250514",
    pipeline="dual_agent_full",           # any pipeline from pipelines.yaml
)

print(result.content)              # Large LLM response
print(result.decomposition)        # Small LLM analysis
print(result.model_used)           # Which large model answered
print(result.small_model_used)     # Which small model preprocessed
```

### Sync Version

```python
from prellm import preprocess_and_execute_sync

result = preprocess_and_execute_sync("Deploy app", large_llm="gpt-4o-mini")
# Works exactly the same, just blocking
```

### Zero-Config

```python
# Defaults: small=ollama/qwen2.5:3b, large=claude-sonnet, strategy=classify
result = await preprocess_and_execute("Refaktoryzuj kod")
```

---

## LLM Provider Examples

preLLM uses **LiteLLM** under the hood, so any model string supported by LiteLLM works.

### Ollama (local, free)

```python
# Start Ollama: ollama serve
# Pull model:   ollama pull qwen2.5:3b

result = await preprocess_and_execute(
    query="Explain Kubernetes pods",
    small_llm="ollama/qwen2.5:3b",       # local small model
    large_llm="ollama/llama3:70b",        # local large model
)
# Cost: $0.00 — both models run locally
```

### Ollama + OpenAI (hybrid)

```python
result = await preprocess_and_execute(
    query="Review my Python code",
    small_llm="ollama/qwen2.5:3b",       # local preprocessing
    large_llm="gpt-4o-mini",             # OpenAI execution
)
# Cost: $0.00 (local) + ~$0.15 (OpenAI) = $0.15
```

### Ollama + Anthropic (hybrid)

```python
result = await preprocess_and_execute(
    query="Deploy microservices to K8s",
    small_llm="ollama/phi3:mini",         # local preprocessing
    large_llm="anthropic/claude-sonnet-4-20250514",  # Anthropic execution
)
```

### OpenAI only

```python
result = await preprocess_and_execute(
    query="Analyze sales data",
    small_llm="gpt-4o-mini",             # cheap OpenAI preprocessing
    large_llm="gpt-4o",                  # powerful OpenAI execution
)
```

### Anthropic only

```python
result = await preprocess_and_execute(
    query="Write a compliance report",
    small_llm="anthropic/claude-haiku",
    large_llm="anthropic/claude-sonnet-4-20250514",
)
```

### Groq (fast inference)

```python
result = await preprocess_and_execute(
    query="Summarize meeting notes",
    small_llm="groq/llama-3.1-8b-instant",   # fast Groq preprocessing
    large_llm="groq/llama-3.3-70b-versatile", # fast Groq execution
)
```

### Mistral

```python
result = await preprocess_and_execute(
    query="Translate technical docs",
    small_llm="mistral/mistral-small-latest",
    large_llm="mistral/mistral-large-latest",
)
```

### OpenRouter (multi-provider + vision)

```python
result = await preprocess_and_execute(
    query="Analyze this UI screenshot",
    small_llm="ollama/qwen2.5:3b",
    large_llm="openrouter/qwen/qwen3-vl-32b-instruct",
)
```

### Azure OpenAI

```python
result = await preprocess_and_execute(
    query="Generate quarterly report",
    small_llm="azure/gpt-4o-mini-deployment",
    large_llm="azure/gpt-4o-deployment",
)
```

### AWS Bedrock

```python
result = await preprocess_and_execute(
    query="Optimize Lambda function",
    small_llm="bedrock/anthropic.claude-haiku",
    large_llm="bedrock/anthropic.claude-sonnet",
)
```

> **Full provider list:** See [LiteLLM docs](https://docs.litellm.ai/docs/providers) — preLLM supports all 100+ providers.

---

## Integration with Existing LiteLLM Projects

### Drop-in Enhancement

If you already use LiteLLM, preLLM adds preprocessing with **one line change:**

```python
# BEFORE — direct litellm call
import litellm
response = await litellm.acompletion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Deploy app to production"}],
)

# AFTER — preLLM preprocessing + same litellm execution
from prellm import preprocess_and_execute
result = await preprocess_and_execute(
    query="Deploy app to production",
    large_llm="gpt-4o",  # same model, now with preprocessing
)
# result.content == same quality, but with structured decomposition
```

### Use Your Existing `.env`

preLLM reads the same environment variables as LiteLLM:

```bash
# .env — works with both litellm and prellm
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...

# preLLM-specific (optional)
PRELLM_SMALL_DEFAULT=ollama/qwen2.5:3b
PRELLM_LARGE_DEFAULT=anthropic/claude-sonnet-4-20250514
PRELLM_STRATEGY=classify

# Legacy names still supported
SMALL_MODEL=ollama/qwen2.5:3b
LARGE_MODEL=gpt-4o-mini
```

### LiteLLM Proxy Integration

If you run a LiteLLM proxy, point preLLM at it:

```python
import os
os.environ["OPENAI_API_BASE"] = "http://localhost:4000"  # your litellm proxy

result = await preprocess_and_execute(
    query="Deploy app",
    small_llm="openai/small-model",   # routed through litellm proxy
    large_llm="openai/large-model",   # routed through litellm proxy
)
```

### OpenAI SDK-Compatible Server

preLLM ships an OpenAI-compatible proxy — use it from **any** OpenAI SDK client:

```bash
# Start preLLM server
prellm serve --port 8080 --small ollama/qwen2.5:3b --large gpt-4o-mini

# Use from OpenAI Python SDK
import openai
client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="any")
response = client.chat.completions.create(
    model="prellm:default",
    messages=[{"role": "user", "content": "Deploy app to production"}],
)

# Use from curl
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"prellm:qwen→claude","messages":[{"role":"user","content":"Deploy app"}]}'

# Use v0.3 pipeline via API
curl http://localhost:8080/v1/chat/completions \
  -d '{"model":"prellm:default","messages":[{"role":"user","content":"Deploy app"}],"prellm":{"pipeline":"dual_agent_full"}}'
```

---

## Two-Agent Architecture (v0.3)

The `pipeline=` parameter activates the new two-agent architecture:

```text
USER QUERY
    │
    ▼
┌─────────────────────────────────────┐
│  PREPROCESSOR AGENT (small LLM)     │
│  PromptRegistry (YAML prompts)      │
│  PromptPipeline (YAML steps)        │
│  → classify → structure → compose   │
│  → IntermediateValidator            │
└──────────────┬──────────────────────┘
               │ structured executor_input
               ▼
┌─────────────────────────────────────┐
│  EXECUTOR AGENT (large LLM)         │
│  → execute with full context        │
│  → ResponseValidator (YAML schema)  │
│  → PreLLMResponse (typed)           │
└─────────────────────────────────────┘
```

### Custom Pipelines (YAML)

Define your own preprocessing pipeline — no Python code changes needed:

```yaml
# configs/pipelines.yaml
pipelines:
  my_pipeline:
    description: "Custom 3-step pipeline"
    steps:
      - name: classify
        prompt: classify          # from configs/prompts.yaml
        output: classification
      - name: extract
        prompt: structure
        output: fields
      - name: compose
        prompt: compose
        input: [query, classification, fields]
        output: composed_prompt
```

```python
result = await preprocess_and_execute(
    query="Deploy app",
    pipeline="my_pipeline",  # uses your custom YAML pipeline
)
```

### Available Pipelines

| Pipeline | Steps | Best for |
|---|---|---|
| `classify` | classify | Quick intent routing |
| `structure` | classify → structure → compose | DevOps, API calls |
| `split` | classify → split → compose | Complex multi-part queries |
| `enrich` | classify → enrich | Incomplete prompts |
| `dual_agent_full` | context → decompose → optimize → format | Maximum quality |
| `passthrough` | (none) | Direct forwarding |

### Custom Prompts (YAML)

All system prompts are in `configs/prompts.yaml` with Jinja2 templating:

```yaml
# configs/prompts.yaml
prompts:
  classify:
    system: |
      You are a query classifier.
      Intents: {{ intents | default("deploy, query, create, delete") }}
      Respond ONLY with JSON: {"intent": "...", "confidence": 0.0-1.0}
    max_tokens: 256
    temperature: 0.1
```

### Response Validation (YAML)

Validate LLM outputs with schemas — no code changes:

```yaml
# configs/response_schemas.yaml
schemas:
  classification:
    required_fields: [intent, confidence]
    types:
      intent: string
      confidence: float
    constraints:
      confidence: {min: 0.0, max: 1.0}
      intent: {enum: [deploy, query, create, delete, other]}
```

---

## 5 Decomposition Strategies (v0.2)

| Strategy | What it does | Best for |
|---|---|---|
| `classify` | Classify intent + domain | General queries, routing |
| `structure` | Extract action, target, params | DevOps commands, API calls |
| `split` | Break into sub-queries | Complex multi-part requests |
| `enrich` | Add missing context | Incomplete prompts, safety |
| `passthrough` | No preprocessing | Simple/direct queries |

### With Domain Rules

```python
result = await preprocess_and_execute(
    query="Usuń bazę danych klientów",
    small_llm="ollama/qwen2.5:3b",
    large_llm="gpt-4o-mini",
    domain_rules=[{
        "name": "destructive_db",
        "keywords": ["delete", "drop", "usuń"],
        "required_fields": ["target_database", "backup_confirmed"],
        "severity": "critical",
    }],
)
print(result.decomposition.missing_fields)  # ["target_database", "backup_confirmed"]
```

---

## Use Cases

### 1. Code Refactoring

```python
result = await preprocess_and_execute(
    query="Popraw mój projekt z hardcode'em",
    small_llm="ollama/qwen2.5:3b",
    large_llm="anthropic/claude-sonnet-4-20250514",
    strategy="structure",
    user_context="gdansk_embedded_python",
)
# Small LLM: classify intent, extract structure, compose prompt
# Large LLM: complete refactored code with tests
# Cost: $0.01 + $0.45 = $0.46
```

### 2. Kubernetes Diagnostics

```python
result = await preprocess_and_execute(
    query="Zdiagnozuj problem z K8s podami",
    small_llm="ollama/qwen2.5:3b",
    large_llm="gpt-4o-mini",
    pipeline="structure",
    user_context={"cluster": "k8s-prod", "namespace": "backend"},
)
# Preprocessor: parse context, identify missing fields, compose prompt
# Executor: root cause + K8s manifests + Prometheus rules
# Cost: $0.02 + $0.38 = $0.40
```

### 3. Business Automation

```python
result = await preprocess_and_execute(
    query="Zautomatyzuj kalkulację leasingu dla camper van",
    small_llm="ollama/qwen2.5:3b",
    large_llm="anthropic/claude-sonnet-4-20250514",
    pipeline="enrich",
    user_context="PL_automotive_leasing",
)
# Preprocessor: domain=automotive, locale=PL, required=[VAT, WIBOR]
# Executor: Python calculator + Excel generator + PDF templates
# Cost: $0.015 + $0.52 = $0.535
```

---

## Configuration (YAML)

```yaml
# configs/prellm_config.yaml
small_model:
  model: "ollama/qwen2.5:3b"
  fallback: ["phi3:mini"]
  max_tokens: 512

large_model:
  model: "gpt-4o-mini"
  fallback: ["llama3", "mistral"]
  max_tokens: 2048

default_strategy: classify

domain_rules:
  - name: production_deploy
    keywords: ["deploy", "push", "release"]
    required_fields: ["environment", "version"]
    severity: critical
    strategy: structure
```

### Per-Domain Defaults

Ready-to-use configs in `configs/defaults/`:

| Domain | File | Covers |
|---|---|---|
| DevOps | `configs/defaults/devops.yaml` | deploy, K8s, monitoring, CI/CD |
| Coding | `configs/defaults/coding.yaml` | refactoring, review, debugging |
| Business | `configs/defaults/business.yaml` | leasing, invoicing, compliance |
| Embedded | `configs/defaults/embedded.yaml` | RPi, ESP32, sensors, IoT |

---

## Process Chains (DevOps Workflows)

```python
from prellm import PreLLM, ProcessChain

engine = PreLLM("configs/prellm_config.yaml")
chain = ProcessChain("configs/deploy.yaml", engine=engine)
result = await chain.execute(env="production", dry_run=True)

for step in result.steps:
    print(f"{step.step_name}: {step.status}")
```

---

## Architecture

```text
preprocess_and_execute(query, small_llm, large_llm, strategy= | pipeline=)
    │
    ├── [strategy path — v0.2]
    │   ├── ContextEngine (env/git/system)
    │   ├── QueryDecomposer (small LLM)
    │   │   └── classify → structure → split → enrich → compose
    │   └── LLMProvider (large LLM via litellm)
    │
    ├── [pipeline path — v0.3]
    │   ├── PreprocessorAgent
    │   │   ├── PromptRegistry (YAML, Jinja2)
    │   │   ├── PromptPipeline (YAML-configurable steps)
    │   │   │   ├── LLM steps (small LLM calls)
    │   │   │   └── Algorithmic steps (validation, formatting)
    │   │   └── ContextEngine + UserMemory (SQLite)
    │   ├── ExecutorAgent
    │   │   ├── LLMProvider (large LLM via litellm)
    │   │   └── ResponseValidator (YAML schemas)
    │   └── 100+ models via LiteLLM
    │
    └── PreLLMResponse (Pydantic v2 validated)
```

## Examples

Ready-to-run examples in `examples/`:

| Example | File | Config |
|---|---|---|
| **Quick Start** | `examples/quick_start.py` | default env (no config) |
| **K8s Debugging** | `examples/k8s_debug.py` | `configs/domains/devops_k8s.yaml` |
| **Polish Leasing** | `examples/polish_leasing.py` | `configs/domains/polish_finance.yaml` |
| **Embedded/IoT** | `examples/embedded_refactor.py` | `configs/domains/embedded.yaml` |
| **Providers** | `examples/providers.py` | env keys per provider |
| **Python SDK** | `examples/python_sdk.py` | env keys per provider |
| **CLI + API** | `examples/cli_examples.sh`, `examples/curl_api.sh` | server running |

### Run Examples

```bash
# Run all examples (real-time output)
make examples

# Run single example
python examples/quick_start.py
python examples/k8s_debug.py
python examples/polish_leasing.py

# CLI + curl demos (server must be running)
bash examples/cli_examples.sh
bash examples/curl_api.sh
```

### K8s Debugging

```python
from prellm import preprocess_and_execute

result = await preprocess_and_execute(
    query="Pod backend-api restartuje sie z CrashLoopBackOff",
    config_path="configs/domains/devops_k8s.yaml",
    strategy="structure",
    user_context={"cluster": "k8s-prod", "namespace": "backend"},
)
```

### Polish Leasing Calculator

```python
result = await preprocess_and_execute(
    query="Oblicz rate leasingu operacyjnego camper van za 250000 PLN netto, 48 miesiecy",
    config_path="configs/domains/polish_finance.yaml",
    strategy="structure",
)
```

### Embedded/IoT Refactoring

```python
result = await preprocess_and_execute(
    query="Zrefaktoruj ESP32 monitoring - za duzo hardcode'ow, brak OTA",
    config_path="configs/domains/embedded.yaml",
    strategy="structure",
    user_context={"mcu": "ESP32-S3", "flash": "8MB", "ram": "512KB"},
)
```

---

## Development

```bash
git clone https://github.com/wronai/prellm
cd prellm
poetry install
poetry run pytest                 # 280+ tests (core + examples)
poetry run pytest --cov           # coverage report
poetry run ruff check prellm/     # linting
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full plan.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Author

Created by **Tom Sapletta** - [tom@sapletta.com](mailto:tom@sapletta.com)
