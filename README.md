# 🧠 preLLM

**One function for small LLM preprocessing before large LLM execution.**
Like `litellm.completion()` but with decomposition.

```python
from prellm import preprocess_and_execute

result = await preprocess_and_execute(
    query="Deploy app to production",
    small_llm="ollama/qwen2.5:3b",
    large_llm="gpt-4o-mini",
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

# Docker
docker run prellm/prellm query "Deploy app" --small ollama/qwen2.5:3b --large gpt-4o-mini
```

## How It Works

```
User Query → Small LLM (≤3B, local) → classify/structure/enrich → Large LLM (cloud) → Validated Response
              Qwen2.5 / Phi3 / Gemma      decomposition pipeline     GPT-4 / Claude / Llama
```

**Result:** 70-80% token savings + enterprise-quality output for the price of a small LLM call.

## Python API

### One Function (recommended)

```python
from prellm import preprocess_and_execute

# Zero-config — just query + models
result = await preprocess_and_execute("Refaktoryzuj kod")

# Full control
result = await preprocess_and_execute(
    query="Deploy app to production",
    small_llm="ollama/qwen2.5:3b",      # local preprocessing
    large_llm="anthropic/claude-sonnet-4-20250514",  # cloud execution
    strategy="structure",                 # classify|structure|split|enrich|passthrough
    user_context="gdansk_embedded_python",
)

print(result.content)              # Large LLM response
print(result.decomposition)        # Small LLM analysis
print(result.model_used)           # "anthropic/claude-sonnet-4-20250514"
print(result.small_model_used)     # "ollama/qwen2.5:3b"
```

### Sync Version

```python
from prellm import preprocess_and_execute_sync

result = preprocess_and_execute_sync("Deploy app", large_llm="gpt-4o-mini")
```

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

### With YAML Config

```python
result = await preprocess_and_execute(
    query="Deploy to staging",
    config_path="configs/prellm_config.yaml",
)
```

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
    strategy="enrich",
    user_context={"cluster": "k8s-prod", "namespace": "backend"},
)
# Small LLM: parse context, identify missing fields, enrich prompt
# Large LLM: root cause + K8s manifests + Prometheus rules
# Cost: $0.02 + $0.38 = $0.40
```

### 3. Business Automation
```python
result = await preprocess_and_execute(
    query="Zautomatyzuj kalkulację leasingu dla camper van",
    small_llm="ollama/qwen2.5:3b",
    large_llm="anthropic/claude-sonnet-4-20250514",
    strategy="enrich",
    user_context="PL_automotive_leasing",
)
# Small LLM: domain=automotive, locale=PL, required=[VAT, WIBOR]
# Large LLM: Python calculator + Excel generator + PDF templates
# Cost: $0.015 + $0.52 = $0.535
```

## 5 Decomposition Strategies

| Strategy | What it does | Best for |
|---|---|---|
| `classify` | Classify intent + domain | General queries, routing |
| `structure` | Extract action, target, params | DevOps commands, API calls |
| `split` | Break into sub-queries | Complex multi-part requests |
| `enrich` | Add missing context | Incomplete prompts, safety |
| `passthrough` | No preprocessing | Simple/direct queries |

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

## Process Chains (DevOps Workflows)

```python
from prellm import PreLLM, ProcessChain

engine = PreLLM("configs/prellm_config.yaml")
chain = ProcessChain("configs/deploy.yaml", engine=engine)
result = await chain.execute(env="production", dry_run=True)

for step in result.steps:
    print(f"{step.step_name}: {step.status}")
```

## Architecture

```
preprocess_and_execute(query, small_llm, large_llm)
    │
    ├── ContextEngine (env/git/system)
    ├── QueryDecomposer (small LLM ≤3B)
    │   ├── classify → intent + domain
    │   ├── structure → action + target + params
    │   ├── split → sub-queries
    │   ├── enrich → missing fields + context
    │   └── compose → optimized prompt
    ├── LLMProvider (large LLM via litellm)
    │   ├── retry + fallback chain
    │   └── 100+ models (OpenAI, Anthropic, Ollama, etc.)
    └── PreLLMResponse (Pydantic v2 validated)
```

## Development

```bash
git clone https://github.com/wronai/prellm
cd prellm
poetry install
poetry run pytest          # 144+ tests
poetry run pytest --cov    # ~80% coverage
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full 12-month plan to make preLLM a standard.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Author

Created by **Tom Sapletta** - [tom@sapletta.com](mailto:tom@sapletta.com)
