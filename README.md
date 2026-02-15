# 🛡️ PromptGuard

**Lightweight LLM prompt middleware — bias detection, standardization, and DevOps process chains via YAML config.**

PromptGuard sits between your application and LLM providers, automatically detecting bias, ambiguity, and dangerous patterns in prompts. It enriches queries with context, validates outputs, and supports multi-step DevOps workflows with approval gates.

## Features

- **Bias & Ambiguity Detection** — regex + NLTK patterns for PL/EN, with DevOps-specific guardrails
- **YAML-Driven Config** — declarative rules, clarification templates, model fallbacks
- **100+ LLM Models** — via LiteLLM proxy (OpenAI, Anthropic, Llama, Mistral, etc.)
- **DevOps Process Chains** — multi-step workflows with approval gates, rollback, and audit trails
- **Context Injection** — auto-enrich prompts with env vars, git info, system state
- **Type-Safe Outputs** — Pydantic v2 validated responses
- **Lightweight** — <50MB, 5 dependencies, async-first

## Quick Start

```bash
# Install
pip install promptguard

# Generate config
promptguard init --devops -o rules.yaml

# Analyze a query (no LLM call)
promptguard analyze "Deploy to production" --config rules.yaml

# Run with LLM
promptguard run "Zdeployuj na staging" --config rules.yaml --model gpt-4o-mini

# Execute a process chain
promptguard process deploy.yaml --guard-config rules.yaml --env production
```

## Python API

```python
from promptguard import PromptGuard, ProcessChain

# Simple query
guard = PromptGuard("rules.yaml")
result = await guard("Deploy to production", model="gpt-4o-mini")
print(result.clarified)  # True — detected missing context
print(result.content)     # Enriched response

# Process chain
chain = ProcessChain("deploy.yaml")
result = await chain.execute(env="production", dry_run=True)
for step in result.steps:
    print(f"{step.step_name}: {step.status}")
```

## Configuration

### rules.yaml
```yaml
bias_patterns:
  - regex: "(deploy|zdeployuj)\\s+(na|to)\\s+(prod|production)"
    action: clarify
    severity: critical
    description: "Production deployment — requires context"

clarify_template: "[KONTEKST]: Podaj szczegóły dla: {query}"
max_retries: 3
policy: devops

models:
  fallback: ["gpt-4o-mini", "llama3"]

context_sources:
  - env: [CLUSTER, NAMESPACE, GIT_SHA]
  - git: [branch, short_sha]
```

### deploy.yaml (Process Chain)
```yaml
process: deploy-production
steps:
  - name: pre-check
    prompt: "Check readiness of {CLUSTER}"
    approval: auto
  - name: deploy
    prompt: "Rolling deploy to {CLUSTER}/{NAMESPACE}"
    approval: manual
    rollback: true
```

## Architecture

```
User Query → BiasDetector → ContextEngine → Enrichment → LiteLLM → Pydantic Validation → Response
                                                              ↑
                                        ProcessChain → Approval Gates → Audit Trail
```

## Development

```bash
git clone https://github.com/softreck/promptguard
cd promptguard
poetry install
poetry run pytest
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Author

Created by **Tom Sapletta** - [tom@sapletta.com](mailto:tom@sapletta.com)
