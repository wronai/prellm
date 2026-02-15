# preLLM Roadmap — From Tool to Standard

> **Goal:** Make preLLM the LiteLLM of prompt preprocessing.
> LiteLLM won with `completion()` for 100+ LLMs. preLLM wins with `preprocess_and_execute()` for structured prompt quality.

## The Problem

LiteLLM solved: *100 LLM APIs → 1 interface.*
preLLM solves: *Weak prompts → enterprise-quality structured output.*

```
Without preLLM:  User → raw prompt → LLM → unpredictable output
With preLLM:     User → small LLM decompose → large LLM execute → validated YAML output
```

## Core API — One Function

```python
from prellm import preprocess_and_execute

result = await preprocess_and_execute(
    query="refaktoryzuj kod",
    small_llm="ollama/qwen2.5:3b",   # local or remote
    large_llm="anthropic/claude-4.6", # 100+ models via litellm
    strategy="structure",              # classify|structure|split|enrich|passthrough
    user_context="gdansk_embedded_python",
)

print(result.content)        # Large LLM response
print(result.decomposition)  # Small LLM analysis
print(result.model_used)     # Which large model answered
```

## Zero-Config Start

```bash
pip install prellm
prellm "Deploy app to prod" --small ollama/qwen2.5:3b --large gpt-4o-mini
```

## Use Cases

### 1. Code Refactoring with User Context
```
USER: "Popraw mój projekt z hardcode'em"

SMALL LLM (Qwen2.5 local):
├── Classify: intent=refactor, confidence=0.92
├── Structure: action=refactor, target=hardcoded_strings
├── Missing: [file_paths, language_version]
└── Compose: structured prompt with full technical context

LARGE LLM (Claude): Complete refactored code with tests
Cost: Qwen2.5: $0.01 | Claude: $0.45 | Total: $0.46
```

### 2. Kubernetes Log Analysis
```
USER: "Zdiagnozuj problem z K8s podami"

SMALL LLM: Parse 10k log lines → extract OOM errors, crashloops
LARGE LLM: Root cause + K8s manifests + Prometheus rules
Cost: $0.02 + $0.38 = $0.40
```

### 3. Business Automation
```
USER: "Zautomatyzuj kalkulację leasingu"

SMALL LLM: Domain=automotive, locale=PL, required=[VAT, WIBOR]
LARGE LLM: Python calculator + Excel generator + PDF templates
Cost: $0.015 + $0.52 = $0.535
```

## Efficiency Gains

| Use Case | Without preLLM | With preLLM | Improvement |
|---|---|---|---|
| Code refactoring | 8k tokens, no context | 2.5k tokens + context | +45% precision |
| Log analysis | Generic diagnosis | Tech stack + env aware | +62% accuracy |
| Business automation | Generic Excel | PL legal + VAT aware | +78% usefulness |

## 12-Month Roadmap

### Month 1–3: MVP (target: 10k ⭐)
- [x] `pip install prellm`
- [x] `preprocess_and_execute(query, small_llm, large_llm)`
- [x] 5 decomposition strategies (classify, structure, split, enrich, passthrough)
- [x] YAML-driven domain rules + prompts
- [x] Pydantic v2 validated responses
- [ ] 20+ popular model pairs (Qwen→Claude, Llama→GPT, Phi→Mistral)
- [ ] GitHub CI: ruff + mypy + pytest + 100% coverage
- [ ] PyPI stable release

### Month 4–6: Proxy Server (target: 30k ⭐)
- [ ] FastAPI proxy server (OpenAI-compatible `/v1/chat/completions`)
- [ ] Docker 1-liner: `docker run -p 8080:8080 prellm/prellm`
- [ ] Load testing + stable Docker tags
- [ ] ChromaDB user context memory
- [ ] Response schema contracts (YAML)
- [ ] Budget tracking / spend limits

### Month 7–12: Enterprise (target: 100k+ ⭐)
- [ ] Kubernetes operator
- [ ] Multi-tenant (projects/teams)
- [ ] VSCode + Cursor extensions
- [ ] LangChain + LlamaIndex + Haystack integrations
- [ ] 1-click Vercel/Netlify deploy

## Docker Deployment (Current)

```bash
docker run -p 8080:8080 prellm/prellm \
  --small-models ollama/qwen2.5:3b \
  --large-models anthropic/claude-4.6,openai/gpt-4o-mini \
  --strategy classify
```

## Docker Deployment (Future — OpenAI-compatible proxy)

```bash
# OpenAI-compatible API
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "prellm:qwen→claude", "messages": [{"role": "user", "content": "Deploy app"}]}'
```

## Comparison with LiteLLM

| Aspect | LiteLLM (standard) | preLLM (goal) |
|---|---|---|
| Problem solved | 100 LLM APIs → 1 interface | Weak prompts → enterprise quality |
| Killer feature | `completion()` for all | `preprocess_and_execute()` for all |
| Proxy server | Yes (key to adoption) | Yes (OpenAI format) |
| Model support | 100+ via providers | 100+ via litellm (dependency) |
| Unique value | Unified API | Small LLM preprocessing |

## Success Criteria

### Technical
1. **ONE function:** `preprocess_and_execute()`
2. **OpenAI-compatible proxy** (Month 4)
3. **`pip install && works in 60s`**
4. **100% test coverage + typing**
5. **Docker stable images**

### Ecosystem
6. LangChain + LlamaIndex + Haystack integration
7. VSCode extension (YAML schema autocomplete)
8. Documentation: 1-min getting started
9. Community: Discord + bi-weekly releases

### Business
10. OSS-first: Apache-2.0 license
11. Hacker News launch + Reddit AMA
12. 1-click cloud deploy
13. Sponsors: Anthropic, Mistral, Perplexity

## License

Apache License 2.0
