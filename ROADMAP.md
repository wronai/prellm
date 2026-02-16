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

### Month 1–3: MVP ✅
- [x] `pip install prellm`
- [x] `preprocess_and_execute(query, small_llm, large_llm)`
- [x] 5 decomposition strategies (classify, structure, split, enrich, passthrough)
- [x] YAML-driven domain rules + prompts
- [x] Pydantic v2 validated responses
- [x] 100+ model providers via LiteLLM (Ollama, OpenAI, Anthropic, Groq, Mistral, Azure, Bedrock, Gemini, Together, DeepSeek)
- [x] GitHub CI: ruff + pytest + coverage (`.github/workflows/ci.yml`)
- [ ] PyPI stable release
- [ ] mypy strict + 100% coverage

### Month 3–4: Two-Agent Architecture v0.3 ✅
- [x] **PreprocessorAgent** (small LLM ≤24B) + **ExecutorAgent** (large LLM)
- [x] **PromptRegistry** — YAML prompts with Jinja2 templating
- [x] **PromptPipeline** — YAML-configurable multi-step pipelines (LLM + algorithmic steps)
- [x] **ResponseValidator** — YAML schema validation for LLM outputs
- [x] **UserMemory** — SQLite-backed user interaction history
- [x] Unified `preprocess_and_execute()` — `strategy=` (v0.2) or `pipeline=` (v0.3)
- [x] 6 built-in pipelines: classify, structure, split, enrich, dual_agent_full, passthrough
- [x] Per-domain default configs: devops, coding, business, embedded
- [x] 329+ tests (v0.3 architecture + trace + budget + context)
- [x] LiteLLM integration guide + provider examples
- [x] OpenAI SDK-compatible server with pipeline support

### Month 4–5: Observability & DevEx ✅ (NEW)
- [x] **Execution Trace** — `--trace` flag generates markdown decision path (.prellm/)
- [x] **Budget Tracking** — `BudgetTracker` with per-request cost recording + monthly limits
- [x] **`prellm budget`** CLI — view spend, reset, per-model breakdown
- [x] **`prellm doctor`** CLI — check config, providers, files
- [x] **`prellm config`** CLI — set/get/list/show/init-env config management
- [x] **`prellm models`** CLI — browse model pairs + OpenRouter catalog
- [x] **SensitiveDataFilter** — recursive context sanitization before large LLM
- [x] **nfo structured logging** — `@log_call` / `@catch` decorators across pipeline
- [x] **FolderCompressor** — .toon format project compression for LLM context

### Month 5–7: Proxy Server & Docker
- [x] FastAPI proxy server (OpenAI-compatible `/v1/chat/completions`)
- [x] Streaming SSE support with stage progress
- [x] Docker image builds and runs (`docker build -t prellm .`)
- [x] Docker test sandbox (`Dockerfile.test` + `docker-compose.test.yml`)
- [x] `docker-compose.yaml` with Ollama + preLLM + auto model pull
- [ ] Docker Hub: published `prellm/prellm` image with stable tags
- [ ] Load testing benchmarks
- [ ] ChromaDB user context memory (upgrade from SQLite MVP)

### Month 7–12: Enterprise (target: 100k+ ⭐)
- [ ] Kubernetes operator
- [ ] Multi-tenant (projects/teams)
- [ ] VSCode + Cursor extensions
- [ ] LangChain + LlamaIndex + Haystack integrations
- [ ] 1-click Vercel/Netlify deploy

## Docker Deployment

```bash
# Build and run
docker build -t prellm .
docker run -p 8080:8080 prellm serve

# Query mode
docker run --rm prellm query "Deploy app" --small ollama/qwen2.5:3b --large gpt-4o-mini

# Full stack with Ollama
docker-compose up -d
```

```bash
# OpenAI-compatible API
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "prellm:qwen→claude", "messages": [{"role": "user", "content": "Deploy app"}]}'
```

## Docker Test Sandbox

```bash
# Run full test suite in Docker
docker-compose -f docker-compose.test.yml run --rm test

# Lint check in Docker
docker-compose -f docker-compose.test.yml run --rm lint

# Verify production image builds
docker-compose -f docker-compose.test.yml run --rm build-check
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
