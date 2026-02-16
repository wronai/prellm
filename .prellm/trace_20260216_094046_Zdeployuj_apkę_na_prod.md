# preLLM Execution Trace

> **Query**: `Zdeployuj apkę na prod`
> **Timestamp**: 2026-02-16 09:40:46
> **Total duration**: 94381ms

## Configuration

| Parameter | Value |
|---|---|
| `small_llm` | `ollama/qwen:7b` |
| `large_llm` | `openrouter/meta-llama/llama-3.3-70b-instruct` |
| `strategy` | `classify` |

## Decision Path

### Step 1: Configuration ✅

Resolved models, strategy, and pipeline parameters.

- **Type**: `config`
- **Status**: ok

<details>
<summary>Outputs</summary>

```json
{
  "small_llm": "ollama/qwen:7b",
  "large_llm": "openrouter/meta-llama/llama-3.3-70b-instruct",
  "strategy": "classify",
  "config_path": null,
  "user_context": null
}
```
</details>

---

### Step 2: Pipeline: classify ✅

llm step in 'classify' pipeline

- **Type**: `llm_call`
- **Status**: ok

<details>
<summary>Outputs</summary>

```json
{
  "classification": {
    "intent": "deploy",
    "confidence": 0.9,
    "domain": "mobile"
  }
}
```
</details>

---

### Step 3: Pipeline: match_rule ✅

algo step in 'classify' pipeline

- **Type**: `pipeline_step`
- **Status**: ok

<details>
<summary>Outputs</summary>

```json
{
  "matched_rule": {}
}
```
</details>

---

### Step 4: Pipeline: enrich_if_needed ⏭️

llm step in 'classify' pipeline

- **Type**: `llm_call`
- **Status**: skipped

<details>
<summary>Outputs</summary>

```json
{
  "enriched_query": null
}
```
</details>

---

### Step 5: PreprocessorAgent.preprocess() ✅

Small LLM (ollama/qwen:7b) preprocessed query using 'classify' strategy.

- **Type**: `agent`
- **Duration**: 3677ms
- **Status**: ok

<details>
<summary>Inputs</summary>

```json
{
  "query": "Zdeployuj apkę na prod",
  "pipeline": "classify",
  "user_context": {}
}
```
</details>

<details>
<summary>Outputs</summary>

```json
{
  "executor_input": "Zdeployuj apkę na prod"
}
```
</details>

---

### Step 6: ExecutorAgent.execute() ✅

Large LLM (openrouter/meta-llama/llama-3.3-70b-instruct) generated final response.

- **Type**: `llm_call`
- **Duration**: 90691ms
- **Status**: ok

<details>
<summary>Inputs</summary>

```json
{
  "executor_input": "Zdeployuj apkę na prod"
}
```
</details>

<details>
<summary>Outputs</summary>

```json
{
  "content_preview": "Żeby wdrożyć aplikację (apk) na produkcję (prod), musisz wykonać kilka kroków. Poniżej przedstawiam ogólny przewodnik, który może się różnić w zależności od Twojej aplikacji, technologii i środowiska produkcyjnego. Przypuszczam, że chodzi o aplikację mobilną na Androida, ale proces może być podobny ",
  "model": "openrouter/meta-llama/llama-3.3-70b-instruct"
}
```
</details>

- **retries**: `0`

---

## Result

**Response** (2473 chars):

```
Żeby wdrożyć aplikację (apk) na produkcję (prod), musisz wykonać kilka kroków. Poniżej przedstawiam ogólny przewodnik, który może się różnić w zależności od Twojej aplikacji, technologii i środowiska produkcyjnego. Przypuszczam, że chodzi o aplikację mobilną na Androida, ale proces może być podobny dla innych platform.

### 1. Przygotowanie aplikacji
- **Zbuduj aplikację w trybie produkcyjnym**: Upewnij się, że Twoja aplikacja jest zbudowana w trybie produkcyjnym, a nie debugowym. W Android Stud...
```

- **model_used**: `openrouter/meta-llama/llama-3.3-70b-instruct`
- **small_model_used**: `ollama/qwen:7b`
- **retries**: `0`
- **strategy**: `classify`
- **classification**: `{'intent': 'deploy', 'confidence': 0.9, 'domain': 'mobile'}`

## Summary

| # | Step | Type | Duration | Status |
|---|---|---|---|---|
| 1 | Configuration | `config` | — | ✅ ok |
| 2 | Pipeline: classify | `llm_call` | — | ✅ ok |
| 3 | Pipeline: match_rule | `pipeline_step` | — | ✅ ok |
| 4 | Pipeline: enrich_if_needed | `llm_call` | — | ⏭️ skipped |
| 5 | PreprocessorAgent.preprocess() | `agent` | 3677ms | ✅ ok |
| 6 | ExecutorAgent.execute() | `llm_call` | 90691ms | ✅ ok |

**Total**: 94381ms
