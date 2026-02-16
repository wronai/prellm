# preLLM Execution Trace

> **Query**: `Zdeployuj apkę na prod`
> **Timestamp**: 2026-02-16 09:26:05
> **Total duration**: 15813ms

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
- **Duration**: 3463ms
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
- **Duration**: 12338ms
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
  "content_preview": "Oto krótki przewodnik, jak wdrożyć aplikację na produkcję (prod):\n\n**Krok 1: Przygotuj aplikację**\n\n1. Upewnij się, że twoja aplikacja jest gotowa do wdrożenia na produkcję.\n2. Zakończ wszystkie niezbędne testy i debugowanie.\n3. Zapisz wszystkie zmiany w kodzie i zatwierdź je w systemie kontroli wer",
  "model": "openrouter/meta-llama/llama-3.3-70b-instruct"
}
```
</details>

- **retries**: `0`

---

## Result

**Response** (1761 chars):

```
Oto krótki przewodnik, jak wdrożyć aplikację na produkcję (prod):

**Krok 1: Przygotuj aplikację**

1. Upewnij się, że twoja aplikacja jest gotowa do wdrożenia na produkcję.
2. Zakończ wszystkie niezbędne testy i debugowanie.
3. Zapisz wszystkie zmiany w kodzie i zatwierdź je w systemie kontroli wersji (np. Git).

**Krok 2: Wybierz platformę wdrożeniową**

1. Wybierz platformę, na której chcesz wdrożyć aplikację (np. AWS, Google Cloud, Azure, itp.).
2. Upewnij się, że platforma jest odpowiednio ...
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
| 5 | PreprocessorAgent.preprocess() | `agent` | 3463ms | ✅ ok |
| 6 | ExecutorAgent.execute() | `llm_call` | 12338ms | ✅ ok |

**Total**: 15813ms
