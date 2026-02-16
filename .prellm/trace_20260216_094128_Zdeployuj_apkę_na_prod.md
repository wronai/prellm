# preLLM Execution Trace

> **Query**: `Zdeployuj apkę na prod`
> **Timestamp**: 2026-02-16 09:41:28
> **Total duration**: 7814ms

## Configuration

| Parameter | Value |
|---|---|
| `small_llm` | `ollama/qwen:7b` |
| `large_llm` | `openrouter/google/gemini-3-flash-preview` |
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
  "large_llm": "openrouter/google/gemini-3-flash-preview",
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
- **Duration**: 1184ms
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

Large LLM (openrouter/google/gemini-3-flash-preview) generated final response.

- **Type**: `llm_call`
- **Duration**: 6619ms
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
  "content_preview": "Żebyśmy mogli to zrobić sprawnie i bezpiecznie, muszę wiedzieć, z czym pracujemy. \"Deployment na prod\" to szerokie pojęcie, więc przejdźmy przez szybki proces decyzyjny.\n\nOdpowiedz na te kilka pytań, a podam Ci gotową listę komend:\n\n### 1. Jaki to stos technologiczny (Stack)?\n*   **Frontend:** React",
  "model": "openrouter/google/gemini-3-flash-preview"
}
```
</details>

- **retries**: `0`

---

## Result

**Response** (1552 chars):

```
Żebyśmy mogli to zrobić sprawnie i bezpiecznie, muszę wiedzieć, z czym pracujemy. "Deployment na prod" to szerokie pojęcie, więc przejdźmy przez szybki proces decyzyjny.

Odpowiedz na te kilka pytań, a podam Ci gotową listę komend:

### 1. Jaki to stos technologiczny (Stack)?
*   **Frontend:** React, Vue, Next.js, zwykły HTML?
*   **Backend:** Node.js, Python (Django/FastAPI), PHP (Laravel), Go?
*   **Baza danych:** PostgreSQL, MongoDB, MySQL, czy brak?

### 2. Gdzie to ma wylądować?
*   **PaaS ...
```

- **model_used**: `openrouter/google/gemini-3-flash-preview`
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
| 5 | PreprocessorAgent.preprocess() | `agent` | 1184ms | ✅ ok |
| 6 | ExecutorAgent.execute() | `llm_call` | 6619ms | ✅ ok |

**Total**: 7814ms
