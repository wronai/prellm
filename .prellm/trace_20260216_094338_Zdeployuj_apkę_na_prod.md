# preLLM Execution Trace

> **Query**: `Zdeployuj apkę na prod`
> **Timestamp**: 2026-02-16 09:43:38
> **Total duration**: 8056ms

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
- **Duration**: 1201ms
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
- **Duration**: 6845ms
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
  "content_preview": "Aby „zdeployować apkę na prod”, musimy najpierw ustalić, co to za aplikacja i jaką infrastrukturą dysponujesz. Ponieważ nie podałeś szczegółów, przygotowałem zestawienie **4 najpopularniejszych scenariuszy**.\n\nWybierz ten, który najbardziej Ci odpowiada:\n\n---\n\n### Scenariusz 1: Najszybszy (PaaS - Pl",
  "model": "openrouter/google/gemini-3-flash-preview"
}
```
</details>

- **retries**: `0`

---

## Result

**Response** (2616 chars):

```
Aby „zdeployować apkę na prod”, musimy najpierw ustalić, co to za aplikacja i jaką infrastrukturą dysponujesz. Ponieważ nie podałeś szczegółów, przygotowałem zestawienie **4 najpopularniejszych scenariuszy**.

Wybierz ten, który najbardziej Ci odpowiada:

---

### Scenariusz 1: Najszybszy (PaaS - Platform as a Service)
Idealny dla aplikacji frontendowych (React, Vue, Next.js) lub prostych backendów (Node.js, Python, Go).

*   **Narzędzia:** Vercel, Netlify, Railway, Render.
*   **Kroki:**
    1....
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
| 5 | PreprocessorAgent.preprocess() | `agent` | 1201ms | ✅ ok |
| 6 | ExecutorAgent.execute() | `llm_call` | 6845ms | ✅ ok |

**Total**: 8056ms
