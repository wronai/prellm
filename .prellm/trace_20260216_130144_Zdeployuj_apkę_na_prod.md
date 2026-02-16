# preLLM Execution Trace

> **Query**: `Zdeployuj apkę na prod`
> **Timestamp**: 2026-02-16 13:01:44
> **Total duration**: 25812ms

## Configuration

| Parameter | Value |
|---|---|
| `small_llm` | `ollama/qwen:7b` |
| `large_llm` | `openrouter/google/gemini-3-flash-preview` |
| `strategy` | `auto` |

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
  "strategy": "auto",
  "config_path": null,
  "user_context": null
}
```
</details>

---

### Step 2: Pipeline: classify ✅

llm step in 'auto' pipeline

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

algo step in 'auto' pipeline

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

### Step 4: Pipeline: compose ✅

llm step in 'auto' pipeline

- **Type**: `llm_call`
- **Status**: ok

<details>
<summary>Outputs</summary>

```json
{
  "composed_prompt": {
    "composed_prompt": "Zaprojektuj i skonfiguruj aplikację w formie APK, aby ją mogli zastosować na produkcyjnym środowisku ('prod'). Upewnij się, że korzystasz z odpowiednich narzędzi takich jak Git, Docker czy Kubernetes. Przykładowo, możesz wykorzystać Git do kontroli wersji kodu, a Docker do stworzenia i zarządzania chrootem dla aplikacji. Należy również dostosować konfigurację aplikacji do wymagań prod环境."
  }
}
```
</details>

---

### Step 5: PreprocessorAgent.preprocess() ✅

Small LLM (ollama/qwen:7b) preprocessed query using 'auto' strategy.

- **Type**: `agent`
- **Duration**: 13432ms
- **Status**: ok

<details>
<summary>Inputs</summary>

```json
{
  "query": "Zdeployuj apkę na prod",
  "pipeline": "auto",
  "user_context": {
    "context_schema": "{\"execution_env\":\"shell\",\"platform\":\"linux\",\"project_type\":null,\"project_summary\":null,\"available_tools\":[\"git\",\"docker\",\"kubectl\",\"terraform\",\"ansible\",\"npm\",\"pnpm\",\"pip\",\"poetry\",\"cargo\",\"make\",\"cmake\",\"gcc\",\"rustc\",\"node\",\"python3\",\"curl\",\"wget\",\"jq\",\"ssh\",\"rsync\"],\"locale\":\"en_US.UTF-8\",\"timezone\":\"CET\",\"user_history_summary\":null,\"sensitive_fields_blocked\":0,\"schema_token_cost\":97}"
  }
}
```
</details>

<details>
<summary>Outputs</summary>

```json
{
  "executor_input": "Zdeployuj apkę na prod\n\nZaprojektuj i skonfiguruj aplikację w formie APK, aby ją mogli zastosować na produkcyjnym środowisku ('prod'). Upewnij się, że korzystasz z odpowiednich narzędzi takich jak Git, Docker czy Kubernetes. Przykładowo, możesz wykorzystać Git do kontroli wersji kodu, a Docker do stworzenia i zarządzania chrootem dla aplikacji. Należy również dostosować konfigurację aplikacji do wymagań prod环境."
}
```
</details>

---

### Step 6: ExecutorAgent.execute() ✅

Large LLM (openrouter/google/gemini-3-flash-preview) generated final response.

- **Type**: `llm_call`
- **Duration**: 12314ms
- **Status**: ok

<details>
<summary>Inputs</summary>

```json
{
  "executor_input": "Zdeployuj apkę na prod\n\nZaprojektuj i skonfiguruj aplikację w formie APK, aby ją mogli zastosować na produkcyjnym środowisku ('prod'). Upewnij się, że korzystasz z odpowiednich narzędzi takich jak Git, Docker czy Kubernetes. Przykładowo, możesz wykorzystać Git do kontroli wersji kodu, a Docker do st"
}
```
</details>

<details>
<summary>Outputs</summary>

```json
{
  "content_preview": "Wdrożenie aplikacji mobilnej (Android) w środowisku produkcyjnym wymaga połączenia automatyzacji CI/CD (Git, Docker) z odpowiednim podpisaniem i optymalizacją pliku APK.\n\nPoniżej znajduje się kompletny projekt procesu wdrożeniowego (Pipeline) dla aplikacji opartej na React Native lub Flutterze, dost",
  "model": "openrouter/google/gemini-3-flash-preview"
}
```
</details>

- **retries**: `0`

---

## Result

**Response** (4079 chars):

```
Wdrożenie aplikacji mobilnej (Android) w środowisku produkcyjnym wymaga połączenia automatyzacji CI/CD (Git, Docker) z odpowiednim podpisaniem i optymalizacją pliku APK.

Poniżej znajduje się kompletny projekt procesu wdrożeniowego (Pipeline) dla aplikacji opartej na React Native lub Flutterze, dostosowany do Twojego środowiska.

### 1. Kontrola wersji (Git)
Zasady dla środowiska `prod`:
- Używamy gałęzi `main` lub `production`.
- Stosujemy tagowanie wersji (SemVer), np. `v1.0.0`.

```bash
git c...
```

- **model_used**: `openrouter/google/gemini-3-flash-preview`
- **small_model_used**: `ollama/qwen:7b`
- **retries**: `0`
- **strategy**: `auto`
- **classification**: `{'intent': 'deploy', 'confidence': 0.9, 'domain': 'mobile'}`

## Summary

| # | Step | Type | Duration | Status |
|---|---|---|---|---|
| 1 | Configuration | `config` | — | ✅ ok |
| 2 | Pipeline: classify | `llm_call` | — | ✅ ok |
| 3 | Pipeline: match_rule | `pipeline_step` | — | ✅ ok |
| 4 | Pipeline: compose | `llm_call` | — | ✅ ok |
| 5 | PreprocessorAgent.preprocess() | `agent` | 13432ms | ✅ ok |
| 6 | ExecutorAgent.execute() | `llm_call` | 12314ms | ✅ ok |

**Total**: 25812ms
