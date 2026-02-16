# preLLM Execution Trace

> **Query**: `Zdeployuj apkę na prod`
> **Timestamp**: 2026-02-16 12:19:25
> **Total duration**: 21333ms

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

### Step 5: Pipeline: compose ✅

llm step in 'classify' pipeline

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

### Step 6: PreprocessorAgent.preprocess() ✅

Small LLM (ollama/qwen:7b) preprocessed query using 'classify' strategy.

- **Type**: `agent`
- **Duration**: 12765ms
- **Status**: ok

<details>
<summary>Inputs</summary>

```json
{
  "query": "Zdeployuj apkę na prod",
  "pipeline": "classify",
  "user_context": {
    "shell_context": "{\"env_vars\":{\"SHELL\":\"/bin/bash\",\"QT_ACCESSIBILITY\":\"1\",\"SNAP_REVISION\":\"578\",\"PYENV_SHELL\":\"bash\",\"XDG_CONFIG_DIRS\":\"/etc/xdg/xdg-ubuntu:/etc/xdg\",\"NVM_INC\":\"/home/tom/.nvm/versions/node/v20.19.5/include/node\",\"XDG_MENU_PREFIX\":\"gnome-\",\"CONDA_EXE\":\"/home/tom/miniconda3/bin/conda\",\"_CE_M\":\"\",\"QT_IM_MODULES\":\"wayland;ibus\",\"SNAP_REAL_HOME\":\"/home/tom\",\"TERMINAL_EMULATOR\":\"JetBrains-JediTerm\",\"SNAP_USER_COMMON\":\"/home/tom/snap/pycharm-professional/common\",\"PROCESS_LAUNCHED_BY_Q\":\"1\",\"MEMORY_PRESSURE_WRITE\":\"c29tZSAyMDAwMDAgMjAwMDAwMAA=\",\"SNAP_INSTANCE_KEY\":\"\",\"XMODIFIERS\":\"@im=ibus\",\"SNAP_EUID\":\"1000\",\"PWD\":\"/home/tom/github/wronai/prellm\",\"PYENV_VIRTUALENV_INIT\":\"1\",\"LOGNAME\":\"tom\",\"XDG_SESSION_TYPE\":\"wayland\",\"CONDA_PREFIX\":\"/home/tom/miniconda3\",\"PROCESS_LAUNCHED_BY_CW\":\"1\",\"PNPM_HOME\":\"/home/tom/.local/share/pnpm\",\"GPG_AGENT_INFO\":\"/run/user/1000/gnupg/S.gpg-agent:0:1\",\"SYSTEMD_EXEC_PID\":\"9894\",\"DESKTOP_STARTUP_ID\":\"gnome-shell/PyCharm/9894-0-nvidia_TIME50588\",\"SNAP_CONTEXT\":\"1VBG0g",
    "runtime_context": {
      "env_safe": {
        "SHELL": "/bin/bash",
        "QT_ACCESSIBILITY": "1",
        "SNAP_REVISION": "578",
        "PYENV_SHELL": "bash",
        "XDG_CONFIG_DIRS": "/etc/xdg/xdg-ubuntu:/etc/xdg",
        "NVM_INC": "/home/tom/.nvm/versions/node/v20.19.5/include/node",
        "XDG_MENU_PREFIX": "gnome-",
        "GNOME_DESKTOP_SESSION_ID": "this-is-deprecated",
        "CONDA_EXE": "/home/tom/miniconda3/bin/conda",
        "_CE_M": "",
        "QT_IM_MODULES": "wayland;ibus",
        "SNAP_REAL_HOME": "/home/tom",
        "TERMINAL_EMULATOR": "JetBrains-JediTerm",
        "SNAP_USER_COMMON": "/home/tom/snap/pycharm-professional/common",
        "PROCESS_LAUNCHED_BY_Q": "1",
        "GNOME_SHELL_SESSION_MODE": "ubuntu",
        "S
... (truncated)
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

### Step 7: ExecutorAgent.execute() ✅

Large LLM (openrouter/google/gemini-3-flash-preview) generated final response.

- **Type**: `llm_call`
- **Duration**: 8505ms
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
  "content_preview": "Wdrożenie aplikacji mobilnej (APK) na produkcję różni się od standardowych aplikacji webowych. Proces ten wymaga zbudowania powtarzalnego środowiska CI/CD (Continuous Integration/Deployment).\n\nPoniżej przedstawiam kompletną architekturę i konfigurację \"Pipeline'u\" produkcyjnego, wykorzystującą dostę",
  "model": "openrouter/google/gemini-3-flash-preview"
}
```
</details>

- **retries**: `0`

---

## Result

**Response** (3565 chars):

```
Wdrożenie aplikacji mobilnej (APK) na produkcję różni się od standardowych aplikacji webowych. Proces ten wymaga zbudowania powtarzalnego środowiska CI/CD (Continuous Integration/Deployment).

Poniżej przedstawiam kompletną architekturę i konfigurację "Pipeline'u" produkcyjnego, wykorzystującą dostępne u Ciebie narzędzia (**Git, Docker, Make**).

### 1. Kontrola wersji (Git)
Struktura gałęzi (Git Flow):
*   `main` – kod produkcyjny (zawsze stabilny).
*   `develop` – integracja funkcji.
*   Tagow...
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
| 5 | Pipeline: compose | `llm_call` | — | ✅ ok |
| 6 | PreprocessorAgent.preprocess() | `agent` | 12765ms | ✅ ok |
| 7 | ExecutorAgent.execute() | `llm_call` | 8505ms | ✅ ok |

**Total**: 21333ms
