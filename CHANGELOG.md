## [0.4.4] - 2026-02-16

### Summary

feat(tests): configuration management system

### Test

- update tests/test_one_function.py
- update tests/test_server.py
- update tests/test_v04_context.py

### Other

- update .prellm/budget.json
- update ".prellm/trace_20260216_120904_Zdeployuj_apk\304\231_na_prod.md"
- config: update prompts.yaml
- update prellm/agents/preprocessor.py
- update prellm/core.py


## [0.4.3] - 2026-02-16

### Summary

feat(config): configuration management system

### Test

- update tests/test_env_config.py
- update tests/test_one_function.py
- update tests/test_server.py
- update tests/test_v04_context.py

### Other

- update .prellm/budget.json
- update ".prellm/trace_20260216_111350_Zdeployuj_apk\304\231_na_prod.md"
- config: update pipelines.yaml
- config: update prompts.yaml
- update prellm/core.py
- update prellm/env_config.py
- update prellm/llm_provider.py


## [0.4.2] - 2026-02-16

### Summary

feat(docs): configuration management system

### Docs

- docs: update README
- docs: update ROADMAP.md
- docs: update flow-graphs.md
- docs: update persistent-context.md
- docs: update sensitive-data.md
- docs: update session-persistence.md

### Test

- update tests/test_v04_context.py
- update tests/test_v04_pipeline.py
- update tests/test_v04_session.py

### Build

- update pyproject.toml

### Other

- update prellm/context/user_memory.py


## [0.4.0] - 2026-02-16

### Summary

**Persistent context layer for small LLMs** — preLLM automatically collects env, compresses codebase,
persists sessions, and injects everything into small-LLM prompts without manual pre-prompts.
Sensitive data never reaches the large-LLM. Solves the "Bielik drifts after 5–10 exchanges" problem.

> **Docs:** [Persistent Context](docs/persistent-context.md) · [Session Persistence](docs/session-persistence.md) · [Sensitive Data](docs/sensitive-data.md) · [Flow Graphs](docs/flow-graphs.md)

### Added

- **`RuntimeContext` model** (`prellm/models.py`) — unified env/process/locale/network/git/system snapshot with `token_estimate` and `sensitive_blocked_count`
- **`SessionSnapshot` model** (`prellm/models.py`) — exportable session with `to_file()`/`from_file()` for persistence across restarts
- **`ContextEngine.gather_runtime()`** — collects full runtime context as Pydantic model, auto-filters sensitive env vars via `SensitiveDataFilter`
- **`ContextEngine._auto_collect_env()`** — auto-discovery of all env vars with safe/masked/blocked classification
- **`ContextEngine._gather_process()`** — PID, CWD, user, parent PID, TTY
- **`ContextEngine._gather_locale()`** — LANG, timezone, encoding (critical for Bielik/Polish locale)
- **`ContextEngine._gather_network()`** — hostname, local IP (no public IP queries)
- **`UserMemory.export_session()`** — export current session to `SessionSnapshot` (like LM Studio 'save session')
- **`UserMemory.import_session()`** — import previously exported session
- **`UserMemory.get_relevant_context()`** — RAG-style retrieval from history with token budget
- **`UserMemory.auto_inject_context()`** — build enriched system_prompt from history + preferences
- **`UserMemory.learn_preference_from_interaction()`** — auto-extract preferences (like Oobabooga Dynamic Context)
- **`CodebaseIndexer.get_compressed_context()`** — full pipeline: index → compress → filter by query relevance with token budget
- **`CodebaseIndexer.estimate_tokens()`** — token estimation helper
- **3 new pipeline algo handlers** (`prellm/pipeline.py`): `runtime_collector`, `sensitive_filter`, `session_injector`
- **`context_aware` pipeline** (`configs/pipelines.yaml`) — 6-step context-aware pipeline with auto-strategy
- **`auto_strategy` prompt** (`configs/prompts.yaml`) — small-LLM strategy selection based on query + runtime context
- **CLI `prellm context show`** — inspect runtime context (`--json`, `--blocked`, `--codebase .`)
- **CLI `prellm session list|export|import|clear`** — manage persistent sessions
- **Trace `context_collection` step type** — runtime/session/sanitize steps visible in execution trace
- **47 new tests** across 3 test files (`test_v04_context.py`, `test_v04_session.py`, `test_v04_pipeline.py`)

### Changed

- **Default strategy changed from `"classify"` to `"auto"`** — small-LLM auto-selects best strategy
- **Default `sanitize=True`** — sensitive data filtered before large-LLM by default
- **New parameter `collect_runtime=True`** — full env/shell/process snapshot collected by default
- **New parameter `session_path`** — path to session persistence SQLite DB
- **New parameter `sensitive_rules`** — custom YAML rules for sensitive data classification
- **`_execute_v3_pipeline` refactored** — split from 1 method (cc≈29) into 5 methods (cc≤10 each): `_prepare_context`, `_run_preprocessing`, `_run_execution`, `_persist_session`, `_record_trace`
- **`__init__.py` exports** — added `RuntimeContext`, `SessionSnapshot`, `SensitiveDataFilter`, `ShellContextCollector`, `FolderCompressor`, `ContextSchemaGenerator`

### Fixed

- Version assertion tests updated for 0.4.0
- `UserMemory.export_session()` — `created_at` field correctly serialized as string

### Backward Compatibility

- All existing parameters and defaults preserved
- `strategy="classify"` still works as before
- `collect_runtime=True` and `sanitize=True` are new defaults but don't break existing usage (context is additive, sanitization only affects large-LLM input)
- `session_path=None` by default — no session persistence unless explicitly enabled


## [0.4.1] - 2026-02-16

### Summary

refactor(tests): CLI interface improvements

### Docs

- docs: update ROADMAP.md

### Test

- update tests/test_folder_compressor.py
- update tests/test_full_context_pipeline.py
- update tests/test_one_function.py
- update tests/test_schema_generator.py
- update tests/test_sensitive_filter.py
- update tests/test_server.py
- update tests/test_shell_collector.py

### Build

- update pyproject.toml

### Other

- update Dockerfile.test
- config: update pipelines.yaml
- config: update prompts.yaml
- config: update sensitive_rules.yaml
- update prellm/__init__.py
- update prellm/analyzers/context_engine.py
- update prellm/cli.py
- update prellm/context/codebase_indexer.py
- update prellm/context/folder_compressor.py
- update prellm/context/sensitive_filter.py
- ... and 4 more


## [0.3.15] - 2026-02-16

### Summary

feat(docs): CLI interface improvements

### Build

- update pyproject.toml

### Other

- update .prellm/budget.json
- update ".prellm/trace_20260216_103122_Zdeployuj_apk\304\231_na_prod.md"
- update prellm/agents/executor.py
- update prellm/cli.py
- update prellm/context/__init__.py
- update prellm/context/folder_compressor.py
- update prellm/context/schema_generator.py
- update prellm/core.py
- update prellm/query_decomposer.py
- update project.functions.toon
- ... and 1 more


## [0.3.14] - 2026-02-16

### Summary

fix(docs): CLI interface improvements

### Docs

- docs: update prellm-tasks-env-context-collection.md

### Test

- update tests/test_budget.py
- update tests/test_trace.py

### Build

- update pyproject.toml

### Ci

- config: update ci.yml

### Other

- update .prellm/budget.json
- update ".prellm/trace_20260216_092605_Zdeployuj_apk\304\231_na_prod.md"
- update ".prellm/trace_20260216_094046_Zdeployuj_apk\304\231_na_prod.md"
- update ".prellm/trace_20260216_094128_Zdeployuj_apk\304\231_na_prod.md"
- update ".prellm/trace_20260216_094338_Zdeployuj_apk\304\231_na_prod.md"
- update ".prellm/trace_20260216_101602_Zdeployuj_apk\304\231_na_prod.md"
- update Dockerfile.test
- build: update Makefile
- config: update docker-compose.test.yml
- update prellm/__init__.py
- ... and 16 more


## [0.3.13] - 2026-02-16

### Summary

docs(docs): code relationship mapping with 2 supporting modules

### Docs

- docs: update README
- docs: update flow-graphs.md


## [0.3.12] - 2026-02-15

### Summary

refactor(examples): deep code analysis engine with 3 supporting modules

### Docs

- docs: update README
- docs: update README
- docs: update README

### Other

- scripts: update curl_api.sh
- update examples/embedded/__init__.py
- update examples/embedded/main.py
- update examples/embedded/test_embedded.py
- update examples/k8s/__init__.py
- update examples/k8s/main.py
- update examples/k8s/test_k8s.py
- update examples/leasing/__init__.py
- update examples/leasing/main.py
- update examples/leasing/test_leasing.py
- ... and 3 more


## [0.3.11] - 2026-02-15

### Summary

feat(docs): CLI interface improvements

### Docs

- docs: update README

### Build

- update pyproject.toml

### Other

- scripts: update cli_examples.sh


## [0.3.10] - 2026-02-15

### Summary

refactor(examples): configuration management system

### Docs

- docs: update README
- docs: update README
- docs: update README

### Test

- update tests/test_one_function.py
- update tests/test_preprocessor_memory.py

### Other

- update examples/embedded/__init__.py
- update examples/embedded/main.py
- update examples/embedded/test_embedded.py
- update examples/k8s/__init__.py
- update examples/k8s/main.py
- update examples/k8s/test_k8s.py
- update examples/leasing/__init__.py
- update examples/leasing/main.py
- update examples/leasing/test_leasing.py
- update prellm/chains/process_chain.py


## [0.3.9] - 2026-02-15

### Summary

feat(goal): CLI interface improvements

### Test

- update tests/test_server.py

### Other

- update prellm/cli.py


## [0.3.7] - 2026-02-15

### Summary

feat(goal): CLI interface improvements

### Other

- update prellm/cli.py
- update prellm/env_config.py


## [0.3.6] - 2026-02-15

### Summary

refactor(tests): deep code analysis engine with 4 supporting modules

### Test

- update tests/test_codebase_indexer.py

### Other

- update prellm/analyzers/context_engine.py
- update prellm/context/__init__.py
- update prellm/context/codebase_indexer.py
- update prellm/context/user_memory.py


## [0.3.5] - 2026-02-15

### Summary

refactor(examples): CLI interface improvements

### Docs

- docs: update README
- docs: update ROADMAP.md

### Test

- update tests/test_one_function.py
- update tests/test_server.py
- update tests/test_structured_output.py

### Build

- update pyproject.toml

### Other

- config: update devops_k8s.yaml
- config: update embedded.yaml
- config: update polish_finance.yaml
- update examples/embedded_refactor.py
- update examples/k8s_debug.py
- update examples/polish_leasing.py
- update examples/providers.py
- update examples/quick_start.py
- update prellm/__init__.py
- update prellm/chains/process_chain.py
- ... and 3 more


## [0.3.4] - 2026-02-15

### Summary

feat(core): unified API with two-agent architecture, comprehensive docs, provider examples

### Added

- **Unified `preprocess_and_execute()` API** — single function with two execution paths:
  - `strategy=` (v0.2, default): strategy-based via PreLLM class
  - `pipeline=` (v0.3): two-agent via PreprocessorAgent + ExecutorAgent + PromptPipeline
- **PreprocessorAgent** (`prellm/agents/preprocessor.py`): small LLM preprocessing with PromptPipeline
- **ExecutorAgent** (`prellm/agents/executor.py`): large LLM execution with ResponseValidator
- **PromptRegistry** (`prellm/prompt_registry.py`): YAML-based prompt loading with Jinja2 templating
- **PromptPipeline** (`prellm/pipeline.py`): YAML-configurable multi-step preprocessing pipelines
- **ResponseValidator** (`prellm/validators.py`): YAML schema validation for LLM outputs
- **UserMemory** (`prellm/context/user_memory.py`): SQLite-backed user interaction history
- **Per-domain default configs**: `configs/defaults/{devops,coding,business,embedded}.yaml`
- **YAML configs**: `configs/prompts.yaml`, `configs/pipelines.yaml`, `configs/response_schemas.yaml`
- **6 built-in pipelines**: classify, structure, split, enrich, dual_agent_full, passthrough
- **65 new tests** across 7 test files (all passing)
- **12 integration tests** covering full two-agent pipeline flows
- **README rewrite** with provider examples (Ollama, OpenAI, Anthropic, Groq, Mistral, Azure, Bedrock, Gemini, Together, DeepSeek)
- **LiteLLM integration guide** — drop-in enhancement, env vars, proxy support
- **OpenAI SDK compatibility** — use preLLM server from any OpenAI client
- **Usage examples**: `examples/quick_start.py`, `examples/providers.py`

### Changed

- `preprocess_and_execute()` now accepts `pipeline=`, `prompts_path=`, `pipelines_path=`, `schemas_path=` parameters
- `preprocess_and_execute_sync()` forwards all new parameters
- `preprocess_and_execute_v3` is now a backward-compatible alias for `preprocess_and_execute`
- Server (`prellm/server.py`): supports `prellm.pipeline` field in request body
- ProcessChain (`prellm/chains/process_chain.py`): supports v0.3 pipeline per-step
- `__init__.py` exports all v0.3 modules (agents, pipeline, registry, validators, memory)
- Version bumped to 0.3.4

### Fixed

- Version assertion tests updated for new version


## [0.3.3] - 2026-02-15

### Summary

fix(tests): configuration management system

### Test

- update tests/test_executor_agent.py
- update tests/test_one_function.py
- update tests/test_preprocessor_agent.py
- update tests/test_server.py
- update tests/test_user_memory.py
- update tests/test_v3_architecture.py
- update tests/test_validators.py

### Other

- config: update business.yaml
- config: update coding.yaml
- config: update devops.yaml
- config: update embedded.yaml
- update prellm/__init__.py
- update prellm/agents/__init__.py
- update prellm/agents/executor.py
- update prellm/agents/preprocessor.py
- update prellm/chains/process_chain.py
- update prellm/context/__init__.py
- ... and 6 more


## [0.3.2] - 2026-02-15

### Summary

fix(config): CLI interface improvements

### Docs

- docs: update flow-graphs.md

### Test

- update tests/test_env_config.py
- update tests/test_pipeline.py
- update tests/test_prompt_registry.py

### Build

- update pyproject.toml

### Other

- update .env.example
- config: update pipelines.yaml
- config: update prompts.yaml
- config: update response_schemas.yaml
- config: update docker-compose.yaml
- scripts: update cli_examples.sh
- scripts: update curl_api.sh
- update examples/python_sdk.py
- update prellm/cli.py
- update prellm/env_config.py
- ... and 5 more


## [0.3.1] - 2026-02-15

### Summary

fix(docs): CLI interface improvements

### Docs

- docs: update README
- docs: update ROADMAP.md

### Test

- update tests/test_one_function.py
- update tests/test_server.py

### Build

- update pyproject.toml

### Other

- update prellm/__init__.py
- update prellm/cli.py
- update prellm/core.py
- update prellm/server.py
- update project.functions.toon
- update project.toon


## [0.2.1] - 2026-02-15

### Summary

feat(tests): CLI interface improvements

### Test

- update tests/test_context.py
- update tests/test_decomposer.py
- update tests/test_integration.py
- update tests/test_models.py
- update tests/test_v2_architecture.py

### Build

- update pyproject.toml

### Other

- config: update config.yaml
- config: update deploy.yaml
- config: update prellm_config.yaml
- update prellm/__init__.py
- update prellm/chains/process_chain.py
- update prellm/cli.py
- update prellm/core.py
- update prellm/llm_provider.py
- update prellm/models.py
- update prellm/query_decomposer.py


## [0.1.15] - 2026-02-15

### Summary

feat(None): configuration management system

### Other

- update project.functions.toon
- scripts: update project.sh
- update project.toon-schema.json


## [0.1.14] - 2026-02-15

### Summary

chore: fix testing infrastructure

### Docs

- docs: update README
- docs: update promptguard-status.md

### Test

- update tests/test_integration.py
- update tests/test_promptguard.py

### Build

- update pyproject.toml

### Config

- config: update goal.yaml

### Other

- build: update Makefile
- update list.txt
- update prellm/__init__.py
- update prellm/chains/process_chain.py
- update prellm/cli.py
- update prellm/core.py


## [0.1.13] - 2026-02-15

### Summary

refactor(docs): CLI interface improvements

### Docs

- docs: update README

### Test

- update tests/test_edge_cases.py
- update tests/test_integration.py
- update tests/test_prellm.py

### Build

- update pyproject.toml

### Config

- config: update goal.yaml

### Other

- build: update Makefile
- update bumpver.toml
- config: update deploy.yaml
- config: update rules.yaml
- update list.txt
- update prellm/__init__.py
- update prellm/analyzers/__init__.py
- update prellm/analyzers/bias_detector.py
- update prellm/analyzers/context_engine.py
- update prellm/chains/__init__.py
- ... and 5 more


## [0.1.5] - 2026-02-15

### Summary

feat(build): code relationship mapping with 4 supporting modules

### Build

- update pyproject.toml

### Config

- config: update goal.yaml

### Other

- build: update Makefile
- update bumpver.toml
- update prellm/__init__.py


## [0.1.1] - 2026-02-15

### Summary

refactor(tests): CLI interface improvements

### Docs

- docs: update README
- docs: update prellm-status.md

### Test

- update tests/__init__.py
- update tests/test_edge_cases.py
- update tests/test_integration.py
- update tests/test_prellm.py

### Build

- update pyproject.toml

### Config

- config: update goal.yaml

### Other

- update .gitignore
- update LICENSE
- build: update Makefile
- config: update deploy.yaml
- config: update rules.yaml
- update prellm/__init__.py
- update prellm/analyzers/__init__.py
- update prellm/analyzers/bias_detector.py
- update prellm/analyzers/context_engine.py
- update prellm/chains/__init__.py
- ... and 5 more


