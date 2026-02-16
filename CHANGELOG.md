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


