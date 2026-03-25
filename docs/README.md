<!-- code2docs:start --># prellm

![version](https://img.shields.io/badge/version-0.1.0-blue) ![python](https://img.shields.io/badge/python-%3E%3D3.9-blue) ![coverage](https://img.shields.io/badge/coverage-unknown-lightgrey) ![functions](https://img.shields.io/badge/functions-310-green)
> **310** functions | **80** classes | **39** files | CC̄ = 4.6

> Auto-generated project documentation from source code analysis.

**Author:** Tom Softreck <tom@sapletta.com>  
**License:** MIT[(LICENSE)](./LICENSE)  
**Repository:** [https://github.com/wronai/gllm](https://github.com/wronai/gllm)

## Installation

### From PyPI

```bash
pip install prellm
```

### From Source

```bash
git clone https://github.com/wronai/gllm
cd prellm
pip install -e .
```


## Quick Start

### CLI Usage

```bash
# Generate full documentation for your project
prellm ./my-project

# Only regenerate README
prellm ./my-project --readme-only

# Preview what would be generated (no file writes)
prellm ./my-project --dry-run

# Check documentation health
prellm check ./my-project

# Sync — regenerate only changed modules
prellm sync ./my-project
```

### Python API

```python
from prellm import generate_readme, generate_docs, Code2DocsConfig

# Quick: generate README
generate_readme("./my-project")

# Full: generate all documentation
config = Code2DocsConfig(project_name="mylib", verbose=True)
docs = generate_docs("./my-project", config=config)
```

## Generated Output

When you run `prellm`, the following files are produced:

```
<project>/
├── README.md                 # Main project README (auto-generated sections)
├── docs/
│   ├── api.md               # Consolidated API reference
│   ├── modules.md           # Module documentation with metrics
│   ├── architecture.md      # Architecture overview with diagrams
│   ├── dependency-graph.md  # Module dependency graphs
│   ├── coverage.md          # Docstring coverage report
│   ├── getting-started.md   # Getting started guide
│   ├── configuration.md    # Configuration reference
│   └── api-changelog.md    # API change tracking
├── examples/
│   ├── quickstart.py       # Basic usage examples
│   └── advanced_usage.py   # Advanced usage examples
├── CONTRIBUTING.md         # Contribution guidelines
└── mkdocs.yml             # MkDocs site configuration
```

## Configuration

Create `prellm.yaml` in your project root (or run `prellm init`):

```yaml
project:
  name: my-project
  source: ./
  output: ./docs/

readme:
  sections:
    - overview
    - install
    - quickstart
    - api
    - structure
  badges:
    - version
    - python
    - coverage
  sync_markers: true

docs:
  api_reference: true
  module_docs: true
  architecture: true
  changelog: true

examples:
  auto_generate: true
  from_entry_points: true

sync:
  strategy: markers    # markers | full | git-diff
  watch: false
  ignore:
    - "tests/"
    - "__pycache__"
```

## Sync Markers

prellm can update only specific sections of an existing README using HTML comment markers:

```markdown
<!-- prellm:start -->
# Project Title
... auto-generated content ...
<!-- prellm:end -->
```

Content outside the markers is preserved when regenerating. Enable this with `sync_markers: true` in your configuration.

## Architecture

```
prellm/
    ├── quick_start    ├── polish_leasing    ├── k8s_debug    ├── python_sdk    ├── embedded_refactor    ├── providers    ├── model_catalog    ├── cli    ├── env_config├── prellm/    ├── trace    ├── prompt_registry    ├── query_decomposer    ├── models    ├── core    ├── llm_provider    ├── pipeline    ├── validators    ├── server    ├── chains/    ├── budget        ├── process_chain    ├── config_wizard        ├── folder_compressor        ├── shell_collector    ├── context/        ├── sensitive_filter        ├── user_memory    ├── analyzers/        ├── schema_generator        ├── codebase_indexer    ├── agents/        ├── executor├── project    ├── cli_examples    ├── curl_api        ├── preprocessor        ├── context_engine    ├── logging_setup```

## API Overview

### Classes

- **`EnvConfig`** — Resolved environment configuration.
- **`TraceStep`** — A single recorded step in the execution trace.
- **`TraceRecorder`** — Records execution trace and generates markdown documentation.
- **`PromptNotFoundError`** — Raised when a prompt name is not found in the registry.
- **`PromptRenderError`** — Raised when a prompt template fails to render.
- **`PromptEntry`** — Single prompt entry with template, max_tokens, and temperature.
- **`PromptRegistry`** — Loads prompts from YAML, caches, validates placeholders.
- **`QueryDecomposer`** — Decomposes user queries using a small LLM before routing to a large model.
- **`SensitivityLevel`** — —
- **`ProcessInfo`** — —
- **`LocaleInfo`** — —
- **`ShellInfo`** — —
- **`NetworkContext`** — —
- **`ShellContext`** — —
- **`ContextSchema`** — —
- **`FilterReport`** — —
- **`CompressedFolder`** — —
- **`RuntimeContext`** — Full runtime snapshot — env, process, locale, network, git, system.
- **`SessionSnapshot`** — Exportable session snapshot — enables persistent context across restarts.
- **`Policy`** — —
- **`ApprovalMode`** — —
- **`StepStatus`** — —
- **`DecompositionStrategy`** — Strategy for how the small LLM preprocesses a query.
- **`BiasPattern`** — —
- **`ModelConfig`** — —
- **`GuardConfig`** — Top-level YAML config model (v0.1 compat).
- **`AnalysisResult`** — Result of query analysis (v0.1 compat).
- **`GuardResponse`** — Response from Prellm (v0.1 compat).
- **`DomainRule`** — Configurable domain rule — keywords, intent, required fields, enrich template.
- **`LLMProviderConfig`** — Configuration for a single LLM provider (small or large).
- **`DecompositionPrompts`** — System prompts for each decomposition step — all configurable via YAML.
- **`PreLLMConfig`** — Top-level config for preLLM v0.2 — fully YAML-driven.
- **`ClassificationResult`** — Output of the CLASSIFY step.
- **`StructureResult`** — Output of the STRUCTURE step.
- **`DecompositionResult`** — Full result of the small LLM decomposition pipeline.
- **`PreLLMResponse`** — Response from preLLM v0.2 — includes decomposition + large LLM output.
- **`ProcessStep`** — —
- **`ProcessConfig`** — —
- **`StepResult`** — Result of a single process chain step.
- **`ProcessResult`** — Result of a full process chain execution.
- **`AuditEntry`** — Single audit log entry for traceability.
- **`PreLLM`** — preLLM v0.2/v0.3 — small LLM decomposition before large LLM routing.
- **`LLMProvider`** — Unified LLM caller with retry and fallback support.
- **`PipelineStep`** — Configuration for a single pipeline step.
- **`PipelineConfig`** — Configuration for a complete pipeline.
- **`StepExecutionResult`** — Result of executing a single pipeline step.
- **`PipelineResult`** — Result of executing a full pipeline.
- **`PromptPipeline`** — Generic pipeline — executes a sequence of LLM + algorithmic steps.
- **`ValidationResult`** — Result of validating data against a schema.
- **`SchemaDefinition`** — Parsed schema definition from YAML.
- **`ResponseValidator`** — Validates LLM responses against YAML-defined schemas.
- **`ChatMessage`** — —
- **`PreLLMExtras`** — preLLM-specific extensions in the request body.
- **`ChatCompletionRequest`** — —
- **`ChatCompletionChoice`** — —
- **`UsageInfo`** — —
- **`PreLLMMeta`** — preLLM metadata in the response.
- **`ChatCompletionResponse`** — —
- **`BatchItem`** — —
- **`HealthResponse`** — —
- **`AuthMiddleware`** — Bearer token auth using LITELLM_MASTER_KEY. Skips auth if key is not set.
- **`BudgetExceededError`** — Raised when the monthly budget limit has been reached.
- **`UsageEntry`** — Single API call cost record.
- **`BudgetTracker`** — Tracks LLM API spend against a monthly budget.
- **`ProcessChain`** — Execute multi-step DevOps workflows with preLLM validation at each step.
- **`FolderCompressor`** — Compresses a project folder into a lightweight representation for LLM context.
- **`ShellContextCollector`** — Collects full shell environment context for LLM prompt enrichment.
- **`SensitiveDataFilter`** — Classifies and filters sensitive data from context before LLM calls.
- **`UserMemory`** — Stores user query history and learned preferences.
- **`ContextSchemaGenerator`** — Generates a structured context schema from available context sources.
- **`CodeSymbol`** — A code symbol extracted from source.
- **`FileIndex`** — Index of a single source file.
- **`CodebaseIndex`** — Full codebase index.
- **`CodebaseIndexer`** — Index a codebase using tree-sitter for AST-based symbol extraction.
- **`ExecutorResult`** — Output of the ExecutorAgent.
- **`ExecutorAgent`** — Agent execution — large LLM (>24B) executes structured tasks.
- **`PreprocessResult`** — Output of the PreprocessorAgent — structured input for the ExecutorAgent.
- **`PreprocessorAgent`** — Agent preprocessing — small LLM (≤24B) analyzes and structures queries.
- **`ContextEngine`** — Collects context from environment, git, and system for prompt enrichment.

### Functions

- `example_zero_config()` — Simplest possible usage — one line, default models.
- `example_strategy()` — Strategy-based preprocessing (classify, structure, split, enrich).
- `example_pipeline()` — Named pipeline — 4-step preprocessing for maximum quality.
- `example_ollama_local()` — Both models run locally via Ollama — $0.00 cost.
- `example_hybrid_ollama_openai()` — Hybrid: local preprocessing + cloud execution.
- `example_hybrid_ollama_anthropic()` — Hybrid: local preprocessing + Anthropic execution.
- `example_domain_rules()` — Domain rules catch missing safety-critical fields.
- `example_with_memory()` — UserMemory enriches queries with interaction history.
- `example_with_codebase()` — CodebaseIndexer enriches queries with relevant source symbols.
- `example_sync()` — Synchronous version — for scripts, notebooks, non-async code.
- `example_openai_sdk_client()` — Use preLLM server from any OpenAI SDK client.
- `main()` — Run all examples (requires LLM providers to be configured).
- `main()` — —
- `main()` — —
- `example_one_function()` — The simplest way to use preLLM — like litellm.completion().
- `example_domain_rules()` — Inline domain rules for safety checks.
- `example_sync()` — Synchronous wrapper — no async needed.
- `example_pipeline()` — Use a named YAML-defined pipeline for maximum preprocessing quality.
- `example_user_memory()` — Enrich queries with interaction history from UserMemory.
- `example_codebase_context()` — Enrich queries with codebase symbols from CodebaseIndexer.
- `example_class_based()` — More control with the PreLLM class.
- `example_custom_pipeline()` — Build a pipeline from components for maximum flexibility.
- `example_openai_sdk()` — Use preLLM as an OpenAI drop-in replacement.
- `example_strategies()` — Demonstrate all 5 decomposition strategies.
- `main()` — —
- `main()` — —
- `run_example(name, small_llm, large_llm)` — Run a single provider example.
- `ollama_local()` — Both models local via Ollama. Cost: $0.00
- `ollama_plus_openai()` — Local preprocessing, OpenAI execution. Cost: ~$0.15
- `ollama_plus_anthropic()` — Local preprocessing, Anthropic execution.
- `openai_only()` — Both models on OpenAI. Cost: ~$0.20
- `anthropic_only()` — Both models on Anthropic.
- `groq_fast()` — Groq for ultra-fast inference. Cost: very low.
- `mistral_cloud()` — Mistral AI cloud models.
- `azure_openai()` — Azure OpenAI deployments.
- `aws_bedrock()` — AWS Bedrock models.
- `google_gemini()` — Google Gemini models.
- `together_ai()` — Together AI hosted models.
- `deepseek()` — DeepSeek models.
- `openrouter_kimi()` — OpenRouter — access many providers through one API. Kimi K2.5 for strong reasoning.
- `mixed_providers_pipeline()` — Pipeline with mixed providers.
- `print_env_setup()` — Print required environment variables for each provider.
- `main()` — —
- `list_model_pairs(provider, search)` — Filter model pairs by provider and/or search term. Pure function — no IO.
- `list_openrouter_models(provider, search)` — Filter OpenRouter models by provider and/or search term. Pure function — no IO.
- `query(prompt, small, large, strategy)` — Preprocess a query with small LLM, then execute with large LLM.
- `context(json_output, schema, blocked, folder)` — Show collected environment context, schema, and blocked sensitive data.
- `process(config, guard_config, dry_run, json_output)` — Execute a DevOps process chain.
- `decompose(query, config, strategy, json_output)` — [v0.2] Decompose a query using small LLM without calling the large model.
- `init(output, devops)` — Generate a starter preLLM config file.
- `serve(host, port, small, large)` — Start the OpenAI-compatible API server.
- `doctor(env_file, live)` — Check configuration and provider connectivity.
- `config_set_cmd(key, value, global_)` — Set a config value persistently.
- `config_get_cmd(key, raw)` — Get a config value.
- `config_list_cmd(raw)` — List all configured values.
- `config_show_cmd()` — Show effective configuration (resolved from all sources).
- `config_init_env(global_, force)` — Generate a starter .env file with all available settings.
- `budget(reset, json_output)` — Show LLM API spend tracking and budget status.
- `models(provider, search)` — List popular model pairs and provider examples.
- `context_show_cmd(json_output, blocked, codebase)` — Show collected runtime context.
- `session_list_cmd(memory)` — List recent interactions in the session.
- `session_export_cmd(output, memory, session_id)` — Export current session to JSON file.
- `session_import_cmd(input_file, memory)` — Import a session from JSON file.
- `session_clear_cmd(memory, force)` — Clear all session data.
- `load_dotenv_if_available(path)` — Load .env file if it exists. No dependency on python-dotenv — just basic parsing.
- `get_env_config(dotenv_path)` — Read all config from environment variables (LiteLLM-compatible).
- `check_providers(env)` — Check which providers are configured and reachable.
- `resolve_alias(key)` — Resolve a user-friendly alias to an env var name.
- `mask_value(key, value)` — Mask secret values for display.
- `config_set(key, value, global_)` — Set a config value persistently in .env file.
- `config_get(key, global_)` — Get a config value. Checks: env var → project .env → global .env → None.
- `config_list(global_, show_secrets)` — List all config values from .env files and environment.
- `check_providers_live(env)` — Check providers with live connectivity tests.
- `get_current_trace()` — Get the active trace recorder for the current execution context.
- `set_current_trace(trace)` — Set the active trace recorder for the current execution context.
- `preprocess_and_execute(query, small_llm, large_llm, strategy)` — One function to preprocess and execute — like litellm.completion() but with small LLM decomposition.
- `preprocess_and_execute_sync(query, small_llm, large_llm, strategy)` — Synchronous version of preprocess_and_execute() — runs the async function in an event loop.
- `health()` — —
- `list_models()` — List available model pairs.
- `chat_completions(req)` — OpenAI-compatible chat completions with preLLM preprocessing.
- `batch_process(items)` — Process multiple queries in parallel.
- `create_app(small_model, large_model, strategy, config_path)` — Factory function to create a configured preLLM API server.
- `get_budget_tracker(monthly_limit, persist_path)` — Get or create the global budget tracker singleton.
- `reset_budget_tracker()` — Reset the global tracker (for testing).
- `ok(msg)` — —
- `warn(msg)` — —
- `fail(msg)` — —
- `info(msg)` — —
- `ask(question, default, required)` — Ask user for input with optional default.
- `ask_yn(question, default)` — Ask yes/no question.
- `ask_choice(question, options, default)` — Ask user to choose from options.
- `check_ollama(base_url)` — Check if Ollama is reachable and list available models.
- `fetch_ollama_models(base_url)` — Return list of installed Ollama models (raw names) or None if unreachable.
- `to_ollama_full(name)` — —
- `strip_ollama_prefix(name)` — —
- `build_ollama_options(installed_raw, recommended)` — —
- `option_index_for_value(options, value, default)` — —
- `install_ollama_model(raw)` — Attempt to install an Ollama model via CLI.
- `validate_ollama_model(base_url, model, installed_raw, reachable)` — —
- `check_api_key_format(provider, key)` — Validate API key format.
- `check_port_available(host, port)` — Check if port is available for the server.
- `mask_key(key)` — Mask API key for display.
- `main()` — —
- `setup_logging(level, markdown_file, terminal_format)` — Initialize nfo logging for the entire preLLM project.
- `get_logger(name)` — Get or create the nfo logger.


## Project Structure

📄 `examples.cli_examples`
📄 `examples.curl_api`
📄 `examples.embedded_refactor` (1 functions)
📄 `examples.k8s_debug` (1 functions)
📄 `examples.polish_leasing` (1 functions)
📄 `examples.providers` (17 functions)
📄 `examples.python_sdk` (11 functions)
📄 `examples.quick_start` (12 functions)
📦 `prellm`
📦 `prellm.agents`
📄 `prellm.agents.executor` (3 functions, 2 classes)
📄 `prellm.agents.preprocessor` (6 functions, 2 classes)
📦 `prellm.analyzers`
📄 `prellm.analyzers.context_engine` (13 functions, 1 classes)
📄 `prellm.budget` (11 functions, 3 classes)
📦 `prellm.chains`
📄 `prellm.chains.process_chain` (10 functions, 1 classes)
📄 `prellm.cli` (24 functions)
📦 `prellm.context`
📄 `prellm.context.codebase_indexer` (14 functions, 4 classes)
📄 `prellm.context.folder_compressor` (10 functions, 1 classes)
📄 `prellm.context.schema_generator` (9 functions, 1 classes)
📄 `prellm.context.sensitive_filter` (11 functions, 1 classes)
📄 `prellm.context.shell_collector` (8 functions, 1 classes)
📄 `prellm.context.user_memory` (15 functions, 1 classes)
📄 `prellm.core` (17 functions, 1 classes)
📄 `prellm.env_config` (12 functions, 1 classes)
📄 `prellm.llm_provider` (6 functions, 1 classes)
📄 `prellm.logging_setup` (3 functions)
📄 `prellm.model_catalog` (2 functions)
📄 `prellm.models` (2 functions, 33 classes)
📄 `prellm.pipeline` (18 functions, 5 classes)
📄 `prellm.prompt_registry` (11 functions, 5 classes)
📄 `prellm.query_decomposer` (10 functions, 1 classes)
📄 `prellm.server` (9 functions, 10 classes)
📄 `prellm.trace` (16 functions, 2 classes)
📄 `prellm.validators` (8 functions, 3 classes)
📄 `project`
📄 `scripts.config_wizard` (19 functions)

## Requirements



## Contributing

**Contributors:**
- Tom Softreck <tom@sapletta.com>
- Tom Sapletta <tom-sapletta-com@users.noreply.github.com>

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/wronai/gllm
cd prellm

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest
```

## Documentation

- 📖 [Full Documentation](https://github.com/wronai/gllm/tree/main/docs) — API reference, module docs, architecture
- 🚀 [Getting Started](https://github.com/wronai/gllm/blob/main/docs/getting-started.md) — Quick start guide
- 📚 [API Reference](https://github.com/wronai/gllm/blob/main/docs/api.md) — Complete API documentation
- 🔧 [Configuration](https://github.com/wronai/gllm/blob/main/docs/configuration.md) — Configuration options
- 💡 [Examples](./examples) — Usage examples and code samples

### Generated Files

| Output | Description | Link |
|--------|-------------|------|
| `README.md` | Project overview (this file) | — |
| `docs/api.md` | Consolidated API reference | [View](./docs/api.md) |
| `docs/modules.md` | Module reference with metrics | [View](./docs/modules.md) |
| `docs/architecture.md` | Architecture with diagrams | [View](./docs/architecture.md) |
| `docs/dependency-graph.md` | Dependency graphs | [View](./docs/dependency-graph.md) |
| `docs/coverage.md` | Docstring coverage report | [View](./docs/coverage.md) |
| `docs/getting-started.md` | Getting started guide | [View](./docs/getting-started.md) |
| `docs/configuration.md` | Configuration reference | [View](./docs/configuration.md) |
| `docs/api-changelog.md` | API change tracking | [View](./docs/api-changelog.md) |
| `CONTRIBUTING.md` | Contribution guidelines | [View](./CONTRIBUTING.md) |
| `examples/` | Usage examples | [Browse](./examples) |
| `mkdocs.yml` | MkDocs configuration | — |

<!-- code2docs:end -->