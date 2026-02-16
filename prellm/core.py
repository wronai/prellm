"""Core preLLM — main entry points for prompt decomposition, enrichment, and LLM calls.

v0.3 architecture (two-agent):
    User Query
      → PreprocessorAgent (small LLM ≤24B, PromptPipeline)
      → ExecutorAgent (large LLM >24B)
      → PreLLMResponse

v0.2 architecture (backward compat):
    User Query
      → ContextEngine (env/git/system)
      → Small LLM ≤3B (classify → structure → compose)
      → Large LLM (GPT-4/Claude/Llama)
      → PreLLMResponse

The old `prellm` class is kept for backward compatibility with v0.1 code.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from nfo.decorators import catch, log_call

from prellm.analyzers.context_engine import ContextEngine
from prellm.llm_provider import LLMProvider
from prellm.models import (
    AuditEntry,
    DecompositionPrompts,
    DecompositionResult,
    DecompositionStrategy,
    DomainRule,
    LLMProviderConfig,
    Policy,
    PreLLMConfig,
    PreLLMResponse,
)
from prellm.query_decomposer import QueryDecomposer
from prellm.agents.executor import ExecutorAgent
from prellm.agents.preprocessor import PreprocessorAgent
from prellm.pipeline import PromptPipeline
from prellm.prompt_registry import PromptRegistry
from prellm.validators import ResponseValidator
from prellm.trace import get_current_trace

logger = logging.getLogger("prellm")


# ============================================================
# 1-function API — like litellm.completion() but with preprocessing
# ============================================================

@catch
async def preprocess_and_execute(
    query: str,
    small_llm: str = "ollama/qwen2.5:3b",
    large_llm: str = "anthropic/claude-sonnet-4-20250514",
    strategy: str | DecompositionStrategy = "auto",
    user_context: str | dict[str, str] | None = None,
    config_path: str | Path | None = None,
    domain_rules: list[dict[str, Any]] | None = None,
    pipeline: str | None = None,
    prompts_path: str | Path | None = None,
    pipelines_path: str | Path | None = None,
    schemas_path: str | Path | None = None,
    memory_path: str | Path | None = None,
    codebase_path: str | Path | None = None,
    collect_env: bool = False,
    collect_runtime: bool = True,
    session_path: str | Path | None = None,
    compress_folder: bool = False,
    sanitize: bool = True,
    sensitive_rules: str | Path | None = None,
    **kwargs: Any,
) -> PreLLMResponse:
    """One function to preprocess and execute — like litellm.completion() but with small LLM decomposition.

    v0.4: automatic persistent context layer for small LLMs. Collects env, compresses codebase,
    persists session, and injects everything into the small-LLM without manual pre-prompts.
    Sensitive data never reaches the large-LLM.

    Args:
        query: The raw user query / prompt.
        small_llm: Model string for the small preprocessing LLM (e.g. "ollama/qwen2.5:3b").
        large_llm: Model string for the large executor LLM (e.g. "anthropic/claude-sonnet-4-20250514").
        strategy: Decomposition strategy — "auto" (default), "classify", "structure", "split", "enrich", "passthrough".
        user_context: Extra context as a string tag (e.g. "gdansk_embedded_python") or dict.
        config_path: Optional YAML config file for domain rules, prompts, etc.
        domain_rules: Optional inline domain rules as list of dicts.
        pipeline: Pipeline name from pipelines.yaml (e.g. "dual_agent_full"). Overrides strategy.
        prompts_path: Path to prompts.yaml.
        pipelines_path: Path to pipelines.yaml.
        schemas_path: Path to response_schemas.yaml.
        memory_path: Path to UserMemory database.
        codebase_path: Folder to compress for context injection.
        collect_env: Collect env vars (legacy, use collect_runtime instead).
        collect_runtime: Collect full runtime context (env, process, locale, network, git, system).
        session_path: Path to session persistence SQLite DB.
        compress_folder: Compress codebase folder into .toon representation.
        sanitize: Filter sensitive data before large-LLM (default: True).
        sensitive_rules: Custom YAML rules for sensitive data classification.
        **kwargs: Extra kwargs passed to the large LLM call (max_tokens, temperature, etc.).

    Returns:
        PreLLMResponse with content, decomposition details, model info.

    Usage:
        # Zero-config with auto strategy:
        result = await preprocess_and_execute("Refaktoryzuj kod")

        # Bielik with full persistent context:
        result = await preprocess_and_execute(
            "Zoptymalizuj monitoring ESP32",
            small_llm="ollama/bielik:7b",
            large_llm="openrouter/google/gemini-3-flash-preview",
            session_path=".prellm/sessions.db",
            codebase_path=".",
        )
    """
    # Resolve pipeline name: pipeline param overrides strategy
    pipeline_name = pipeline or (strategy.value if isinstance(strategy, DecompositionStrategy) else strategy)

    # Load config overrides from YAML if provided
    config_overrides: dict[str, Any] = {}
    if config_path:
        config = PreLLM._load_config(Path(config_path))
        # Use config models unless explicitly overridden
        if small_llm == "ollama/qwen2.5:3b":
            small_llm = config.small_model.model
        if large_llm == "anthropic/claude-sonnet-4-20250514":
            large_llm = config.large_model.model
            kwargs.setdefault("max_tokens", config.large_model.max_tokens)
        # Inject domain rules from config
        if config.domain_rules and not domain_rules:
            domain_rules = [r.model_dump() for r in config.domain_rules]

    logger.info(f"preLLM pipeline: {small_llm} \u2192 {large_llm} | strategy={pipeline_name}")

    # Record trace config
    trace = get_current_trace()
    if trace:
        trace.step(
            name="Configuration",
            step_type="config",
            description="Resolved models, strategy, and pipeline parameters.",
            outputs={
                "small_llm": small_llm,
                "large_llm": large_llm,
                "strategy": pipeline_name,
                "config_path": str(config_path) if config_path else None,
                "user_context": user_context,
            },
        )

    return await _execute_v3_pipeline(
        query=query,
        small_llm=small_llm,
        large_llm=large_llm,
        pipeline=pipeline_name,
        user_context=user_context,
        domain_rules=domain_rules,
        prompts_path=prompts_path,
        pipelines_path=pipelines_path,
        schemas_path=schemas_path,
        memory_path=memory_path or (str(session_path) if session_path else None),
        codebase_path=codebase_path,
        collect_env=collect_env or collect_runtime,
        compress_folder=compress_folder or bool(codebase_path),
        sanitize=sanitize,
        sensitive_rules=sensitive_rules,
        **kwargs,
    )


# Sync wrapper for non-async code
def preprocess_and_execute_sync(
    query: str,
    small_llm: str = "ollama/qwen2.5:3b",
    large_llm: str = "anthropic/claude-sonnet-4-20250514",
    strategy: str | DecompositionStrategy = "classify",
    user_context: str | dict[str, str] | None = None,
    config_path: str | Path | None = None,
    domain_rules: list[dict[str, Any]] | None = None,
    pipeline: str | None = None,
    prompts_path: str | Path | None = None,
    pipelines_path: str | Path | None = None,
    schemas_path: str | Path | None = None,
    **kwargs: Any,
) -> PreLLMResponse:
    """Synchronous version of preprocess_and_execute() — runs the async function in an event loop.

    Usage:
        from prellm import preprocess_and_execute_sync
        result = preprocess_and_execute_sync("Deploy app", large_llm="gpt-4o-mini")
    """
    import asyncio
    return asyncio.run(preprocess_and_execute(
        query=query,
        small_llm=small_llm,
        large_llm=large_llm,
        strategy=strategy,
        user_context=user_context,
        config_path=config_path,
        domain_rules=domain_rules,
        pipeline=pipeline,
        prompts_path=prompts_path,
        pipelines_path=pipelines_path,
        schemas_path=schemas_path,
        **kwargs,
    ))


# ============================================================
# v0.3 — Two-agent execution (internal, called by preprocess_and_execute)
# ============================================================

@log_call
async def _execute_v3_pipeline(
    query: str,
    small_llm: str,
    large_llm: str,
    pipeline: str,
    user_context: str | dict[str, str] | None = None,
    domain_rules: list[dict[str, Any]] | None = None,
    prompts_path: str | Path | None = None,
    pipelines_path: str | Path | None = None,
    schemas_path: str | Path | None = None,
    memory_path: str | Path | None = None,
    codebase_path: str | Path | None = None,
    collect_env: bool = False,
    compress_folder: bool = False,
    sanitize: bool = False,
    sensitive_rules: str | Path | None = None,
    **kwargs: Any,
) -> PreLLMResponse:
    """Two-agent execution path — PreprocessorAgent + ExecutorAgent + PromptPipeline.

    v0.4 refactor: split into _prepare_context, _run_preprocessing, _run_execution_with_sanitize,
    _persist_session to reduce cyclomatic complexity.
    """
    max_tokens = kwargs.pop("max_tokens", 2048)
    temperature = kwargs.pop("temperature", 0.7)

    # Build LLM providers
    small_llm_config = LLMProviderConfig(model=small_llm, max_tokens=512, temperature=0.0)
    large_llm_config = LLMProviderConfig(model=large_llm, max_tokens=max_tokens, temperature=temperature)

    small_provider = LLMProvider(small_llm_config)
    large_provider = LLMProvider(large_llm_config)

    # Build registry and pipeline
    registry = PromptRegistry(prompts_path=prompts_path)
    prompt_pipeline = PromptPipeline.from_yaml(
        pipelines_path=pipelines_path,
        pipeline_name=pipeline,
        registry=registry,
        small_llm=small_provider,
    )

    # 1. Prepare context (env, codebase, memory, sensitive filter)
    extra_context, sensitive_filter, user_memory, codebase_indexer = _prepare_context(
        user_context=user_context,
        domain_rules=domain_rules,
        collect_env=collect_env,
        compress_folder=compress_folder,
        codebase_path=codebase_path,
        sanitize=sanitize,
        sensitive_rules=sensitive_rules,
        memory_path=memory_path,
    )

    # Build agents
    preprocessor = PreprocessorAgent(
        small_llm=small_provider,
        registry=registry,
        pipeline=prompt_pipeline,
        context_engine=ContextEngine(),
        user_memory=user_memory,
        codebase_indexer=codebase_indexer,
        codebase_path=str(codebase_path) if codebase_path else None,
    )
    executor = ExecutorAgent(
        large_llm=large_provider,
        response_validator=ResponseValidator(schemas_path=schemas_path) if schemas_path else None,
        sensitive_filter=sensitive_filter,
    )

    # 2. Build compact pipeline context for small LLM (no raw blobs)
    pipeline_context = _build_pipeline_context(extra_context)

    # 2b. Run preprocessing (small LLM) with compact context only
    prep_result, prep_duration_ms = await _run_preprocessing(
        preprocessor, query, pipeline_context, pipeline
    )

    # 3. Build system_prompt from preprocessing + FULL context for the large LLM
    system_prompt = _build_executor_system_prompt(prep_result, extra_context)

    # 4. Run execution with sanitization (large LLM)
    exec_result, exec_duration_ms = await _run_execution(
        executor, prep_result.executor_input, system_prompt=system_prompt, **kwargs
    )

    # 5. Persist session if memory available
    await _persist_session(user_memory, query, exec_result)

    # 6. Build response
    trace = get_current_trace()
    _record_trace(trace, pipeline, small_llm, large_llm, query, pipeline_context,
                  prep_result, exec_result, prep_duration_ms, exec_duration_ms)

    decomposition_result = _build_decomposition_result(query, pipeline, prep_result)

    response = PreLLMResponse(
        content=exec_result.content or "No response from any model.",
        decomposition=decomposition_result,
        model_used=exec_result.model_used,
        small_model_used=small_llm,
        retries=exec_result.retries,
        clarified=bool(decomposition_result and decomposition_result.missing_fields),
        needs_more_context=bool(decomposition_result and decomposition_result.missing_fields) and not exec_result.content,
    )

    if trace:
        trace.set_result(
            content=response.content,
            model_used=response.model_used,
            small_model_used=response.small_model_used,
            retries=response.retries,
            strategy=pipeline,
            classification=(
                decomposition_result.classification.model_dump()
                if decomposition_result and decomposition_result.classification
                else None
            ),
        )

    return response


def _prepare_context(
    user_context: str | dict[str, str] | None,
    domain_rules: list[dict[str, Any]] | None,
    collect_env: bool,
    compress_folder: bool,
    codebase_path: str | Path | None,
    sanitize: bool,
    sensitive_rules: str | Path | None,
    memory_path: str | Path | None,
) -> tuple[dict[str, Any], Any, Any, Any]:
    """Gather all context: env, codebase, schema, sensitive filter, memory, indexer.

    Returns (extra_context, sensitive_filter, user_memory, codebase_indexer).
    """
    extra_context: dict[str, Any] = {}
    if isinstance(user_context, str) and user_context:
        extra_context["user_context"] = user_context
    elif isinstance(user_context, dict):
        extra_context.update(user_context)

    if domain_rules:
        extra_context["domain_rules"] = domain_rules

    # Collect shell context + runtime context
    shell_ctx = None
    if collect_env:
        try:
            from prellm.context.shell_collector import ShellContextCollector
            shell_ctx = ShellContextCollector().collect_all()
            extra_context["shell_context"] = shell_ctx.model_dump_json()
        except Exception as e:
            logger.warning(f"Shell context collection failed: {e}")

        # Also collect RuntimeContext
        try:
            runtime_ctx = ContextEngine().gather_runtime()
            extra_context["runtime_context"] = runtime_ctx.model_dump()
        except Exception as e:
            logger.warning(f"RuntimeContext collection failed: {e}")

    # Compress folder
    compressed = None
    if compress_folder and codebase_path:
        try:
            from prellm.context.folder_compressor import FolderCompressor
            compressed = FolderCompressor().compress(codebase_path)
            extra_context["folder_compressed"] = compressed.toon_output
        except Exception as e:
            logger.warning(f"Folder compression failed: {e}")

    # Generate context schema
    if collect_env or compress_folder:
        try:
            from prellm.context.schema_generator import ContextSchemaGenerator
            env_schema = ContextSchemaGenerator().generate(
                shell_context=shell_ctx,
                folder_compressed=compressed,
            )
            extra_context["context_schema"] = env_schema.model_dump_json()
        except Exception as e:
            logger.warning(f"Context schema generation failed: {e}")

    # Build sensitive filter
    sensitive_filter = None
    if sanitize:
        try:
            from prellm.context.sensitive_filter import SensitiveDataFilter
            sensitive_filter = SensitiveDataFilter(
                rules_path=sensitive_rules if sensitive_rules else None,
            )
            filtered = sensitive_filter.filter_context_for_large_llm(extra_context)
            extra_context = filtered
        except Exception as e:
            logger.warning(f"Sensitive filtering failed: {e}")

    # Build optional context enrichment
    user_memory = None
    if memory_path:
        try:
            from prellm.context.user_memory import UserMemory
            user_memory = UserMemory(path=memory_path)
        except Exception as e:
            logger.warning(f"Failed to initialize UserMemory: {e}")

    codebase_indexer = None
    if codebase_path:
        try:
            from prellm.context.codebase_indexer import CodebaseIndexer
            codebase_indexer = CodebaseIndexer()
        except Exception as e:
            logger.warning(f"Failed to initialize CodebaseIndexer: {e}")

    return extra_context, sensitive_filter, user_memory, codebase_indexer


def _build_pipeline_context(extra_context: dict[str, Any]) -> dict[str, Any]:
    """Build compact context for small LLM pipeline — strips raw blobs.

    The small LLM only needs:
    - context_schema (compact summary of environment)
    - domain_rules (if any)
    - user_context (if any)
    - folder_compressed (toon format, if any)

    Raw shell_context and runtime_context are kept only in extra_context
    for _build_executor_system_prompt (large LLM).
    """
    # Keys that are too large / not useful for the small LLM pipeline
    _LARGE_BLOB_KEYS = {"shell_context", "runtime_context"}

    return {k: v for k, v in extra_context.items() if k not in _LARGE_BLOB_KEYS}


async def _run_preprocessing(
    preprocessor: PreprocessorAgent,
    query: str,
    extra_context: dict[str, Any],
    pipeline: str,
) -> tuple[Any, float]:
    """Run the small-LLM preprocessing step. Returns (prep_result, duration_ms)."""
    _t0 = time.time()
    prep_result = await preprocessor.preprocess(
        query=query,
        user_context=extra_context or None,
        pipeline_name=pipeline,
    )
    duration_ms = (time.time() - _t0) * 1000
    return prep_result, duration_ms


def _build_executor_system_prompt(
    prep_result: Any,
    extra_context: dict[str, Any],
) -> str:
    """Build a system prompt for the large LLM from preprocessing results and context.

    Injects classification, context schema, and runtime info so the large LLM
    understands the user's intent and environment.
    """
    parts: list[str] = []

    # 1. Classification context
    if prep_result.decomposition:
        state = prep_result.decomposition.state
        classification = state.get("classification")
        if isinstance(classification, dict):
            intent = classification.get("intent", "unknown")
            confidence = classification.get("confidence", 0)
            domain = classification.get("domain", "general")
            parts.append(
                f"User intent: {intent} (confidence: {confidence}, domain: {domain})"
            )

        matched_rule = state.get("matched_rule")
        if isinstance(matched_rule, dict) and matched_rule.get("name"):
            parts.append(f"Matched domain rule: {matched_rule['name']}")
            if matched_rule.get("required_fields"):
                parts.append(f"Required fields: {', '.join(matched_rule['required_fields'])}")

    # 2. Available tools from context schema
    ctx_schema = extra_context.get("context_schema")
    if ctx_schema:
        try:
            import json
            schema_data = json.loads(ctx_schema) if isinstance(ctx_schema, str) else ctx_schema
            tools = schema_data.get("available_tools", [])
            if tools:
                parts.append(f"Available tools on user's system: {', '.join(tools[:15])}")
            platform = schema_data.get("platform")
            if platform:
                parts.append(f"Platform: {platform}")
            locale = schema_data.get("locale")
            if locale:
                parts.append(f"Locale: {locale}")
        except Exception:
            pass

    # 3. Runtime context summary
    runtime = extra_context.get("runtime_context")
    if isinstance(runtime, dict):
        sys_info = runtime.get("system", {})
        proc_info = runtime.get("process", {})
        if sys_info.get("os"):
            parts.append(f"OS: {sys_info['os']} {sys_info.get('arch', '')}")
        if sys_info.get("python"):
            parts.append(f"Python: {sys_info['python']}")
        if proc_info.get("cwd"):
            parts.append(f"Working directory: {proc_info['cwd']}")

    # 4. User preferences / history
    user_ctx = extra_context.get("user_context")
    if user_ctx:
        parts.append(f"User context: {user_ctx}")

    if not parts:
        return ""

    return "Context from preprocessing:\n" + "\n".join(f"- {p}" for p in parts)


async def _run_execution(
    executor: ExecutorAgent,
    executor_input: str,
    system_prompt: str = "",
    **kwargs: Any,
) -> tuple[Any, float]:
    """Run the large-LLM execution step. Returns (exec_result, duration_ms)."""
    _t0 = time.time()
    exec_result = await executor.execute(
        executor_input=executor_input,
        system_prompt=system_prompt,
        **kwargs,
    )
    duration_ms = (time.time() - _t0) * 1000
    return exec_result, duration_ms


async def _persist_session(
    user_memory: Any,
    query: str,
    exec_result: Any,
) -> None:
    """Persist interaction to UserMemory if available."""
    if not user_memory:
        return
    try:
        content = exec_result.content or ""
        await user_memory.add_interaction(
            query=query,
            response_summary=content[:500],
            metadata={"model": exec_result.model_used},
        )
        # Auto-learn preferences from interaction
        await user_memory.learn_preference_from_interaction(query, content)
    except Exception as e:
        logger.warning(f"Session persistence failed: {e}")


def _record_trace(
    trace: Any,
    pipeline: str,
    small_llm: str,
    large_llm: str,
    query: str,
    extra_context: dict[str, Any],
    prep_result: Any,
    exec_result: Any,
    prep_duration_ms: float,
    exec_duration_ms: float,
) -> None:
    """Record preprocessing and execution steps to trace."""
    if not trace:
        return

    # Record individual pipeline steps from preprocessor
    if prep_result.decomposition and prep_result.decomposition.steps_executed:
        for ps in prep_result.decomposition.steps_executed:
            step_type = "context_collection" if ps.step_type == "algo" and ps.step_name in (
                "collect_runtime", "inject_session", "sanitize"
            ) else ("pipeline_step" if ps.step_type == "algo" else "llm_call")
            trace.step(
                name=f"Pipeline: {ps.step_name}",
                step_type=step_type,
                description=f"{ps.step_type} step in '{pipeline}' pipeline",
                outputs={ps.output_key: ps.output_value} if ps.output_key else {},
                status="skipped" if ps.skipped else ("error" if ps.error else "ok"),
                error=ps.error,
            )
    trace.step(
        name="PreprocessorAgent.preprocess()",
        step_type="agent",
        description=f"Small LLM ({small_llm}) preprocessed query using '{pipeline}' strategy.",
        inputs={"query": query, "pipeline": pipeline, "user_context": extra_context},
        outputs={"executor_input": prep_result.executor_input},
        duration_ms=prep_duration_ms,
    )

    trace.step(
        name="ExecutorAgent.execute()",
        step_type="llm_call",
        description=f"Large LLM ({large_llm}) generated final response.",
        inputs={"executor_input": prep_result.executor_input},
        outputs={"content_preview": exec_result.content or "", "model": exec_result.model_used},
        duration_ms=exec_duration_ms,
        metadata={"retries": exec_result.retries},
    )


def _build_decomposition_result(
    query: str,
    pipeline_name: str,
    prep_result: Any,
) -> DecompositionResult | None:
    """Build a backward-compatible DecompositionResult from pipeline state."""
    from prellm.models import ClassificationResult, StructureResult

    if not prep_result.decomposition:
        return None

    state = prep_result.decomposition.state
    strategy_values = [s.value for s in DecompositionStrategy]
    strategy = DecompositionStrategy(pipeline_name) if pipeline_name in strategy_values else DecompositionStrategy.CLASSIFY

    result = DecompositionResult(
        strategy=strategy,
        original_query=query,
        composed_prompt=prep_result.executor_input,
    )

    # Extract classification from pipeline state
    classification = state.get("classification")
    if isinstance(classification, dict):
        result.classification = ClassificationResult(
            intent=classification.get("intent", "unknown"),
            confidence=float(classification.get("confidence", 0.0)),
            domain=classification.get("domain", "general"),
        )

    # Extract structure from pipeline state
    fields = state.get("fields")
    if isinstance(fields, dict):
        result.structure = StructureResult(
            action=fields.get("action", ""),
            target=fields.get("target", ""),
            parameters=fields.get("parameters", {}),
        )

    # Extract sub_queries from pipeline state
    sub_queries = state.get("sub_queries")
    if isinstance(sub_queries, dict) and "sub_queries" in sub_queries:
        result.sub_queries = [str(q) for q in sub_queries["sub_queries"]]
    elif isinstance(sub_queries, list):
        result.sub_queries = [str(q) for q in sub_queries]

    # Extract missing_fields from pipeline state
    missing_fields = state.get("missing_fields")
    if isinstance(missing_fields, list):
        result.missing_fields = missing_fields

    # Extract matched_rule from pipeline state
    matched_rule = state.get("matched_rule")
    if isinstance(matched_rule, dict) and "name" in matched_rule:
        result.matched_rule = matched_rule["name"]
        # Also extract missing fields from rule matching
        if not result.missing_fields and matched_rule.get("required_fields"):
            result.missing_fields = matched_rule["required_fields"]

    return result


# Backward-compatible alias
preprocess_and_execute_v3 = preprocess_and_execute


# ============================================================
# v0.2 — PreLLM (class-based architecture, backward compat)
# ============================================================

class PreLLM:
    """preLLM v0.2/v0.3 — small LLM decomposition before large LLM routing.

    Usage:
        engine = PreLLM("prellm_config.yaml")
        result = await engine("Zdeployuj apkę na prod")

    Or with inline config:
        engine = PreLLM(config=PreLLMConfig(...))
        result = await engine("Deploy the app", strategy=DecompositionStrategy.STRUCTURE)

    v0.3 two-agent mode:
        engine = PreLLM(config=config, use_agents=True)
        result = await engine("Deploy the app", pipeline="dual_agent_full")
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: PreLLMConfig | None = None,
    ):
        if config:
            self.config = config
        elif config_path:
            self.config = self._load_config(Path(config_path))
        else:
            self.config = PreLLMConfig()

        self.small_llm = LLMProvider(self.config.small_model)
        self.large_llm = LLMProvider(self.config.large_model)
        self.decomposer = QueryDecomposer(
            small_llm=self.small_llm,
            prompts=self.config.prompts,
            domain_rules=self.config.domain_rules,
        )
        self.context_engine = ContextEngine(self.config.context_sources)
        self.audit_log: list[AuditEntry] = []

    async def __call__(
        self,
        query: str,
        strategy: DecompositionStrategy | None = None,
        extra_context: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> PreLLMResponse:
        """Full pipeline: context → decompose (small LLM) → large LLM → response.

        Args:
            query: The raw user query.
            strategy: Override the default decomposition strategy.
            extra_context: Additional key-value context to inject.
            **kwargs: Extra kwargs passed to the large LLM call.

        Returns:
            PreLLMResponse with decomposition details and large LLM output.
        """
        active_strategy = strategy or self.config.default_strategy

        # Step 1: Gather context
        ctx = self.context_engine.gather()
        if extra_context:
            ctx.update(extra_context)

        # Step 2: Decompose with small LLM
        decomposition = await self.decomposer.decompose(
            query=query,
            strategy=active_strategy,
            context=ctx,
        )

        # Step 3: Call large LLM with composed prompt
        prompt_for_large = decomposition.composed_prompt or query
        retries = 0

        try:
            content = await self.large_llm.complete(
                user_message=prompt_for_large,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Large LLM call failed: {e}")
            content = "No response from any model."
            retries += 1

        # Step 4: Build response
        result = PreLLMResponse(
            content=content,
            decomposition=decomposition,
            model_used=self.config.large_model.model,
            small_model_used=self.config.small_model.model,
            retries=retries,
            clarified=bool(decomposition.missing_fields),
            needs_more_context=bool(decomposition.missing_fields) and content == "No response from any model.",
        )

        # Audit
        self._audit("query", query, result)

        return result

    async def decompose_only(
        self,
        query: str,
        strategy: DecompositionStrategy | None = None,
        extra_context: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run decomposition without calling the large LLM — useful for dry-run / testing."""
        active_strategy = strategy or self.config.default_strategy

        ctx = self.context_engine.gather()
        if extra_context:
            ctx.update(extra_context)

        decomposition = await self.decomposer.decompose(
            query=query,
            strategy=active_strategy,
            context=ctx,
        )

        return {
            "strategy": decomposition.strategy.value,
            "original_query": decomposition.original_query,
            "classification": decomposition.classification.model_dump() if decomposition.classification else None,
            "structure": decomposition.structure.model_dump() if decomposition.structure else None,
            "sub_queries": decomposition.sub_queries,
            "missing_fields": decomposition.missing_fields,
            "matched_rule": decomposition.matched_rule,
            "composed_prompt": decomposition.composed_prompt,
        }

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return audit log as list of dicts."""
        return [entry.model_dump() for entry in self.audit_log]

    def _audit(self, action: str, query: str, response: PreLLMResponse) -> None:
        entry = AuditEntry(
            action=action,
            query=query,
            response_summary=response.content[:200] if response.content else "",
            model=response.model_used,
            policy=self.config.policy,
            metadata={
                "small_model": response.small_model_used,
                "strategy": response.decomposition.strategy.value if response.decomposition else "unknown",
            },
        )
        self.audit_log.append(entry)

    @staticmethod
    def _load_config(path: Path) -> PreLLMConfig:
        """Load preLLM v0.2 config from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        # Parse small_model
        small_raw = raw.get("small_model", {})
        small_model = LLMProviderConfig(**small_raw) if isinstance(small_raw, dict) and small_raw else LLMProviderConfig(
            model="phi3:mini", max_tokens=512, temperature=0.0
        )

        # Parse large_model
        large_raw = raw.get("large_model", {})
        large_model = LLMProviderConfig(**large_raw) if isinstance(large_raw, dict) and large_raw else LLMProviderConfig(
            model="gpt-4o-mini", max_tokens=2048
        )

        # Parse domain_rules
        domain_rules = []
        for r in raw.get("domain_rules", []):
            if isinstance(r, dict):
                domain_rules.append(DomainRule(**r))

        # Parse prompts
        prompts_raw = raw.get("prompts", {})
        prompts = DecompositionPrompts(**prompts_raw) if isinstance(prompts_raw, dict) and prompts_raw else DecompositionPrompts()

        # Parse default_strategy
        strategy_str = raw.get("default_strategy", "classify")
        default_strategy = DecompositionStrategy(strategy_str)

        return PreLLMConfig(
            small_model=small_model,
            large_model=large_model,
            domain_rules=domain_rules,
            prompts=prompts,
            default_strategy=default_strategy,
            context_sources=raw.get("context_sources", []),
            max_retries=raw.get("max_retries", 3),
            policy=Policy(raw.get("policy", "strict")),
        )
