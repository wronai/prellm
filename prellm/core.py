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
    strategy: str | DecompositionStrategy = "classify",
    user_context: str | dict[str, str] | None = None,
    config_path: str | Path | None = None,
    domain_rules: list[dict[str, Any]] | None = None,
    pipeline: str | None = None,
    prompts_path: str | Path | None = None,
    pipelines_path: str | Path | None = None,
    schemas_path: str | Path | None = None,
    memory_path: str | Path | None = None,
    codebase_path: str | Path | None = None,
    **kwargs: Any,
) -> PreLLMResponse:
    """One function to preprocess and execute — like litellm.completion() but with small LLM decomposition.

    Always uses the v0.3 two-agent pipeline internally. The `strategy` parameter maps
    directly to a pipeline name (classify, structure, split, enrich, passthrough).
    The `pipeline` parameter overrides `strategy` for custom pipelines (e.g. "dual_agent_full").

    Args:
        query: The raw user query / prompt.
        small_llm: Model string for the small preprocessing LLM (e.g. "ollama/qwen2.5:3b").
        large_llm: Model string for the large executor LLM (e.g. "anthropic/claude-sonnet-4-20250514").
        strategy: Decomposition strategy — "classify", "structure", "split", "enrich", or "passthrough".
        user_context: Extra context as a string tag (e.g. "gdansk_embedded_python") or dict.
        config_path: Optional YAML config file for domain rules, prompts, etc.
        domain_rules: Optional inline domain rules as list of dicts.
        pipeline: Pipeline name from pipelines.yaml (e.g. "dual_agent_full"). Overrides strategy.
        prompts_path: Path to prompts.yaml.
        pipelines_path: Path to pipelines.yaml.
        schemas_path: Path to response_schemas.yaml.
        **kwargs: Extra kwargs passed to the large LLM call (max_tokens, temperature, etc.).

    Returns:
        PreLLMResponse with content, decomposition details, model info.

    Usage:
        # Strategy-based (maps to pipeline name)
        result = await preprocess_and_execute("Deploy app", strategy="structure")

        # Explicit pipeline name
        result = await preprocess_and_execute("Deploy app", pipeline="dual_agent_full")

    Zero-config:
        result = await preprocess_and_execute("Refaktoryzuj kod")
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
        memory_path=memory_path,
        codebase_path=codebase_path,
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
    **kwargs: Any,
) -> PreLLMResponse:
    """Two-agent execution path — PreprocessorAgent + ExecutorAgent + PromptPipeline."""
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

    # Build context
    extra_context: dict[str, Any] = {}
    if isinstance(user_context, str) and user_context:
        extra_context["user_context"] = user_context
    elif isinstance(user_context, dict):
        extra_context.update(user_context)

    # Inject domain rules into context for pipeline algo steps
    if domain_rules:
        extra_context["domain_rules"] = domain_rules

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
    )

    # 1. Preprocess
    trace = get_current_trace()
    _t0 = time.time()
    prep_result = await preprocessor.preprocess(
        query=query,
        user_context=extra_context or None,
        pipeline_name=pipeline,
    )
    _t1 = time.time()

    if trace:
        # Record individual pipeline steps from preprocessor
        if prep_result.decomposition and prep_result.decomposition.steps_executed:
            for ps in prep_result.decomposition.steps_executed:
                trace.step(
                    name=f"Pipeline: {ps.step_name}",
                    step_type="pipeline_step" if ps.step_type == "algo" else "llm_call",
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
            duration_ms=(_t1 - _t0) * 1000,
        )

    # 2. Execute
    _t2 = time.time()
    exec_result = await executor.execute(
        executor_input=prep_result.executor_input,
        **kwargs,
    )
    _t3 = time.time()

    if trace:
        content_preview = (exec_result.content or "")[:300]
        trace.step(
            name="ExecutorAgent.execute()",
            step_type="llm_call",
            description=f"Large LLM ({large_llm}) generated final response.",
            inputs={"executor_input": prep_result.executor_input[:300]},
            outputs={"content_preview": content_preview, "model": exec_result.model_used},
            duration_ms=(_t3 - _t2) * 1000,
            metadata={"retries": exec_result.retries},
        )

    # 3. Build backward-compatible DecompositionResult from pipeline state
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
