"""Core preLLM — main entry points for prompt decomposition, enrichment, and LLM calls.

v0.2 architecture:
    User Query
      → ContextEngine (env/git/system)
      → Small LLM ≤3B (classify → structure → compose)
      → Large LLM (GPT-4/Claude/Llama)
      → PreLLMResponse

The old `prellm` class is kept for backward compatibility with v0.1 code.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from prellm.analyzers.bias_detector import BiasDetector
from prellm.analyzers.context_engine import ContextEngine
from prellm.llm_provider import LLMProvider
from prellm.models import (
    AuditEntry,
    BiasPattern,
    DecompositionPrompts,
    DecompositionStrategy,
    DomainRule,
    GuardConfig,
    GuardResponse,
    LLMProviderConfig,
    ModelConfig,
    Policy,
    PreLLMConfig,
    PreLLMResponse,
)
from prellm.query_decomposer import QueryDecomposer

logger = logging.getLogger("prellm")


# ============================================================
# 1-function API — like litellm.completion() but with preprocessing
# ============================================================

async def preprocess_and_execute(
    query: str,
    small_llm: str = "ollama/qwen2.5:3b",
    large_llm: str = "anthropic/claude-sonnet-4-20250514",
    strategy: str | DecompositionStrategy = "classify",
    user_context: str | dict[str, str] | None = None,
    config_path: str | Path | None = None,
    domain_rules: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> PreLLMResponse:
    """One function to preprocess and execute — like litellm.completion() but with small LLM decomposition.

    Args:
        query: The raw user query / prompt.
        small_llm: Model string for the small preprocessing LLM (e.g. "ollama/qwen2.5:3b").
        large_llm: Model string for the large executor LLM (e.g. "anthropic/claude-sonnet-4-20250514").
        strategy: Decomposition strategy — "classify", "structure", "split", "enrich", or "passthrough".
        user_context: Extra context as a string tag (e.g. "gdansk_embedded_python") or dict.
        config_path: Optional YAML config file for domain rules, prompts, etc.
        domain_rules: Optional inline domain rules as list of dicts.
        **kwargs: Extra kwargs passed to the large LLM call (max_tokens, temperature, etc.).

    Returns:
        PreLLMResponse with content, decomposition details, model info.

    Usage:
        from prellm import preprocess_and_execute

        result = await preprocess_and_execute(
            query="Deploy app to production",
            small_llm="ollama/qwen2.5:3b",
            large_llm="gpt-4o-mini",
        )
        print(result.content)

    Zero-config:
        result = await preprocess_and_execute("Refaktoryzuj kod")
    """
    # Resolve strategy
    if isinstance(strategy, str):
        strat = DecompositionStrategy(strategy)
    else:
        strat = strategy

    # Build config — from file or inline
    if config_path:
        config = PreLLM._load_config(Path(config_path))
        # Override models if explicitly provided (non-default)
        if small_llm != "ollama/qwen2.5:3b":
            config.small_model = LLMProviderConfig(model=small_llm, max_tokens=512, temperature=0.0)
        if large_llm != "anthropic/claude-sonnet-4-20250514":
            config.large_model = LLMProviderConfig(model=large_llm, max_tokens=kwargs.pop("max_tokens", 2048))
    else:
        # Extract LLM-specific kwargs
        max_tokens = kwargs.pop("max_tokens", 2048)
        temperature = kwargs.pop("temperature", 0.7)

        rules = []
        if domain_rules:
            for r in domain_rules:
                rules.append(DomainRule(**r) if isinstance(r, dict) else r)

        config = PreLLMConfig(
            small_model=LLMProviderConfig(model=small_llm, max_tokens=512, temperature=0.0),
            large_model=LLMProviderConfig(model=large_llm, max_tokens=max_tokens, temperature=temperature),
            default_strategy=strat,
            domain_rules=rules,
        )

    # Build context
    extra_context: dict[str, str] | None = None
    if isinstance(user_context, str) and user_context:
        extra_context = {"user_context": user_context}
    elif isinstance(user_context, dict):
        extra_context = user_context

    # Execute
    engine = PreLLM(config=config)
    return await engine(query, strategy=strat, extra_context=extra_context, **kwargs)


# Sync wrapper for non-async code
def preprocess_and_execute_sync(
    query: str,
    small_llm: str = "ollama/qwen2.5:3b",
    large_llm: str = "anthropic/claude-sonnet-4-20250514",
    strategy: str | DecompositionStrategy = "classify",
    user_context: str | dict[str, str] | None = None,
    config_path: str | Path | None = None,
    domain_rules: list[dict[str, Any]] | None = None,
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
        **kwargs,
    ))


# ============================================================
# v0.2 — PreLLM (new architecture)
# ============================================================

class PreLLM:
    """preLLM v0.2 — small LLM decomposition before large LLM routing.

    Usage:
        engine = PreLLM("prellm_config.yaml")
        result = await engine("Zdeployuj apkę na prod")

    Or with inline config:
        engine = PreLLM(config=PreLLMConfig(...))
        result = await engine("Deploy the app", strategy=DecompositionStrategy.STRUCTURE)
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


# ============================================================
# v0.1 compat — prellm (old class, kept for backward compatibility)
# ============================================================

class prellm:
    """v0.1 Prellm middleware — analyze, enrich, and proxy LLM calls.

    DEPRECATED: Use PreLLM for v0.2 small-LLM decomposition pipeline.

    Usage:
        guard = prellm("rules.yaml")
        result = await guard("Zdeployuj na produkcję", model="gpt-4o-mini")
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: GuardConfig | None = None,
    ):
        if config:
            self.config = config
        elif config_path:
            self.config = self._load_config(Path(config_path))
        else:
            self.config = GuardConfig()

        self.detector = BiasDetector(self.config.bias_patterns or None)
        self.context_engine = ContextEngine(self.config.context_sources)
        self.audit_log: list[AuditEntry] = []

    async def __call__(
        self,
        query: str,
        model: str | None = None,
        extra_context: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> GuardResponse:
        """Analyze query, enrich if needed, call LLM, validate response."""
        import litellm

        target_model = model or self.config.models.fallback[0]

        # Step 1: Analyze
        analysis = self.detector.analyze(query)
        analysis.original_query = query

        # Step 2: Enrich if needed
        if analysis.needs_clarify:
            enriched = self.config.clarify_template.format(query=query)
            enriched = self.context_engine.enrich_prompt(enriched, extra_context)
            analysis.enriched_query = enriched
        else:
            enriched = self.context_engine.enrich_prompt(query, extra_context)
            analysis.enriched_query = enriched

        # Step 3: Call LLM with retry/fallback
        response_content = ""
        model_used = target_model
        retries = 0

        fallback_models = [target_model] + [
            m for m in self.config.models.fallback if m != target_model
        ]

        for attempt_model in fallback_models:
            for attempt in range(self.config.max_retries):
                try:
                    resp = await litellm.acompletion(
                        model=attempt_model,
                        messages=[{"role": "user", "content": enriched}],
                        max_tokens=self.config.models.max_tokens,
                        timeout=self.config.models.timeout,
                        **kwargs,
                    )
                    response_content = resp.choices[0].message.content
                    model_used = attempt_model
                    break
                except Exception as e:
                    retries += 1
                    logger.warning(f"Attempt {attempt + 1} with {attempt_model} failed: {e}")
            if response_content:
                break

        # Step 4: Build response
        result = GuardResponse(
            content=response_content or "No response from any model.",
            clarified=analysis.needs_clarify,
            needs_more_context=analysis.needs_clarify and not response_content,
            model_used=model_used,
            analysis=analysis,
            retries=retries,
        )

        # Audit
        self._audit("query", query, result, model_used)

        return result

    def analyze_only(self, query: str) -> dict[str, Any]:
        """Run analysis without calling LLM — useful for dry-run / testing."""
        analysis = self.detector.analyze(query)
        return {
            "needs_clarify": analysis.needs_clarify,
            "patterns": analysis.detected_patterns,
            "ambiguity_flags": analysis.ambiguity_flags,
            "readability": analysis.readability_score,
            "enriched": self.config.clarify_template.format(query=query)
            if analysis.needs_clarify
            else query,
        }

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return audit log as list of dicts."""
        return [entry.model_dump() for entry in self.audit_log]

    def _audit(self, action: str, query: str, response: GuardResponse, model: str) -> None:
        entry = AuditEntry(
            action=action,
            query=query,
            response_summary=response.content[:200] if response.content else "",
            model=model,
            policy=self.config.policy,
        )
        self.audit_log.append(entry)

    @staticmethod
    def _load_config(path: Path) -> GuardConfig:
        """Load config from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        # Normalize bias_patterns
        patterns = []
        for p in raw.get("bias_patterns", []):
            if isinstance(p, dict):
                patterns.append(BiasPattern(**p))

        models_raw = raw.get("models", {})
        models = ModelConfig(**models_raw) if isinstance(models_raw, dict) else ModelConfig()

        return GuardConfig(
            bias_patterns=patterns,
            clarify_template=raw.get("clarify_template", GuardConfig.model_fields["clarify_template"].default),
            max_retries=raw.get("max_retries", 3),
            policy=Policy(raw.get("policy", "strict")),
            models=models,
            context_sources=raw.get("context_sources", []),
        )
