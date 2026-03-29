"""Legacy PreLLM class (v0.2/v0.3) — backward compatibility wrapper.

This module contains the class-based API for backward compatibility with v0.1/v0.2 code.
New code should use the functional API: preprocess_and_execute().
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from prellm.agents.executor import ExecutorAgent
from prellm.agents.preprocessor import PreprocessorAgent
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
from prellm.pipeline import PromptPipeline
from prellm.prompt_registry import PromptRegistry
from prellm.query_decomposer import QueryDecomposer
from prellm.validators import ResponseValidator

logger = logging.getLogger("prellm")


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

    Deprecated: Use preprocess_and_execute() for new code.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: PreLLMConfig | None = None,
    ):
        import warnings
        warnings.warn(
            "PreLLM class is deprecated. Use preprocess_and_execute() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
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
