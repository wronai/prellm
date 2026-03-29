"""Main entry point helpers for preprocess_and_execute."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from prellm.models import DecompositionStrategy, PreLLMConfig
from prellm.trace import get_current_trace

logger = logging.getLogger("prellm")


def _resolve_pipeline_name(
    pipeline: str | None,
    strategy: str | DecompositionStrategy,
) -> str:
    """Resolve pipeline name from pipeline param or strategy."""
    if pipeline:
        return pipeline
    if isinstance(strategy, DecompositionStrategy):
        return strategy.value
    return strategy


def _load_config_overrides(
    config_path: str | Path | None,
    small_llm: str,
    large_llm: str,
    domain_rules: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], str, str, list[dict[str, Any]] | None]:
    """Load config from YAML and apply model overrides.
    
    Returns (config_overrides, small_llm, large_llm, domain_rules).
    """
    config_overrides: dict[str, Any] = {}
    
    if not config_path:
        return config_overrides, small_llm, large_llm, domain_rules
    
    # Import here to avoid circular dependency
    from prellm.core import PreLLM
    config = PreLLM._load_config(Path(config_path))
    
    # Use config models unless explicitly overridden (not default values)
    if small_llm == "ollama/qwen2.5:3b":
        small_llm = config.small_model.model
    if large_llm == "anthropic/claude-sonnet-4-20250514":
        large_llm = config.large_model.model
        config_overrides["max_tokens"] = config.large_model.max_tokens
    
    # Inject domain rules from config
    if config.domain_rules and not domain_rules:
        domain_rules = [r.model_dump() for r in config.domain_rules]
    
    return config_overrides, small_llm, large_llm, domain_rules


def _record_config_trace(
    trace: Any,
    small_llm: str,
    large_llm: str,
    pipeline_name: str,
    config_path: str | Path | None,
    user_context: str | dict[str, str] | None,
) -> None:
    """Record configuration step to trace if available."""
    if not trace:
        return
    
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
