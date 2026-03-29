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
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from nfo.decorators import catch

from prellm.core.legacy import PreLLM
from prellm.core.main import (
    _resolve_pipeline_name,
    _load_config_overrides,
    _record_config_trace,
)
from prellm.core.pipeline import _execute_v3_pipeline
from prellm.models import DecompositionStrategy
from prellm.trace import get_current_trace

__all__ = [
    "preprocess_and_execute",
    "preprocess_and_execute_sync",
    "preprocess_and_execute_v3",
    "PreLLM",
]


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
) -> Any:
    """One function to preprocess and execute — like litellm.completion() but with small LLM decomposition."""
    pipeline_name = _resolve_pipeline_name(pipeline, strategy)
    config_overrides, small_llm, large_llm, domain_rules = _load_config_overrides(
        config_path, small_llm, large_llm, domain_rules
    )
    kwargs.update(config_overrides)

    _record_config_trace(get_current_trace(), small_llm, large_llm, pipeline_name, config_path, user_context)

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
) -> Any:
    """Synchronous version of preprocess_and_execute()."""
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


# Backward-compatible alias
preprocess_and_execute_v3 = preprocess_and_execute
