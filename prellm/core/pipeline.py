"""Pipeline execution helpers for preLLM preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prellm.core.context import _prepare_context
from prellm.core.execution import (
    _build_pipeline_response,
    _build_pipeline_runtime,
    _run_pipeline_stages,
)


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
) -> Any:  # Returns PreLLMResponse
    """Two-agent execution path — PreprocessorAgent + ExecutorAgent + PromptPipeline.

    v0.4 refactor: split into _prepare_context, _run_preprocessing, _run_execution_with_sanitize,
    _persist_session to reduce cyclomatic complexity.
    """
    max_tokens = kwargs.pop("max_tokens", 2048)
    temperature = kwargs.pop("temperature", 0.7)

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

    runtime = _build_pipeline_runtime(
        small_llm=small_llm,
        large_llm=large_llm,
        max_tokens=max_tokens,
        temperature=temperature,
        prompts_path=prompts_path,
        pipelines_path=pipelines_path,
        pipeline=pipeline,
        schemas_path=schemas_path,
        user_memory=user_memory,
        codebase_indexer=codebase_indexer,
        sensitive_filter=sensitive_filter,
        codebase_path=codebase_path,
    )

    artifacts = await _run_pipeline_stages(
        runtime=runtime,
        query=query,
        pipeline=pipeline,
        extra_context=extra_context,
        kwargs=kwargs,
    )

    return _build_pipeline_response(
        query=query,
        pipeline=pipeline,
        small_llm=small_llm,
        large_llm=large_llm,
        artifacts=artifacts,
    )
