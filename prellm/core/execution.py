"""Pipeline execution orchestration helpers for preLLM preprocessing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prellm.agents.executor import ExecutorAgent
from prellm.agents.preprocessor import PreprocessorAgent
from prellm.llm_provider import LLMProvider
from prellm.models import LLMProviderConfig, PreLLMResponse
from prellm.pipeline import PromptPipeline
from prellm.prompt_registry import PromptRegistry


@dataclass
class PipelineRuntime:
    """Fully built runtime objects for the two-agent pipeline."""

    preprocessor: PreprocessorAgent
    executor: ExecutorAgent
    user_memory: Any = None


@dataclass
class PipelineExecutionArtifacts:
    """Intermediate results captured while running the pipeline."""

    extra_context: dict[str, Any]
    pipeline_context: dict[str, Any]
    prep_result: Any
    prep_duration_ms: float
    exec_result: Any
    exec_duration_ms: float


def _build_llm_providers(
    small_llm: str,
    large_llm: str,
    max_tokens: int,
    temperature: float,
) -> tuple[LLMProvider, LLMProvider]:
    """Build small and large LLM provider instances."""
    small_llm_config = LLMProviderConfig(model=small_llm, max_tokens=512, temperature=0.0)
    large_llm_config = LLMProviderConfig(model=large_llm, max_tokens=max_tokens, temperature=temperature)
    return LLMProvider(small_llm_config), LLMProvider(large_llm_config)


def _build_prompt_registry(prompts_path: str | Path | None) -> PromptRegistry:
    """Build prompt registry from YAML files."""
    return PromptRegistry(prompts_path=prompts_path)


def _build_prompt_pipeline(
    pipelines_path: str | Path | None,
    pipeline: str,
    registry: PromptRegistry,
    small_provider: LLMProvider,
) -> PromptPipeline:
    """Build pipeline definition for the preprocessor."""
    return PromptPipeline.from_yaml(
        pipelines_path=pipelines_path,
        pipeline_name=pipeline,
        registry=registry,
        small_llm=small_provider,
    )


def _build_preprocessor(
    small_provider: LLMProvider,
    registry: PromptRegistry,
    prompt_pipeline: PromptPipeline,
    user_memory: Any,
    codebase_indexer: Any,
    codebase_path: str | Path | None,
) -> PreprocessorAgent:
    """Build the preprocessor agent with all optional context sources."""
    from prellm.analyzers.context_engine import ContextEngine

    return PreprocessorAgent(
        small_llm=small_provider,
        registry=registry,
        pipeline=prompt_pipeline,
        context_engine=ContextEngine(),
        user_memory=user_memory,
        codebase_indexer=codebase_indexer,
        codebase_path=str(codebase_path) if codebase_path else None,
    )


def _build_executor(
    large_provider: LLMProvider,
    schemas_path: str | Path | None,
    sensitive_filter: Any,
) -> ExecutorAgent:
    """Build the executor agent and optional response validator."""
    from prellm.validators import ResponseValidator

    return ExecutorAgent(
        large_llm=large_provider,
        response_validator=ResponseValidator(schemas_path=schemas_path) if schemas_path else None,
        sensitive_filter=sensitive_filter,
    )


def _build_pipeline_runtime(
    small_llm: str,
    large_llm: str,
    max_tokens: int,
    temperature: float,
    prompts_path: str | Path | None,
    pipelines_path: str | Path | None,
    pipeline: str,
    schemas_path: str | Path | None,
    user_memory: Any,
    codebase_indexer: Any,
    sensitive_filter: Any,
    codebase_path: str | Path | None,
) -> PipelineRuntime:
    """Build all runtime objects needed by the two-agent pipeline."""
    small_provider, large_provider = _build_llm_providers(
        small_llm=small_llm,
        large_llm=large_llm,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    registry = _build_prompt_registry(prompts_path)
    prompt_pipeline = _build_prompt_pipeline(
        pipelines_path=pipelines_path,
        pipeline=pipeline,
        registry=registry,
        small_provider=small_provider,
    )
    preprocessor = _build_preprocessor(
        small_provider=small_provider,
        registry=registry,
        prompt_pipeline=prompt_pipeline,
        user_memory=user_memory,
        codebase_indexer=codebase_indexer,
        codebase_path=codebase_path,
    )
    executor = _build_executor(
        large_provider=large_provider,
        schemas_path=schemas_path,
        sensitive_filter=sensitive_filter,
    )
    return PipelineRuntime(preprocessor=preprocessor, executor=executor, user_memory=user_memory)


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


async def _run_pipeline_stages(
    runtime: PipelineRuntime,
    query: str,
    pipeline: str,
    extra_context: dict[str, Any],
    kwargs: dict[str, Any],
) -> PipelineExecutionArtifacts:
    """Run preprocessing, execution, and session persistence for a pipeline run."""
    from prellm.core.context import _build_pipeline_context
    from prellm.core.prompts import _build_executor_system_prompt
    from prellm.core.tracing import _persist_session

    pipeline_context = _build_pipeline_context(extra_context)
    prep_result, prep_duration_ms = await _run_preprocessing(
        runtime.preprocessor,
        query,
        pipeline_context,
        pipeline,
    )
    system_prompt = _build_executor_system_prompt(prep_result, extra_context)
    exec_result, exec_duration_ms = await _run_execution(
        runtime.executor,
        prep_result.executor_input,
        system_prompt=system_prompt,
        **kwargs,
    )
    await _persist_session(runtime.user_memory, query, exec_result)
    return PipelineExecutionArtifacts(
        extra_context=extra_context,
        pipeline_context=pipeline_context,
        prep_result=prep_result,
        prep_duration_ms=prep_duration_ms,
        exec_result=exec_result,
        exec_duration_ms=exec_duration_ms,
    )


def _build_pipeline_response(
    query: str,
    pipeline: str,
    small_llm: str,
    large_llm: str,
    artifacts: PipelineExecutionArtifacts,
) -> PreLLMResponse:
    """Build the final response object and record trace metadata."""
    from prellm.core.results import _build_decomposition_result
    from prellm.core.tracing import _record_trace
    from prellm.trace import get_current_trace

    trace = get_current_trace()
    _record_trace(
        trace,
        pipeline,
        small_llm,
        large_llm,
        query,
        artifacts.pipeline_context,
        artifacts.prep_result,
        artifacts.exec_result,
        artifacts.prep_duration_ms,
        artifacts.exec_duration_ms,
    )

    decomposition_result = _build_decomposition_result(query, pipeline, artifacts.prep_result)
    response = PreLLMResponse(
        content=artifacts.exec_result.content or "No response from any model.",
        decomposition=decomposition_result,
        model_used=artifacts.exec_result.model_used,
        small_model_used=small_llm,
        retries=artifacts.exec_result.retries,
        clarified=bool(decomposition_result and decomposition_result.missing_fields),
        needs_more_context=bool(decomposition_result and decomposition_result.missing_fields) and not artifacts.exec_result.content,
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
