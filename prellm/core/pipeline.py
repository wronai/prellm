"""Pipeline execution helpers for preLLM preprocessing."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from prellm.agents.preprocessor import PreprocessorAgent
from prellm.agents.executor import ExecutorAgent
from prellm.llm_provider import LLMProvider
from prellm.models import LLMProviderConfig
from prellm.pipeline import PromptPipeline
from prellm.prompt_registry import PromptRegistry

logger = logging.getLogger("prellm")


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
    from prellm.analyzers.context_engine import ContextEngine
    from prellm.validators import ResponseValidator
    from prellm.core.context import _prepare_context, _build_pipeline_context
    from prellm.core.prompts import _build_executor_system_prompt
    from prellm.core.tracing import _persist_session, _record_trace
    from prellm.core.results import _build_decomposition_result
    from prellm.trace import get_current_trace
    from prellm.models import PreLLMResponse

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
