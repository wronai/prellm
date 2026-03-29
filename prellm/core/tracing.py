"""Tracing and session persistence helpers for preLLM preprocessing."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("prellm")


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
