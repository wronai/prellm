"""PreprocessorAgent — small LLM (≤24B) analyzes and structures user queries.

Responsible for:
- Gathering context (env, git, user history)
- Executing PromptPipeline (classify → structure → compose)
- Validating intermediate results
- Producing structured executor_input

Does NOT call the large LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from nfo.decorators import log_call
from pydantic import BaseModel, Field

from prellm.analyzers.context_engine import ContextEngine
from prellm.llm_provider import LLMProvider
from prellm.pipeline import PipelineResult, PromptPipeline
from prellm.prompt_registry import PromptRegistry

logger = logging.getLogger("prellm.agents.preprocessor")

# Lazy imports to avoid hard dependencies
_UserMemory = None
_CodebaseIndexer = None

def _get_user_memory_class():
    global _UserMemory
    if _UserMemory is None:
        from prellm.context.user_memory import UserMemory
        _UserMemory = UserMemory
    return _UserMemory

def _get_codebase_indexer_class():
    global _CodebaseIndexer
    if _CodebaseIndexer is None:
        from prellm.context.codebase_indexer import CodebaseIndexer
        _CodebaseIndexer = CodebaseIndexer
    return _CodebaseIndexer


class PreprocessResult(BaseModel):
    """Output of the PreprocessorAgent — structured input for the ExecutorAgent."""
    original_query: str = ""
    executor_input: str = ""
    decomposition: PipelineResult | None = None
    context_used: dict[str, Any] = Field(default_factory=dict)
    pipeline_name: str = ""
    confidence: float = 0.0


class PreprocessorAgent:
    """Agent preprocessing — small LLM (≤24B) analyzes and structures queries.

    Usage:
        agent = PreprocessorAgent(
            small_llm=LLMProvider(config),
            registry=PromptRegistry(),
            pipeline=PromptPipeline.from_yaml(...),
            context_engine=ContextEngine([...]),
        )
        result = await agent.preprocess("Deploy app to production")
    """

    def __init__(
        self,
        small_llm: LLMProvider,
        registry: PromptRegistry,
        pipeline: PromptPipeline,
        context_engine: ContextEngine | None = None,
        user_memory: Any | None = None,
        codebase_indexer: Any | None = None,
        codebase_path: str | None = None,
    ):
        self.small_llm = small_llm
        self.registry = registry
        self.pipeline = pipeline
        self.context_engine = context_engine or ContextEngine()
        self.user_memory = user_memory
        self.codebase_indexer = codebase_indexer
        self._codebase_index = None

        # Pre-index codebase if path provided
        if codebase_path and codebase_indexer:
            try:
                self._codebase_index = codebase_indexer.index_directory(codebase_path)
                logger.info(f"Indexed codebase: {self._codebase_index.total_files} files, {self._codebase_index.total_symbols} symbols")
            except Exception as e:
                logger.warning(f"Failed to index codebase at {codebase_path}: {e}")

    @log_call
    async def preprocess(
        self,
        query: str,
        user_context: dict[str, Any] | None = None,
        pipeline_name: str | None = None,
    ) -> PreprocessResult:
        """Preprocess a query and return structured input for the Executor.

        Args:
            query: The raw user query.
            user_context: Optional extra context dict.
            pipeline_name: Override pipeline name for tracking.

        Returns:
            PreprocessResult with executor_input and decomposition details.
        """
        # 1. Gather environment context
        env_ctx = self.context_engine.gather()
        full_ctx: dict[str, Any] = {**env_ctx, **(user_context or {})}

        # 2. Enrich with UserMemory (recent interactions)
        if self.user_memory:
            try:
                recent = await self.user_memory.get_recent_context(query, limit=3)
                if recent:
                    history_lines = [f"- {r['query']}: {r['response_summary'][:100]}" for r in recent]
                    full_ctx["user_history"] = "\n".join(history_lines)
                prefs = await self.user_memory.get_user_preferences()
                if prefs:
                    full_ctx["user_preferences"] = ", ".join(f"{k}={v}" for k, v in prefs.items())
            except Exception as e:
                logger.warning(f"UserMemory enrichment failed: {e}")

        # 3. Enrich with CodebaseIndexer (relevant symbols)
        if self.codebase_indexer and self._codebase_index:
            try:
                codebase_ctx = self.codebase_indexer.get_context_for_query(self._codebase_index, query)
                if codebase_ctx:
                    full_ctx["codebase_context"] = codebase_ctx
            except Exception as e:
                logger.warning(f"CodebaseIndexer enrichment failed: {e}")

        # 4. Execute pipeline preprocessing
        pipeline_result = await self.pipeline.execute(query, context=full_ctx)

        # 3. Extract composed prompt from pipeline state
        executor_input = self._extract_executor_input(query, pipeline_result)

        # 4. Extract confidence if available
        confidence = self._extract_confidence(pipeline_result)

        return PreprocessResult(
            original_query=query,
            executor_input=executor_input,
            decomposition=pipeline_result,
            context_used=full_ctx,
            pipeline_name=pipeline_name or self.pipeline.config.name,
            confidence=confidence,
        )

    @staticmethod
    def _extract_executor_input(query: str, pipeline_result: PipelineResult) -> str:
        """Extract the best executor input from pipeline state."""
        state = pipeline_result.state

        # Try common output keys in priority order
        for key in ("composed_prompt", "executor_input", "meta_prompt", "enriched_query", "enriched"):
            value = state.get(key)
            if value:
                if isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    return value.get("composed_prompt", str(value))

        # Fallback: return original query
        return state.get("query", query)

    @staticmethod
    def _extract_confidence(pipeline_result: PipelineResult) -> float:
        """Extract confidence score from classification step if available."""
        classification = pipeline_result.state.get("classification", {})
        if isinstance(classification, dict):
            return float(classification.get("confidence", 0.0))
        return 0.0
