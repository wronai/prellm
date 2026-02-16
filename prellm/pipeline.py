"""PromptPipeline — generic, YAML-configurable pipeline for multi-step LLM preprocessing.

Supports two types of steps:
  1. LLM steps — call a small LLM with a prompt from PromptRegistry
  2. Algorithmic steps — execute Python logic (domain matching, validation, formatting)

Pipelines are defined in configs/pipelines.yaml and executed sequentially with
optional conditional skipping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import yaml
from nfo.decorators import log_call
from pydantic import BaseModel, Field

from prellm.llm_provider import LLMProvider
from prellm.prompt_registry import PromptRegistry

logger = logging.getLogger("prellm.pipeline")

_DEFAULT_PIPELINES_PATH = Path(__file__).parent.parent / "configs" / "pipelines.yaml"


# ============================================================
# Pipeline Config Models
# ============================================================

class PipelineStep(BaseModel):
    """Configuration for a single pipeline step."""
    name: str
    prompt: str | None = None        # prompt name from PromptRegistry (LLM step)
    type: str | None = None           # algorithmic step type (non-LLM step)
    input: str | list[str] = "query"
    output: str = ""
    condition: str | None = None
    parallel: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    output_schema: str | None = None   # schema name for output validation


class PipelineConfig(BaseModel):
    """Configuration for a complete pipeline."""
    name: str
    description: str = ""
    steps: list[PipelineStep] = Field(default_factory=list)


class StepExecutionResult(BaseModel):
    """Result of executing a single pipeline step."""
    step_name: str
    step_type: str = "llm"  # "llm" or "algo"
    output_key: str = ""
    output_value: Any = None
    skipped: bool = False
    error: str | None = None


class PipelineResult(BaseModel):
    """Result of executing a full pipeline."""
    state: dict[str, Any] = Field(default_factory=dict)
    steps_executed: list[StepExecutionResult] = Field(default_factory=list)
    pipeline_name: str = ""
    success: bool = True
    error: str | None = None


# ============================================================
# Pipeline Engine
# ============================================================

class PromptPipeline:
    """Generic pipeline — executes a sequence of LLM + algorithmic steps.

    Usage:
        pipeline = PromptPipeline(
            config=PipelineConfig(name="classify", steps=[...]),
            registry=PromptRegistry(),
            small_llm=LLMProvider(config),
        )
        result = await pipeline.execute("Deploy app to prod")
    """

    def __init__(
        self,
        config: PipelineConfig,
        registry: PromptRegistry,
        small_llm: LLMProvider,
        validators: dict[str, Callable[..., Any]] | None = None,
    ):
        self.config = config
        self.registry = registry
        self.small_llm = small_llm
        self._algo_handlers: dict[str, Callable[..., Any]] = {
            "domain_rule_matcher": self._algo_domain_rule_matcher,
            "field_validator": self._algo_field_validator,
            "yaml_formatter": self._algo_yaml_formatter,
            "runtime_collector": self._algo_runtime_collector,
            "sensitive_filter": self._algo_sensitive_filter,
            "session_injector": self._algo_session_injector,
            "shell_context_collector": self._algo_shell_context_collector,
            "folder_compressor": self._algo_folder_compressor,
            "context_schema_generator": self._algo_context_schema_generator,
        }
        if validators:
            self._algo_handlers.update(validators)

    @classmethod
    def from_yaml(
        cls,
        pipelines_path: Path | str | None,
        pipeline_name: str,
        registry: PromptRegistry | None = None,
        small_llm: LLMProvider | None = None,
        validators: dict[str, Callable[..., Any]] | None = None,
    ) -> PromptPipeline:
        """Load a named pipeline from a YAML file.

        Args:
            pipelines_path: Path to pipelines.yaml (or None for default).
            pipeline_name: Name of the pipeline to load.
            registry: PromptRegistry instance (or None to create default).
            small_llm: LLMProvider for LLM steps (required for execution).
            validators: Additional algorithmic step handlers.

        Returns:
            Configured PromptPipeline instance.

        Raises:
            KeyError: If pipeline_name not found in YAML.
        """
        path = Path(pipelines_path) if pipelines_path else _DEFAULT_PIPELINES_PATH

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        pipelines_raw = raw.get("pipelines", {})
        if pipeline_name not in pipelines_raw:
            available = sorted(pipelines_raw.keys())
            raise KeyError(f"Pipeline '{pipeline_name}' not found. Available: {available}")

        pipe_data = pipelines_raw[pipeline_name]
        steps = []
        for step_raw in pipe_data.get("steps", []):
            # Normalize input field
            input_val = step_raw.get("input", "query")
            steps.append(PipelineStep(
                name=step_raw["name"],
                prompt=step_raw.get("prompt"),
                type=step_raw.get("type"),
                input=input_val,
                output=step_raw.get("output", ""),
                condition=step_raw.get("condition"),
                parallel=step_raw.get("parallel", False),
                config=step_raw.get("config", {}),
                output_schema=step_raw.get("schema"),
            ))

        config = PipelineConfig(
            name=pipeline_name,
            description=pipe_data.get("description", ""),
            steps=steps,
        )

        return cls(
            config=config,
            registry=registry or PromptRegistry(),
            small_llm=small_llm,  # type: ignore[arg-type]
            validators=validators,
        )

    @log_call
    async def execute(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Execute pipeline step by step, collecting intermediate results.

        Args:
            query: The raw user query.
            context: Optional context dict (env, git, user history, etc.).

        Returns:
            PipelineResult with full state and step execution details.
        """
        state: dict[str, Any] = {"query": query, "context": context or {}}
        steps_executed: list[StepExecutionResult] = []

        for step in self.config.steps:
            step_result = StepExecutionResult(
                step_name=step.name,
                output_key=step.output,
            )

            # Check condition
            if step.condition and not self._evaluate_condition(step.condition, state):
                step_result.skipped = True
                steps_executed.append(step_result)
                logger.debug(f"Step '{step.name}' skipped (condition not met)")
                continue

            try:
                if step.prompt:
                    # LLM step
                    step_result.step_type = "llm"
                    result = await self._execute_llm_step(step, state)
                elif step.type:
                    # Algorithmic step
                    step_result.step_type = "algo"
                    result = self._execute_algo_step(step, state)
                else:
                    logger.warning(f"Step '{step.name}' has neither prompt nor type — skipping")
                    step_result.skipped = True
                    steps_executed.append(step_result)
                    continue

                if step.output:
                    state[step.output] = result
                step_result.output_value = result

            except Exception as e:
                logger.error(f"Step '{step.name}' failed: {e}")
                step_result.error = str(e)
                steps_executed.append(step_result)
                return PipelineResult(
                    state=state,
                    steps_executed=steps_executed,
                    pipeline_name=self.config.name,
                    success=False,
                    error=f"Step '{step.name}' failed: {e}",
                )

            steps_executed.append(step_result)

        return PipelineResult(
            state=state,
            steps_executed=steps_executed,
            pipeline_name=self.config.name,
            success=True,
        )

    async def _execute_llm_step(self, step: PipelineStep, state: dict[str, Any]) -> Any:
        """Execute an LLM step — call small_llm with prompt from registry."""
        # Gather input variables for prompt rendering
        template_vars = self._gather_inputs(step, state)
        template_vars.update(step.config)

        # Get rendered prompt
        system_prompt = self.registry.get(step.prompt, **template_vars)

        # Build user message from inputs
        user_message = self._build_user_message(step, state)

        # Get entry for max_tokens/temperature
        entry = self.registry.get_entry(step.prompt)

        # Call LLM
        result = await self.small_llm.complete_json(
            user_message=user_message,
            system_prompt=system_prompt,
        )

        logger.debug(f"LLM step '{step.name}' result: {str(result)[:200]}")
        return result

    def _execute_algo_step(self, step: PipelineStep, state: dict[str, Any]) -> Any:
        """Execute an algorithmic (non-LLM) step."""
        handler = self._algo_handlers.get(step.type or "")
        if handler is None:
            raise ValueError(f"Unknown algorithmic step type: '{step.type}'. "
                           f"Available: {sorted(self._algo_handlers.keys())}")

        inputs = self._gather_inputs(step, state)
        return handler(inputs, state, step.config)

    def _gather_inputs(self, step: PipelineStep, state: dict[str, Any]) -> dict[str, Any]:
        """Gather input values from state for a step."""
        result: dict[str, Any] = {}
        if isinstance(step.input, str):
            keys = [step.input]
        else:
            keys = step.input

        for key in keys:
            if key in state:
                result[key] = state[key]
        return result

    def _build_user_message(self, step: PipelineStep, state: dict[str, Any]) -> str:
        """Build user message string from step inputs."""
        parts: list[str] = []
        if "query" in state:
            parts.append(f"Query: {state['query']}")

        inputs = self._gather_inputs(step, state)
        for key, value in inputs.items():
            if key != "query":
                parts.append(f"{key}: {value}")

        return "\n".join(parts) if parts else state.get("query", "")

    def _evaluate_condition(self, condition: str, state: dict[str, Any]) -> bool:
        """Safely evaluate a condition string against state.

        Supports simple expressions like:
          - "matched_rule.get('action') == 'enrich'"
          - "classification.get('confidence', 0) > 0.5"
        """
        try:
            return bool(eval(condition, {"__builtins__": {}}, state))  # noqa: S307
        except Exception:
            logger.debug(f"Condition '{condition}' evaluated to False (exception)")
            return False

    def register_algo_handler(self, name: str, handler: Callable[..., Any]) -> None:
        """Register a custom algorithmic step handler."""
        self._algo_handlers[name] = handler

    # ============================================================
    # Built-in algorithmic step handlers
    # ============================================================

    @staticmethod
    def _algo_domain_rule_matcher(
        inputs: dict[str, Any], state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Match classification against domain rules in context."""
        classification = inputs.get("classification", {})
        domain_rules = state.get("context", {}).get("domain_rules", [])

        if not domain_rules or not classification:
            return {}

        intent = classification.get("intent", "") if isinstance(classification, dict) else ""
        best_match: dict[str, Any] = {}
        best_score = 0

        for rule in domain_rules:
            score = 0
            if isinstance(rule, dict) and rule.get("intent") == intent:
                score += 2
            if score > best_score:
                best_score = score
                best_match = rule if isinstance(rule, dict) else {}

        return best_match

    @staticmethod
    def _algo_field_validator(
        inputs: dict[str, Any], state: dict[str, Any], config: dict[str, Any]
    ) -> list[str]:
        """Validate extracted fields against required fields from matched rule."""
        fields = inputs.get("fields", {})
        matched_rule = inputs.get("matched_rule", state.get("matched_rule", {}))

        if not isinstance(matched_rule, dict):
            return []

        required = matched_rule.get("required_fields", [])
        field_keys = set(fields.keys()) if isinstance(fields, dict) else set()

        return [f for f in required if f not in field_keys]

    @staticmethod
    def _algo_yaml_formatter(
        inputs: dict[str, Any], state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Format pipeline state into structured executor input."""
        meta_prompt = inputs.get("meta_prompt", "")
        query = state.get("query", "")

        return {
            "original_query": query,
            "composed_prompt": meta_prompt if isinstance(meta_prompt, str) else str(meta_prompt),
            "context": state.get("context", {}),
            "pipeline_state": {
                k: v for k, v in state.items()
                if k not in ("query", "context")
            },
        }

    @staticmethod
    def _algo_runtime_collector(
        inputs: dict[str, Any], state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect RuntimeContext and add to state."""
        try:
            from prellm.analyzers.context_engine import ContextEngine
            engine = ContextEngine()
            runtime = engine.gather_runtime()
            return runtime.model_dump()
        except Exception as e:
            logger.warning(f"Runtime collection failed: {e}")
            return {}

    @staticmethod
    def _algo_sensitive_filter(
        inputs: dict[str, Any], state: dict[str, Any], config: dict[str, Any]
    ) -> str:
        """Filter state through SensitiveDataFilter before output."""
        try:
            from prellm.context.sensitive_filter import SensitiveDataFilter
            sf = SensitiveDataFilter()
            # Filter the composed prompt text
            text = ""
            for key in ("composed_prompt", "sanitized_prompt", "query"):
                val = inputs.get(key) or state.get(key)
                if val and isinstance(val, str):
                    text = val
                    break
            if not text:
                text = str(inputs) if inputs else state.get("query", "")
            return sf.sanitize_text(text)
        except Exception as e:
            logger.warning(f"Sensitive filtering failed: {e}")
            return inputs.get("composed_prompt", state.get("query", ""))

    @staticmethod
    def _algo_session_injector(
        inputs: dict[str, Any], state: dict[str, Any], config: dict[str, Any]
    ) -> str:
        """Inject relevant context from UserMemory (RAG-style)."""
        try:
            import asyncio
            memory_path = state.get("context", {}).get("memory_path")
            if not memory_path:
                return ""
            from prellm.context.user_memory import UserMemory
            memory = UserMemory(path=memory_path)
            query = state.get("query", "")
            # Run async in sync context
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(memory.auto_inject_context(query))
            finally:
                loop.close()
                memory.close()
            return result
        except Exception as e:
            logger.warning(f"Session injection failed: {e}")
            return ""

    @staticmethod
    def _algo_shell_context_collector(
        inputs: dict[str, Any], state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect shell context as an algo step."""
        from prellm.context.shell_collector import ShellContextCollector
        collector = ShellContextCollector()
        ctx = collector.collect_all()
        return ctx.model_dump()

    @staticmethod
    def _algo_folder_compressor(
        inputs: dict[str, Any], state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Compress folder as an algo step."""
        from prellm.context.folder_compressor import FolderCompressor
        codebase_path = state.get("context", {}).get("codebase_path", ".")
        compressor = FolderCompressor()
        compressed = compressor.compress(codebase_path)
        return compressed.model_dump()

    @staticmethod
    def _algo_context_schema_generator(
        inputs: dict[str, Any], state: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate context schema as an algo step."""
        from prellm.context.schema_generator import ContextSchemaGenerator
        from prellm.models import ShellContext, CompressedFolder
        shell_data = inputs.get("shell_context")
        folder_data = inputs.get("compressed_folder")
        shell_ctx = ShellContext(**shell_data) if isinstance(shell_data, dict) else None
        folder_ctx = CompressedFolder(**folder_data) if isinstance(folder_data, dict) else None
        gen = ContextSchemaGenerator()
        schema = gen.generate(shell_context=shell_ctx, folder_compressed=folder_ctx)
        return schema.model_dump()

