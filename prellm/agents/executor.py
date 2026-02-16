"""ExecutorAgent — large LLM (>24B) executes structured tasks.

Responsible for:
- Calling the large LLM with the structured prompt from PreprocessorAgent
- Retry/fallback between models
- Validating responses against response_schema
- Returning typed response

Does NOT perform preprocessing.
"""

from __future__ import annotations

import logging
from typing import Any

from nfo.decorators import log_call
from pydantic import BaseModel, Field

from prellm.llm_provider import LLMProvider
from prellm.validators import ResponseValidator, ValidationResult

logger = logging.getLogger("prellm.agents.executor")


class ExecutorResult(BaseModel):
    """Output of the ExecutorAgent."""
    content: str = ""
    model_used: str = ""
    schema_valid: bool | None = None
    validation: ValidationResult | None = None
    retries: int = 0


class ExecutorAgent:
    """Agent execution — large LLM (>24B) executes structured tasks.

    Usage:
        agent = ExecutorAgent(
            large_llm=LLMProvider(config),
            response_validator=ResponseValidator(),
            response_schema_name="final_response",
        )
        result = await agent.execute("structured prompt from preprocessor")
    """

    def __init__(
        self,
        large_llm: LLMProvider,
        response_validator: ResponseValidator | None = None,
        response_schema_name: str | None = None,
    ):
        self.large_llm = large_llm
        self.response_validator = response_validator
        self.response_schema_name = response_schema_name

    @log_call
    async def execute(
        self,
        executor_input: str,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> ExecutorResult:
        """Execute the task with the large LLM.

        Args:
            executor_input: Structured prompt from PreprocessorAgent.
            system_prompt: Optional system prompt override.
            **kwargs: Extra kwargs passed to the LLM call.

        Returns:
            ExecutorResult with content, model info, and validation status.
        """
        retries = 0

        try:
            content = await self.large_llm.complete(
                user_message=executor_input,
                system_prompt=system_prompt,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Large LLM call failed: {e}")
            content = ""
            retries += 1

        model_used = self.large_llm.config.model

        # Optional schema validation
        validation = None
        schema_valid = None
        if self.response_validator and self.response_schema_name and content:
            validation = self._validate_response(content)
            schema_valid = validation.valid if validation else None

        return ExecutorResult(
            content=content,
            model_used=model_used,
            schema_valid=schema_valid,
            validation=validation,
            retries=retries,
        )

    def _validate_response(self, content: str) -> ValidationResult | None:
        """Validate response content against the configured schema."""
        if not self.response_validator or not self.response_schema_name:
            return None

        import json
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return self.response_validator.validate(data, self.response_schema_name)
        except (json.JSONDecodeError, TypeError):
            pass

        # Non-JSON responses: validate as {"content": content}
        return self.response_validator.validate(
            {"content": content}, self.response_schema_name
        )
