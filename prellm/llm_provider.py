"""LLMProvider — unified abstraction for small and large LLM calls with retry/fallback.

Wraps LiteLLM to provide consistent interface for both the small preprocessing
model (≤3B) and the large target model (GPT-4/Claude/Llama).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from prellm.models import LLMProviderConfig

logger = logging.getLogger("prellm.llm_provider")


class LLMProvider:
    """Unified LLM caller with retry and fallback support.

    Usage:
        provider = LLMProvider(LLMProviderConfig(model="phi3:mini", fallback=["qwen2:1.5b"]))
        result = await provider.complete("Classify this query", system_prompt="You are a classifier.")
    """

    def __init__(self, config: LLMProviderConfig):
        self.config = config

    async def complete(
        self,
        user_message: str,
        system_prompt: str = "",
        response_format: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a completion request with retry/fallback.

        Args:
            user_message: The user message content.
            system_prompt: Optional system prompt prepended to messages.
            response_format: If "json", hint the model to return JSON.
            **kwargs: Extra kwargs passed to litellm.acompletion.

        Returns:
            The response content string.
        """
        import litellm

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        models_to_try = [self.config.model] + [
            m for m in self.config.fallback if m != self.config.model
        ]

        last_error: Exception | None = None

        for model in models_to_try:
            for attempt in range(self.config.max_retries):
                try:
                    completion_kwargs: dict[str, Any] = {
                        "model": model,
                        "messages": messages,
                        "max_tokens": self.config.max_tokens,
                        "timeout": self.config.timeout,
                        "temperature": self.config.temperature,
                        **kwargs,
                    }

                    if response_format == "json":
                        completion_kwargs["response_format"] = {"type": "json_object"}

                    resp = await litellm.acompletion(**completion_kwargs)
                    content = resp.choices[0].message.content or ""

                    logger.debug(f"LLM response from {model} (attempt {attempt + 1}): {content[:100]}...")
                    return content

                except Exception as e:
                    last_error = e
                    logger.warning(f"Attempt {attempt + 1} with {model} failed: {e}")

        raise RuntimeError(
            f"All models failed after retries. Last error: {last_error}"
        )

    async def complete_json(
        self,
        user_message: str,
        system_prompt: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Complete and parse response as JSON.

        Falls back to extracting JSON from the response text if the model
        doesn't return clean JSON.
        """
        raw = await self.complete(
            user_message=user_message,
            system_prompt=system_prompt,
            response_format="json",
            **kwargs,
        )
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Best-effort JSON extraction from LLM output."""
        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        if "```" in text:
            for block in text.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue

        # Try finding first { ... } or [ ... ]
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass

        logger.warning(f"Could not parse JSON from LLM output: {text[:200]}")
        return {}
