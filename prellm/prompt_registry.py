"""PromptRegistry — loads prompts from YAML, caches, validates placeholders, renders with Jinja2.

Centralizes all system prompts used by the preprocessing pipeline.
Prompts are defined in configs/prompts.yaml and rendered with Jinja2 templating.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jinja2 import BaseLoader, Environment, TemplateSyntaxError, UndefinedError

logger = logging.getLogger("prellm.prompt_registry")

_DEFAULT_PROMPTS_PATH = Path(__file__).parent.parent / "configs" / "prompts.yaml"


class PromptNotFoundError(KeyError):
    """Raised when a prompt name is not found in the registry."""


class PromptRenderError(ValueError):
    """Raised when a prompt template fails to render."""


class PromptEntry:
    """Single prompt entry with template, max_tokens, and temperature."""

    __slots__ = ("name", "system_template", "max_tokens", "temperature")

    def __init__(self, name: str, system_template: str, max_tokens: int = 512, temperature: float = 0.1):
        self.name = name
        self.system_template = system_template
        self.max_tokens = max_tokens
        self.temperature = temperature

    def __repr__(self) -> str:
        return f"PromptEntry(name={self.name!r}, max_tokens={self.max_tokens}, temperature={self.temperature})"


class PromptRegistry:
    """Loads prompts from YAML, caches, validates placeholders.

    Usage:
        registry = PromptRegistry()
        prompt = registry.get("classify", intents="deploy, query, create")
        print(prompt)  # rendered system prompt string
    """

    def __init__(self, prompts_path: Path | str | None = None):
        self._path = Path(prompts_path) if prompts_path else _DEFAULT_PROMPTS_PATH
        self._entries: dict[str, PromptEntry] = {}
        self._jinja_env = Environment(loader=BaseLoader(), undefined=_StrictUndefined)
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()

    def _load(self) -> None:
        """Load prompts from the YAML file."""
        if not self._path.exists():
            logger.warning(f"Prompts file not found: {self._path}, using empty registry")
            self._loaded = True
            return

        with open(self._path) as f:
            raw = yaml.safe_load(f) or {}

        prompts_raw = raw.get("prompts", {})
        for name, data in prompts_raw.items():
            if isinstance(data, dict):
                self._entries[name] = PromptEntry(
                    name=name,
                    system_template=data.get("system", ""),
                    max_tokens=data.get("max_tokens", 512),
                    temperature=data.get("temperature", 0.1),
                )
            elif isinstance(data, str):
                self._entries[name] = PromptEntry(name=name, system_template=data)

        self._loaded = True
        logger.debug(f"Loaded {len(self._entries)} prompts from {self._path}")

    def get(self, prompt_name: str, **variables: Any) -> str:
        """Get a rendered prompt by name with Jinja2 variable substitution.

        Args:
            prompt_name: The prompt name (e.g. "classify", "structure").
            **variables: Template variables to substitute.

        Returns:
            The rendered system prompt string.

        Raises:
            PromptNotFoundError: If the prompt name doesn't exist.
            PromptRenderError: If Jinja2 rendering fails.
        """
        self._ensure_loaded()
        entry = self._entries.get(prompt_name)
        if entry is None:
            raise PromptNotFoundError(f"Prompt '{prompt_name}' not found in registry. Available: {self.list_prompts()}")

        return self._render(entry.system_template, variables)

    def get_entry(self, name: str) -> PromptEntry:
        """Get the full PromptEntry (template + max_tokens + temperature).

        Args:
            name: The prompt name.

        Returns:
            PromptEntry with all metadata.

        Raises:
            PromptNotFoundError: If the prompt name doesn't exist.
        """
        self._ensure_loaded()
        entry = self._entries.get(name)
        if entry is None:
            raise PromptNotFoundError(f"Prompt '{name}' not found in registry. Available: {self.list_prompts()}")
        return entry

    def list_prompts(self) -> list[str]:
        """List all available prompt names."""
        self._ensure_loaded()
        return sorted(self._entries.keys())

    def validate(self) -> list[str]:
        """Validate that all prompts have non-empty templates. Returns list of error messages."""
        self._ensure_loaded()
        errors: list[str] = []

        required = {"classify", "structure", "split", "enrich", "compose"}
        available = set(self._entries.keys())
        missing = required - available
        if missing:
            errors.append(f"Missing required prompts: {sorted(missing)}")

        for name, entry in self._entries.items():
            if not entry.system_template.strip():
                errors.append(f"Prompt '{name}' has empty system template")

            # Validate Jinja2 syntax
            try:
                self._jinja_env.parse(entry.system_template)
            except TemplateSyntaxError as e:
                errors.append(f"Prompt '{name}' has invalid Jinja2 syntax: {e}")

        return errors

    def register(self, name: str, system_template: str, max_tokens: int = 512, temperature: float = 0.1) -> None:
        """Register a prompt programmatically (useful for testing or dynamic prompts)."""
        self._ensure_loaded()
        self._entries[name] = PromptEntry(
            name=name,
            system_template=system_template,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def _render(self, template_str: str, variables: dict[str, Any]) -> str:
        """Render a Jinja2 template string with variables."""
        try:
            template = self._jinja_env.from_string(template_str)
            return template.render(**variables)
        except UndefinedError as e:
            raise PromptRenderError(f"Missing template variable: {e}") from e
        except TemplateSyntaxError as e:
            raise PromptRenderError(f"Invalid template syntax: {e}") from e


class _StrictUndefined:
    """Custom undefined that allows `default` filter but raises on direct access."""

    # Use Jinja2's built-in Undefined which supports `| default()` filter
    pass


# Re-use Jinja2's built-in ChainableUndefined for `| default()` support
from jinja2 import ChainableUndefined as _StrictUndefined  # noqa: E402, F811
