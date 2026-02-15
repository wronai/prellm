"""Tests for PromptRegistry — YAML-based prompt loading, caching, Jinja2 rendering."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from prellm.prompt_registry import PromptNotFoundError, PromptRegistry, PromptRenderError


@pytest.fixture
def sample_prompts_yaml(tmp_path: Path) -> Path:
    """Create a temporary prompts.yaml for testing."""
    data = {
        "prompts": {
            "classify": {
                "system": (
                    "You are a query classifier.\n"
                    "Intents: {{ intents | default('deploy, query') }}\n"
                    "Respond ONLY with JSON."
                ),
                "max_tokens": 256,
                "temperature": 0.1,
            },
            "structure": {
                "system": "Extract structured fields.\n{% if context %}Context: {{ context }}{% endif %}",
                "max_tokens": 512,
                "temperature": 0.1,
            },
            "split": {
                "system": "Split into {{ max_subtasks | default(3) }} sub-questions.",
                "max_tokens": 256,
                "temperature": 0.1,
            },
            "enrich": {
                "system": "Enrich the query with missing context.",
                "max_tokens": 512,
                "temperature": 0.2,
            },
            "compose": {
                "system": "Compose a final prompt for the large LLM.",
                "max_tokens": 512,
                "temperature": 0.2,
            },
        }
    }
    path = tmp_path / "prompts.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


@pytest.fixture
def registry(sample_prompts_yaml: Path) -> PromptRegistry:
    return PromptRegistry(prompts_path=sample_prompts_yaml)


def test_prompt_registry_loads_yaml(registry: PromptRegistry):
    """Registry loads all prompts from YAML file."""
    prompts = registry.list_prompts()
    assert "classify" in prompts
    assert "structure" in prompts
    assert "split" in prompts
    assert "enrich" in prompts
    assert "compose" in prompts
    assert len(prompts) == 5


def test_prompt_registry_get_with_variables(registry: PromptRegistry):
    """Registry renders Jinja2 variables in prompts."""
    result = registry.get("classify", intents="deploy, create, delete")
    assert "deploy, create, delete" in result
    assert "Respond ONLY with JSON" in result


def test_prompt_registry_get_with_default(registry: PromptRegistry):
    """Registry uses Jinja2 default filter when variable not provided."""
    result = registry.get("classify")
    assert "deploy, query" in result


def test_prompt_registry_get_conditional(registry: PromptRegistry):
    """Registry handles Jinja2 conditionals."""
    with_ctx = registry.get("structure", context="production environment")
    assert "Context: production environment" in with_ctx

    without_ctx = registry.get("structure")
    assert "Context:" not in without_ctx


def test_prompt_registry_missing_prompt_raises(registry: PromptRegistry):
    """Registry raises PromptNotFoundError for missing prompts."""
    with pytest.raises(PromptNotFoundError, match="nonexistent"):
        registry.get("nonexistent")


def test_prompt_registry_validate_all_required(registry: PromptRegistry):
    """Registry validates that all required prompts exist."""
    errors = registry.validate()
    assert errors == []


def test_prompt_registry_validate_missing_required(tmp_path: Path):
    """Registry reports missing required prompts."""
    path = tmp_path / "incomplete.yaml"
    with open(path, "w") as f:
        yaml.dump({"prompts": {"classify": {"system": "test"}}}, f)

    reg = PromptRegistry(prompts_path=path)
    errors = reg.validate()
    assert any("Missing required prompts" in e for e in errors)


def test_prompt_registry_get_entry(registry: PromptRegistry):
    """Registry returns full PromptEntry with metadata."""
    entry = registry.get_entry("classify")
    assert entry.name == "classify"
    assert entry.max_tokens == 256
    assert entry.temperature == 0.1
    assert "classifier" in entry.system_template


def test_prompt_registry_caching(registry: PromptRegistry):
    """Registry loads YAML only once (lazy loading)."""
    # First access triggers load
    registry.get("classify")
    assert registry._loaded is True

    # Modify internal state to prove no reload
    registry._entries["classify"].max_tokens = 999
    entry = registry.get_entry("classify")
    assert entry.max_tokens == 999  # Still cached, not reloaded


def test_prompt_registry_register_programmatic(registry: PromptRegistry):
    """Registry supports programmatic prompt registration."""
    registry.register("custom", "Custom prompt: {{ name }}", max_tokens=100)
    result = registry.get("custom", name="test")
    assert result == "Custom prompt: test"
    assert "custom" in registry.list_prompts()


def test_prompt_registry_nonexistent_file():
    """Registry handles missing YAML file gracefully."""
    reg = PromptRegistry(prompts_path="/nonexistent/prompts.yaml")
    assert reg.list_prompts() == []


def test_prompt_registry_default_path():
    """Registry uses default configs/prompts.yaml path."""
    reg = PromptRegistry()
    # Should load without error (file may or may not exist in test env)
    prompts = reg.list_prompts()
    assert isinstance(prompts, list)
