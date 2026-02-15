"""Tests for PromptPipeline — YAML-configurable multi-step preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from prellm.llm_provider import LLMProvider
from prellm.models import LLMProviderConfig
from prellm.pipeline import (
    PipelineConfig,
    PipelineResult,
    PipelineStep,
    PromptPipeline,
)
from prellm.prompt_registry import PromptRegistry


@pytest.fixture
def sample_prompts_yaml(tmp_path: Path) -> Path:
    data = {
        "prompts": {
            "classify": {
                "system": "Classify the query. Respond with JSON.",
                "max_tokens": 256,
                "temperature": 0.1,
            },
            "structure": {
                "system": "Extract structured fields as JSON.",
                "max_tokens": 512,
                "temperature": 0.1,
            },
            "split": {
                "system": "Split into {{ max_subtasks | default(3) }} sub-questions. JSON.",
                "max_tokens": 256,
                "temperature": 0.1,
            },
            "enrich": {
                "system": "Enrich the query.",
                "max_tokens": 512,
                "temperature": 0.2,
            },
            "compose": {
                "system": "Compose a final prompt.",
                "max_tokens": 512,
                "temperature": 0.2,
            },
            "context_analyze": {
                "system": "Analyze context.",
                "max_tokens": 256,
                "temperature": 0.1,
            },
        }
    }
    path = tmp_path / "prompts.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


@pytest.fixture
def sample_pipelines_yaml(tmp_path: Path) -> Path:
    data = {
        "pipelines": {
            "classify": {
                "description": "Classify intent",
                "steps": [
                    {"name": "classify", "prompt": "classify", "output": "classification"},
                ],
            },
            "structure": {
                "description": "Full structural decomposition",
                "steps": [
                    {"name": "classify", "prompt": "classify", "output": "classification"},
                    {"name": "extract_fields", "prompt": "structure", "output": "fields"},
                    {
                        "name": "compose",
                        "prompt": "compose",
                        "input": ["query", "classification", "fields"],
                        "output": "composed_prompt",
                    },
                ],
            },
            "conditional": {
                "description": "Pipeline with conditional step",
                "steps": [
                    {"name": "classify", "prompt": "classify", "output": "classification"},
                    {
                        "name": "enrich_if_needed",
                        "prompt": "enrich",
                        "condition": "classification.get('confidence', 1.0) < 0.5",
                        "output": "enriched",
                    },
                ],
            },
            "with_algo": {
                "description": "Pipeline with algorithmic step",
                "steps": [
                    {"name": "classify", "prompt": "classify", "output": "classification"},
                    {
                        "name": "match_rule",
                        "type": "domain_rule_matcher",
                        "input": "classification",
                        "output": "matched_rule",
                    },
                ],
            },
            "passthrough": {
                "description": "No steps",
                "steps": [],
            },
        }
    }
    path = tmp_path / "pipelines.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


@pytest.fixture
def registry(sample_prompts_yaml: Path) -> PromptRegistry:
    return PromptRegistry(prompts_path=sample_prompts_yaml)


@pytest.fixture
def mock_llm() -> LLMProvider:
    """Create a mock LLMProvider that returns predictable JSON."""
    provider = MagicMock(spec=LLMProvider)
    provider.complete_json = AsyncMock(return_value={
        "intent": "deploy",
        "confidence": 0.9,
        "domain": "devops",
    })
    provider.complete = AsyncMock(return_value="composed prompt result")
    return provider


def _make_pipeline(
    registry: PromptRegistry,
    mock_llm: LLMProvider,
    steps: list[dict],
    name: str = "test",
) -> PromptPipeline:
    config = PipelineConfig(
        name=name,
        steps=[PipelineStep(**s) for s in steps],
    )
    return PromptPipeline(config=config, registry=registry, small_llm=mock_llm)


@pytest.mark.asyncio
async def test_pipeline_sequential_execution(registry: PromptRegistry, mock_llm: LLMProvider):
    """Pipeline executes steps sequentially, propagating state."""
    pipeline = _make_pipeline(registry, mock_llm, [
        {"name": "classify", "prompt": "classify", "output": "classification"},
        {"name": "extract", "prompt": "structure", "output": "fields"},
    ])
    result = await pipeline.execute("Deploy app")

    assert result.success is True
    assert result.pipeline_name == "test"
    assert len(result.steps_executed) == 2
    assert result.steps_executed[0].step_name == "classify"
    assert result.steps_executed[1].step_name == "extract"
    assert "classification" in result.state
    assert "fields" in result.state


@pytest.mark.asyncio
async def test_pipeline_conditional_step_skip(registry: PromptRegistry, mock_llm: LLMProvider):
    """Pipeline skips steps whose condition evaluates to False."""
    # Mock returns high confidence — condition "confidence < 0.5" is False
    mock_llm.complete_json = AsyncMock(return_value={
        "intent": "deploy", "confidence": 0.9, "domain": "devops"
    })
    pipeline = _make_pipeline(registry, mock_llm, [
        {"name": "classify", "prompt": "classify", "output": "classification"},
        {
            "name": "enrich_if_needed",
            "prompt": "enrich",
            "condition": "classification.get('confidence', 1.0) < 0.5",
            "output": "enriched",
        },
    ])
    result = await pipeline.execute("Deploy app")

    assert result.success is True
    assert result.steps_executed[1].skipped is True
    assert "enriched" not in result.state


@pytest.mark.asyncio
async def test_pipeline_conditional_step_executes(registry: PromptRegistry, mock_llm: LLMProvider):
    """Pipeline executes steps whose condition evaluates to True."""
    # Mock returns low confidence — condition is True
    mock_llm.complete_json = AsyncMock(return_value={
        "intent": "unknown", "confidence": 0.2, "domain": "general"
    })
    pipeline = _make_pipeline(registry, mock_llm, [
        {"name": "classify", "prompt": "classify", "output": "classification"},
        {
            "name": "enrich_if_needed",
            "prompt": "enrich",
            "condition": "classification.get('confidence', 1.0) < 0.5",
            "output": "enriched",
        },
    ])
    result = await pipeline.execute("something vague")

    assert result.success is True
    assert result.steps_executed[1].skipped is False
    assert "enriched" in result.state


@pytest.mark.asyncio
async def test_pipeline_algo_step_execution(registry: PromptRegistry, mock_llm: LLMProvider):
    """Pipeline executes algorithmic (non-LLM) steps."""
    pipeline = _make_pipeline(registry, mock_llm, [
        {"name": "classify", "prompt": "classify", "output": "classification"},
        {
            "name": "match_rule",
            "type": "domain_rule_matcher",
            "input": "classification",
            "output": "matched_rule",
        },
    ])
    result = await pipeline.execute("Deploy app")

    assert result.success is True
    assert result.steps_executed[1].step_type == "algo"
    assert "matched_rule" in result.state


@pytest.mark.asyncio
async def test_pipeline_llm_step_execution(registry: PromptRegistry, mock_llm: LLMProvider):
    """Pipeline LLM steps call complete_json on small_llm."""
    pipeline = _make_pipeline(registry, mock_llm, [
        {"name": "classify", "prompt": "classify", "output": "classification"},
    ])
    result = await pipeline.execute("Deploy app")

    assert result.success is True
    mock_llm.complete_json.assert_called_once()
    call_kwargs = mock_llm.complete_json.call_args
    assert "system_prompt" in call_kwargs.kwargs or len(call_kwargs.args) >= 2


@pytest.mark.asyncio
async def test_pipeline_state_propagation(registry: PromptRegistry, mock_llm: LLMProvider):
    """State from earlier steps is available to later steps."""
    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"intent": "deploy", "confidence": 0.9}
        return {"action": "deploy", "target": "production"}

    mock_llm.complete_json = AsyncMock(side_effect=side_effect)

    pipeline = _make_pipeline(registry, mock_llm, [
        {"name": "classify", "prompt": "classify", "output": "classification"},
        {"name": "extract", "prompt": "structure", "output": "fields"},
    ])
    result = await pipeline.execute("Deploy app")

    assert result.state["classification"] == {"intent": "deploy", "confidence": 0.9}
    assert result.state["fields"] == {"action": "deploy", "target": "production"}


@pytest.mark.asyncio
async def test_pipeline_from_yaml_config(
    sample_pipelines_yaml: Path, registry: PromptRegistry, mock_llm: LLMProvider
):
    """Pipeline loads correctly from YAML file."""
    pipeline = PromptPipeline.from_yaml(
        pipelines_path=sample_pipelines_yaml,
        pipeline_name="classify",
        registry=registry,
        small_llm=mock_llm,
    )
    assert pipeline.config.name == "classify"
    assert len(pipeline.config.steps) == 1
    assert pipeline.config.steps[0].prompt == "classify"

    result = await pipeline.execute("Test query")
    assert result.success is True


def test_pipeline_from_yaml_missing_pipeline(sample_pipelines_yaml: Path):
    """Loading a non-existent pipeline raises KeyError."""
    with pytest.raises(KeyError, match="nonexistent"):
        PromptPipeline.from_yaml(
            pipelines_path=sample_pipelines_yaml,
            pipeline_name="nonexistent",
        )


@pytest.mark.asyncio
async def test_pipeline_missing_prompt_error(mock_llm: LLMProvider):
    """Pipeline with a prompt not in registry fails gracefully."""
    empty_registry = PromptRegistry(prompts_path="/nonexistent/prompts.yaml")
    pipeline = _make_pipeline(empty_registry, mock_llm, [
        {"name": "classify", "prompt": "classify", "output": "classification"},
    ])
    result = await pipeline.execute("Test")

    assert result.success is False
    assert "classify" in (result.error or "")


@pytest.mark.asyncio
async def test_pipeline_custom_validator(registry: PromptRegistry, mock_llm: LLMProvider):
    """Pipeline supports custom algorithmic step handlers."""
    def custom_handler(inputs, state, config):
        return {"custom": True, "query_length": len(state.get("query", ""))}

    config = PipelineConfig(
        name="custom_test",
        steps=[
            PipelineStep(name="custom_step", type="custom_handler", output="custom_result"),
        ],
    )
    pipeline = PromptPipeline(config=config, registry=registry, small_llm=mock_llm)
    pipeline.register_algo_handler("custom_handler", custom_handler)

    result = await pipeline.execute("Hello world")
    assert result.success is True
    assert result.state["custom_result"] == {"custom": True, "query_length": 11}


@pytest.mark.asyncio
async def test_pipeline_empty_passthrough(registry: PromptRegistry, mock_llm: LLMProvider):
    """Empty pipeline (passthrough) returns state with just query and context."""
    pipeline = _make_pipeline(registry, mock_llm, [], name="passthrough")
    result = await pipeline.execute("Just pass me through", context={"env": "prod"})

    assert result.success is True
    assert result.state["query"] == "Just pass me through"
    assert result.state["context"] == {"env": "prod"}
    assert len(result.steps_executed) == 0
