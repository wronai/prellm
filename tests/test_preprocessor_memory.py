"""Tests for PreprocessorAgent integration with UserMemory and CodebaseIndexer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from prellm.agents.preprocessor import PreprocessorAgent, PreprocessResult
from prellm.llm_provider import LLMProvider
from prellm.models import LLMProviderConfig
from prellm.pipeline import PipelineConfig, PipelineStep, PromptPipeline
from prellm.prompt_registry import PromptRegistry


@pytest.fixture
def tmp_prompts(tmp_path: Path) -> Path:
    data = {
        "prompts": {
            "classify": {"system": "Classify. JSON.", "max_tokens": 256, "temperature": 0.1},
            "compose": {"system": "Compose.", "max_tokens": 512, "temperature": 0.2},
        }
    }
    path = tmp_path / "prompts.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


@pytest.fixture
def mock_small_llm() -> LLMProvider:
    provider = MagicMock(spec=LLMProvider)
    provider.config = LLMProviderConfig(model="test:small", max_tokens=512)
    provider.complete_json = AsyncMock(return_value={"intent": "deploy", "confidence": 0.9})
    provider.complete = AsyncMock(return_value="composed prompt")
    return provider


@pytest.fixture
def simple_pipeline(tmp_prompts, mock_small_llm) -> PromptPipeline:
    registry = PromptRegistry(prompts_path=tmp_prompts)
    config = PipelineConfig(
        name="classify",
        steps=[PipelineStep(name="classify", prompt="classify", output="classification")],
    )
    return PromptPipeline(config=config, registry=registry, small_llm=mock_small_llm)


@pytest.mark.asyncio
async def test_preprocessor_with_user_memory(tmp_prompts, mock_small_llm, simple_pipeline):
    """PreprocessorAgent enriches context with UserMemory data."""
    mock_memory = MagicMock()
    mock_memory.get_recent_context = AsyncMock(return_value=[
        {"query": "Deploy v1", "response_summary": "Deployed to staging", "metadata": {}, "timestamp": 1.0},
    ])
    mock_memory.get_user_preferences = AsyncMock(return_value={"env": "staging"})

    agent = PreprocessorAgent(
        small_llm=mock_small_llm,
        registry=PromptRegistry(prompts_path=tmp_prompts),
        pipeline=simple_pipeline,
        user_memory=mock_memory,
    )

    result = await agent.preprocess("Deploy v2")
    assert isinstance(result, PreprocessResult)
    assert "user_history" in result.context_used
    assert "user_preferences" in result.context_used
    assert "Deploy v1" in result.context_used["user_history"]
    mock_memory.get_recent_context.assert_called_once()
    mock_memory.get_user_preferences.assert_called_once()


@pytest.mark.asyncio
async def test_preprocessor_with_codebase_indexer(tmp_path, tmp_prompts, mock_small_llm, simple_pipeline):
    """PreprocessorAgent enriches context with CodebaseIndexer data."""
    # Create a simple Python file to index
    src = tmp_path / "src"
    src.mkdir()
    (src / "app.py").write_text("def deploy_app():\n    pass\n")

    from prellm.context.codebase_indexer import CodebaseIndexer
    indexer = CodebaseIndexer()

    agent = PreprocessorAgent(
        small_llm=mock_small_llm,
        registry=PromptRegistry(prompts_path=tmp_prompts),
        pipeline=simple_pipeline,
        codebase_indexer=indexer,
        codebase_path=str(src),
    )

    result = await agent.preprocess("deploy")
    assert isinstance(result, PreprocessResult)
    assert "codebase_context" in result.context_used
    assert "deploy_app" in result.context_used["codebase_context"]


@pytest.mark.asyncio
async def test_preprocessor_without_memory_or_indexer(tmp_prompts, mock_small_llm, simple_pipeline):
    """PreprocessorAgent works fine without UserMemory or CodebaseIndexer."""
    agent = PreprocessorAgent(
        small_llm=mock_small_llm,
        registry=PromptRegistry(prompts_path=tmp_prompts),
        pipeline=simple_pipeline,
    )

    result = await agent.preprocess("Deploy app")
    assert isinstance(result, PreprocessResult)
    assert result.executor_input != ""
    assert "user_history" not in result.context_used
    assert "codebase_context" not in result.context_used


@pytest.mark.asyncio
async def test_preprocessor_memory_failure_graceful(tmp_prompts, mock_small_llm, simple_pipeline):
    """PreprocessorAgent handles UserMemory failures gracefully."""
    mock_memory = MagicMock()
    mock_memory.get_recent_context = AsyncMock(side_effect=Exception("DB error"))
    mock_memory.get_user_preferences = AsyncMock(side_effect=Exception("DB error"))

    agent = PreprocessorAgent(
        small_llm=mock_small_llm,
        registry=PromptRegistry(prompts_path=tmp_prompts),
        pipeline=simple_pipeline,
        user_memory=mock_memory,
    )

    # Should not raise — graceful degradation
    result = await agent.preprocess("Deploy app")
    assert isinstance(result, PreprocessResult)
    assert result.executor_input != ""


@pytest.mark.asyncio
async def test_preprocessor_empty_memory(tmp_prompts, mock_small_llm, simple_pipeline):
    """PreprocessorAgent handles empty UserMemory results."""
    mock_memory = MagicMock()
    mock_memory.get_recent_context = AsyncMock(return_value=[])
    mock_memory.get_user_preferences = AsyncMock(return_value={})

    agent = PreprocessorAgent(
        small_llm=mock_small_llm,
        registry=PromptRegistry(prompts_path=tmp_prompts),
        pipeline=simple_pipeline,
        user_memory=mock_memory,
    )

    result = await agent.preprocess("Deploy app")
    assert isinstance(result, PreprocessResult)
    # Empty results should not add keys
    assert "user_history" not in result.context_used
    assert "user_preferences" not in result.context_used
