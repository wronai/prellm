"""Tests for LLMProvider and QueryDecomposer — the core small LLM pipeline."""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from prellm.llm_provider import LLMProvider
from prellm.query_decomposer import QueryDecomposer
from prellm.models import (
    ClassificationResult,
    DecompositionPrompts,
    DecompositionStrategy,
    DomainRule,
    LLMProviderConfig,
    PreLLMConfig,
)


# === LLMProvider Tests ===

class TestLLMProvider:
    @pytest.mark.asyncio
    async def test_complete_returns_content(self):
        provider = LLMProvider(LLMProviderConfig(model="test-model"))
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Hello world"

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
            content = await provider.complete("Say hello")

        assert content == "Hello world"

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(self):
        provider = LLMProvider(LLMProviderConfig(model="test-model"))
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"intent": "deploy"}'

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)) as mock_call:
            content = await provider.complete("Deploy app", system_prompt="Classify this")

        call_args = mock_call.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_complete_json_parses(self):
        provider = LLMProvider(LLMProviderConfig(model="test-model"))
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"intent": "deploy", "confidence": 0.95}'

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
            data = await provider.complete_json("Classify this")

        assert data["intent"] == "deploy"
        assert data["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_complete_json_handles_markdown_fence(self):
        provider = LLMProvider(LLMProviderConfig(model="test-model"))
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '```json\n{"intent": "scale"}\n```'

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
            data = await provider.complete_json("Classify")

        assert data["intent"] == "scale"

    @pytest.mark.asyncio
    async def test_complete_json_returns_empty_on_bad_json(self):
        provider = LLMProvider(LLMProviderConfig(model="test-model"))
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Not valid JSON at all"

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
            data = await provider.complete_json("Classify")

        assert data == {}

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        provider = LLMProvider(
            LLMProviderConfig(model="primary", fallback=["secondary"], max_retries=1)
        )
        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs["model"] == "primary":
                raise Exception("Primary failed")
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = "Fallback response"
            return resp

        with patch("litellm.acompletion", side_effect=mock_completion):
            content = await provider.complete("Test")

        assert content == "Fallback response"

    @pytest.mark.asyncio
    async def test_all_models_fail(self):
        provider = LLMProvider(
            LLMProviderConfig(model="m1", fallback=["m2"], max_retries=1)
        )

        with patch("litellm.acompletion", new=AsyncMock(side_effect=Exception("fail"))):
            with pytest.raises(RuntimeError, match="All models failed"):
                await provider.complete("Test")


# === QueryDecomposer Tests ===

class TestQueryDecomposer:
    def _make_decomposer(
        self,
        domain_rules: list[DomainRule] | None = None,
    ) -> tuple[QueryDecomposer, LLMProvider]:
        config = LLMProviderConfig(model="test-small")
        provider = LLMProvider(config)
        decomposer = QueryDecomposer(
            small_llm=provider,
            prompts=DecompositionPrompts(),
            domain_rules=domain_rules or [],
        )
        return decomposer, provider

    @pytest.mark.asyncio
    async def test_passthrough_strategy(self):
        decomposer, _ = self._make_decomposer()
        result = await decomposer.decompose("Hello world", strategy=DecompositionStrategy.PASSTHROUGH)
        assert result.strategy == DecompositionStrategy.PASSTHROUGH
        assert result.composed_prompt == "Hello world"
        assert result.classification is None

    @pytest.mark.asyncio
    async def test_classify_strategy(self):
        rules = [
            DomainRule(
                name="deploy", intent="deploy", keywords=["deploy"],
                required_fields=["environment"],
            )
        ]
        decomposer, provider = self._make_decomposer(domain_rules=rules)

        classify_data = {"intent": "deploy", "confidence": 0.9, "domain": "devops"}
        provider.complete_json = AsyncMock(return_value=classify_data)
        provider.complete = AsyncMock(return_value="Composed prompt for deploy")

        result = await decomposer.decompose("Deploy the app", strategy=DecompositionStrategy.CLASSIFY)
        assert result.classification is not None
        assert result.classification.intent == "deploy"
        assert result.classification.confidence == 0.9
        assert result.matched_rule == "deploy"

    @pytest.mark.asyncio
    async def test_structure_strategy(self):
        decomposer, provider = self._make_decomposer()

        classify_data = {"intent": "deploy", "confidence": 0.8, "domain": "devops"}
        structure_data = {"action": "deploy", "target": "app", "parameters": {"env": "staging"}}

        provider.complete_json = AsyncMock(side_effect=[classify_data, structure_data])
        provider.complete = AsyncMock(return_value="Deploy app to staging environment safely")

        result = await decomposer.decompose("Deploy app to staging", strategy=DecompositionStrategy.STRUCTURE)
        assert result.classification.intent == "deploy"
        assert result.structure is not None
        assert result.structure.action == "deploy"
        assert result.structure.target == "app"

    @pytest.mark.asyncio
    async def test_split_strategy(self):
        decomposer, provider = self._make_decomposer()

        classify_data = {"intent": "query", "confidence": 0.7, "domain": "general"}
        split_data = {"sub_queries": ["What is cluster status?", "Are there alerts?"]}

        provider.complete_json = AsyncMock(side_effect=[classify_data, split_data])
        provider.complete = AsyncMock(return_value="Composed")

        result = await decomposer.decompose(
            "Check everything on the cluster", strategy=DecompositionStrategy.SPLIT
        )
        assert len(result.sub_queries) == 2
        assert "cluster status" in result.sub_queries[0]

    @pytest.mark.asyncio
    async def test_enrich_strategy(self):
        rules = [
            DomainRule(
                name="deploy", keywords=["deploy"],
                intent="deploy",
                required_fields=["version"],
            )
        ]
        decomposer, provider = self._make_decomposer(domain_rules=rules)

        classify_data = {"intent": "deploy", "confidence": 0.8, "domain": "devops"}
        provider.complete_json = AsyncMock(return_value=classify_data)
        provider.complete = AsyncMock(
            return_value="Deploy app v2.1 to staging with rollback enabled. Missing: version."
        )

        result = await decomposer.decompose("Deploy app", strategy=DecompositionStrategy.ENRICH)
        assert "v2.1" in result.composed_prompt
        assert "version" in result.missing_fields

    @pytest.mark.asyncio
    async def test_missing_fields_from_domain_rules(self):
        rules = [
            DomainRule(
                name="deploy", keywords=["deploy"],
                required_fields=["environment", "version"],
            )
        ]
        decomposer, provider = self._make_decomposer(domain_rules=rules)

        classify_data = {"intent": "deploy", "confidence": 0.9, "domain": "devops"}
        structure_data = {"action": "deploy", "target": "app", "parameters": {}}

        provider.complete_json = AsyncMock(side_effect=[classify_data, structure_data])
        provider.complete = AsyncMock(return_value="Deploy app (missing env, version)")

        result = await decomposer.decompose("Deploy the app", strategy=DecompositionStrategy.STRUCTURE)
        assert "environment" in result.missing_fields
        assert "version" in result.missing_fields
