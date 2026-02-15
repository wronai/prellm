"""Tests for preLLM v0.2 architecture — QueryDecomposer, LLMProvider, PreLLM, DomainRule."""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from prellm.models import (
    ClassificationResult,
    DecompositionPrompts,
    DecompositionResult,
    DecompositionStrategy,
    DomainRule,
    LLMProviderConfig,
    PreLLMConfig,
    PreLLMResponse,
    StructureResult,
    ProcessConfig,
    ProcessStep,
    ApprovalMode,
    StepStatus,
)
from prellm.llm_provider import LLMProvider
from prellm.query_decomposer import QueryDecomposer
from prellm.core import PreLLM


# ============================================================
# LLMProvider Tests
# ============================================================

class TestLLMProvider:
    def test_init_with_config(self):
        config = LLMProviderConfig(model="phi3:mini", max_retries=2, timeout=10)
        provider = LLMProvider(config)
        assert provider.config.model == "phi3:mini"
        assert provider.config.max_retries == 2

    @pytest.mark.asyncio
    async def test_complete_success(self):
        config = LLMProviderConfig(model="test-model")
        provider = LLMProvider(config)

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Test response"

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
            result = await provider.complete("Hello")

        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_complete_with_system_prompt(self):
        config = LLMProviderConfig(model="test-model")
        provider = LLMProvider(config)

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Classified"

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)) as mock_call:
            result = await provider.complete("Query", system_prompt="You are a classifier")

        assert result == "Classified"
        call_kwargs = mock_call.call_args[1]
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_complete_fallback(self):
        config = LLMProviderConfig(model="primary", fallback=["fallback1"], max_retries=1)
        provider = LLMProvider(config)

        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs["model"] == "primary":
                raise Exception("Primary failed")
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            mock_resp.choices[0].message.content = "Fallback response"
            return mock_resp

        with patch("litellm.acompletion", side_effect=mock_completion):
            result = await provider.complete("Hello")

        assert result == "Fallback response"
        assert call_count == 2  # 1 primary attempt + 1 fallback attempt

    @pytest.mark.asyncio
    async def test_complete_all_fail(self):
        config = LLMProviderConfig(model="primary", fallback=["fallback1"], max_retries=1)
        provider = LLMProvider(config)

        with patch("litellm.acompletion", new=AsyncMock(side_effect=Exception("All fail"))):
            with pytest.raises(RuntimeError, match="All models failed"):
                await provider.complete("Hello")

    @pytest.mark.asyncio
    async def test_complete_json_success(self):
        config = LLMProviderConfig(model="test-model")
        provider = LLMProvider(config)

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"intent": "deploy", "confidence": 0.95}'

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
            result = await provider.complete_json("Classify this")

        assert result["intent"] == "deploy"
        assert result["confidence"] == 0.95

    def test_parse_json_direct(self):
        result = LLMProvider._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_markdown_block(self):
        text = "Here is the result:\n```json\n{\"key\": \"value\"}\n```"
        result = LLMProvider._parse_json(text)
        assert result == {"key": "value"}

    def test_parse_json_embedded(self):
        text = "The answer is {\"key\": \"value\"} as shown."
        result = LLMProvider._parse_json(text)
        assert result == {"key": "value"}

    def test_parse_json_invalid(self):
        result = LLMProvider._parse_json("not json at all")
        assert result == {}


# ============================================================
# QueryDecomposer Tests
# ============================================================

class TestQueryDecomposer:
    def _make_decomposer(self, llm_response: str = '{"intent": "deploy", "confidence": 0.9, "domain": "devops"}'):
        config = LLMProviderConfig(model="test-small")
        provider = LLMProvider(config)
        decomposer = QueryDecomposer(
            small_llm=provider,
            prompts=DecompositionPrompts(),
            domain_rules=[
                DomainRule(
                    name="production_deploy",
                    keywords=["deploy", "zdeployuj", "push"],
                    intent="deploy",
                    required_fields=["environment_details", "version"],
                    severity="critical",
                    strategy=DecompositionStrategy.STRUCTURE,
                ),
                DomainRule(
                    name="monitoring",
                    keywords=["status", "health", "logs"],
                    intent="monitoring",
                    required_fields=[],
                    severity="low",
                    strategy=DecompositionStrategy.PASSTHROUGH,
                ),
            ],
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
        decomposer, provider = self._make_decomposer()

        classify_json = '{"intent": "deploy", "confidence": 0.95, "domain": "devops"}'
        composed_text = "Deploy application to production with version details."

        call_count = 0

        async def mock_complete(user_message, system_prompt="", response_format=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if response_format == "json":
                return classify_json
            return composed_text

        provider.complete = AsyncMock(side_effect=mock_complete)
        provider.complete_json = AsyncMock(return_value=json.loads(classify_json))

        result = await decomposer.decompose("Deploy app to prod", strategy=DecompositionStrategy.CLASSIFY)

        assert result.strategy == DecompositionStrategy.CLASSIFY
        assert result.classification is not None
        assert result.classification.intent == "deploy"
        assert result.classification.confidence == 0.95
        assert result.matched_rule == "production_deploy"
        assert "environment_details" in result.missing_fields

    @pytest.mark.asyncio
    async def test_structure_strategy(self):
        decomposer, provider = self._make_decomposer()

        classify_data = {"intent": "deploy", "confidence": 0.9, "domain": "devops"}
        structure_data = {"action": "deploy", "target": "app", "parameters": {"env": "prod"}}

        provider.complete_json = AsyncMock(side_effect=[classify_data, structure_data])
        provider.complete = AsyncMock(return_value="Composed prompt for deploy")

        result = await decomposer.decompose("Deploy app to prod", strategy=DecompositionStrategy.STRUCTURE)

        assert result.structure is not None
        assert result.structure.action == "deploy"
        assert result.structure.target == "app"

    @pytest.mark.asyncio
    async def test_split_strategy(self):
        decomposer, provider = self._make_decomposer()

        classify_data = {"intent": "general", "confidence": 0.5, "domain": "general"}
        split_data = {"sub_queries": ["Check status", "Deploy app", "Run tests"]}

        provider.complete_json = AsyncMock(side_effect=[classify_data, split_data])
        provider.complete = AsyncMock(return_value="Composed prompt")

        result = await decomposer.decompose(
            "Check status, deploy app, and run tests",
            strategy=DecompositionStrategy.SPLIT,
        )

        assert len(result.sub_queries) == 3
        assert "Check status" in result.sub_queries

    @pytest.mark.asyncio
    async def test_enrich_strategy(self):
        decomposer, provider = self._make_decomposer()

        classify_data = {"intent": "deploy", "confidence": 0.9, "domain": "devops"}

        provider.complete_json = AsyncMock(return_value=classify_data)
        provider.complete = AsyncMock(return_value="Enriched: Deploy app v2.0 to production with rollback plan.")

        result = await decomposer.decompose(
            "Deploy app to prod",
            strategy=DecompositionStrategy.ENRICH,
        )

        assert "Enriched" in result.composed_prompt

    def test_match_domain_rule_by_keyword(self):
        decomposer, _ = self._make_decomposer()
        classification = ClassificationResult(intent="unknown", confidence=0.5, domain="general")
        rule = decomposer._match_domain_rule("Deploy the app now", classification)
        assert rule is not None
        assert rule.name == "production_deploy"

    def test_match_domain_rule_by_intent(self):
        decomposer, _ = self._make_decomposer()
        classification = ClassificationResult(intent="deploy", confidence=0.9, domain="devops")
        rule = decomposer._match_domain_rule("Push new version", classification)
        assert rule is not None
        assert rule.name == "production_deploy"

    def test_no_domain_rule_match(self):
        decomposer, _ = self._make_decomposer()
        classification = ClassificationResult(intent="unknown", confidence=0.1, domain="general")
        rule = decomposer._match_domain_rule("What is the weather today?", classification)
        assert rule is None

    def test_find_missing_fields(self):
        rule = DomainRule(
            name="test",
            required_fields=["environment_details", "version", "cluster"],
        )
        missing = QueryDecomposer._find_missing_fields(
            "Deploy app version 2.0",
            rule,
            {"cluster": "k8s-prod"},
        )
        assert "environment_details" in missing
        assert "cluster" not in missing  # provided in context

    @pytest.mark.asyncio
    async def test_classify_failure_graceful(self):
        decomposer, provider = self._make_decomposer()
        provider.complete_json = AsyncMock(side_effect=Exception("LLM down"))
        provider.complete = AsyncMock(return_value="Fallback compose")

        result = await decomposer.decompose("Deploy app", strategy=DecompositionStrategy.CLASSIFY)
        assert result.classification is not None
        assert result.classification.intent == "unknown"


# ============================================================
# DomainRule Model Tests
# ============================================================

class TestDomainRule:
    def test_defaults(self):
        rule = DomainRule(name="test")
        assert rule.keywords == []
        assert rule.required_fields == []
        assert rule.severity == "medium"
        assert rule.strategy == DecompositionStrategy.CLASSIFY

    def test_full_rule(self):
        rule = DomainRule(
            name="production_deploy",
            keywords=["deploy", "push"],
            intent="deploy",
            required_fields=["env", "version"],
            enrich_template="Deploy {version} to {env}",
            severity="critical",
            strategy=DecompositionStrategy.STRUCTURE,
        )
        assert rule.name == "production_deploy"
        assert len(rule.keywords) == 2
        assert rule.strategy == DecompositionStrategy.STRUCTURE


# ============================================================
# PreLLMConfig Tests
# ============================================================

class TestPreLLMConfig:
    def test_defaults(self):
        config = PreLLMConfig()
        assert config.small_model.model == "phi3:mini"
        assert config.large_model.model == "gpt-4o-mini"
        assert config.default_strategy == DecompositionStrategy.CLASSIFY
        assert config.domain_rules == []

    def test_custom_config(self):
        config = PreLLMConfig(
            small_model=LLMProviderConfig(model="qwen2:1.5b"),
            large_model=LLMProviderConfig(model="gpt-4"),
            default_strategy=DecompositionStrategy.STRUCTURE,
            domain_rules=[DomainRule(name="test")],
        )
        assert config.small_model.model == "qwen2:1.5b"
        assert config.large_model.model == "gpt-4"
        assert len(config.domain_rules) == 1


# ============================================================
# PreLLM Core Tests
# ============================================================

class TestPreLLMCore:
    def test_init_default(self):
        engine = PreLLM()
        assert engine.config.small_model.model == "phi3:mini"
        assert engine.config.large_model.model == "gpt-4o-mini"

    def test_init_with_config(self):
        config = PreLLMConfig(
            small_model=LLMProviderConfig(model="test-small"),
            large_model=LLMProviderConfig(model="test-large"),
        )
        engine = PreLLM(config=config)
        assert engine.config.small_model.model == "test-small"

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        config = PreLLMConfig(
            small_model=LLMProviderConfig(model="test-small"),
            large_model=LLMProviderConfig(model="test-large"),
        )
        engine = PreLLM(config=config)

        # Mock small LLM (classify + compose)
        classify_resp = MagicMock()
        classify_resp.choices = [MagicMock()]
        classify_resp.choices[0].message.content = '{"intent": "deploy", "confidence": 0.9, "domain": "devops"}'

        compose_resp = MagicMock()
        compose_resp.choices = [MagicMock()]
        compose_resp.choices[0].message.content = "Composed: Deploy app to production"

        # Mock large LLM
        large_resp = MagicMock()
        large_resp.choices = [MagicMock()]
        large_resp.choices[0].message.content = "Deployment initiated successfully."

        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return classify_resp
            elif call_count == 2:
                return compose_resp
            else:
                return large_resp

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            result = await engine("Deploy app to prod")

        assert isinstance(result, PreLLMResponse)
        assert result.content == "Deployment initiated successfully."
        assert result.model_used == "test-large"
        assert result.small_model_used == "test-small"

    @pytest.mark.asyncio
    async def test_decompose_only(self):
        config = PreLLMConfig(
            small_model=LLMProviderConfig(model="test-small"),
        )
        engine = PreLLM(config=config)

        classify_resp = MagicMock()
        classify_resp.choices = [MagicMock()]
        classify_resp.choices[0].message.content = '{"intent": "monitoring", "confidence": 0.8, "domain": "devops"}'

        compose_resp = MagicMock()
        compose_resp.choices = [MagicMock()]
        compose_resp.choices[0].message.content = "Check system health"

        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return classify_resp
            return compose_resp

        with patch("litellm.acompletion", side_effect=mock_acompletion):
            result = await engine.decompose_only("Check health")

        assert result["strategy"] == "classify"
        assert result["original_query"] == "Check health"
        assert result["classification"]["intent"] == "monitoring"

    @pytest.mark.asyncio
    async def test_passthrough_strategy(self):
        config = PreLLMConfig(
            small_model=LLMProviderConfig(model="test-small"),
            large_model=LLMProviderConfig(model="test-large"),
        )
        engine = PreLLM(config=config)

        large_resp = MagicMock()
        large_resp.choices = [MagicMock()]
        large_resp.choices[0].message.content = "Direct response"

        with patch("litellm.acompletion", new=AsyncMock(return_value=large_resp)):
            result = await engine("Simple query", strategy=DecompositionStrategy.PASSTHROUGH)

        assert result.content == "Direct response"
        assert result.decomposition.strategy == DecompositionStrategy.PASSTHROUGH

    def test_audit_log_empty(self):
        engine = PreLLM()
        assert engine.get_audit_log() == []

    @pytest.mark.asyncio
    async def test_large_llm_failure(self):
        config = PreLLMConfig(
            small_model=LLMProviderConfig(model="test-small", max_retries=1),
            large_model=LLMProviderConfig(model="test-large", max_retries=1),
        )
        engine = PreLLM(config=config)

        # Small LLM works for passthrough (no calls needed)
        # Large LLM fails
        with patch("litellm.acompletion", new=AsyncMock(side_effect=RuntimeError("All models failed after retries. Last error: down"))):
            result = await engine("Query", strategy=DecompositionStrategy.PASSTHROUGH)

        assert result.content == "No response from any model."
        assert result.retries == 1


# ============================================================
# PreLLM Config Loading Tests
# ============================================================

class TestPreLLMConfigLoading:
    def test_load_yaml_config(self, tmp_path):
        import yaml
        config_data = {
            "small_model": {"model": "phi3:mini", "max_tokens": 256},
            "large_model": {"model": "gpt-4", "max_tokens": 4096},
            "default_strategy": "structure",
            "policy": "devops",
            "domain_rules": [
                {"name": "test_rule", "keywords": ["test"], "intent": "testing"},
            ],
            "context_sources": [{"env": ["HOME"]}],
        }
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        engine = PreLLM(config_path=config_path)
        assert engine.config.small_model.model == "phi3:mini"
        assert engine.config.large_model.model == "gpt-4"
        assert engine.config.default_strategy == DecompositionStrategy.STRUCTURE
        assert len(engine.config.domain_rules) == 1
        assert engine.config.domain_rules[0].name == "test_rule"

    def test_load_empty_yaml(self, tmp_path):
        config_path = tmp_path / "empty.yaml"
        with open(config_path, "w") as f:
            f.write("{}")

        engine = PreLLM(config_path=config_path)
        assert engine.config.small_model.model == "phi3:mini"
        assert engine.config.default_strategy == DecompositionStrategy.CLASSIFY

    def test_load_shipped_config(self):
        from pathlib import Path
        config_path = Path(__file__).parent.parent / "configs" / "prellm_config.yaml"
        if config_path.exists():
            engine = PreLLM(config_path=config_path)
            assert engine.config.small_model.model == "phi3:mini"
            assert len(engine.config.domain_rules) >= 6


# ============================================================
# ProcessChain v0.2 Integration Tests
# ============================================================

class TestProcessChainV2:
    @pytest.mark.asyncio
    async def test_chain_with_v2_engine_dry_run(self):
        from prellm.chains.process_chain import ProcessChain

        config = ProcessConfig(
            process="test-v2",
            steps=[
                ProcessStep(name="classify-step", prompt="Deploy app", approval=ApprovalMode.AUTO,
                            strategy=DecompositionStrategy.CLASSIFY),
                ProcessStep(name="passthrough-step", prompt="Check health", approval=ApprovalMode.AUTO,
                            strategy=DecompositionStrategy.PASSTHROUGH),
            ],
        )

        prellm_config = PreLLMConfig(
            small_model=LLMProviderConfig(model="test-small"),
            large_model=LLMProviderConfig(model="test-large"),
        )
        engine = PreLLM(config=prellm_config)

        # Mock the decompose_only method
        engine.decompose_only = AsyncMock(return_value={
            "strategy": "classify",
            "original_query": "Deploy app",
            "classification": {"intent": "deploy", "confidence": 0.9, "domain": "devops"},
            "structure": None,
            "sub_queries": [],
            "missing_fields": [],
            "matched_rule": None,
            "composed_prompt": "Deploy app",
        })

        chain = ProcessChain(config=config, engine=engine)
        result = await chain.execute(dry_run=True)

        assert result.completed is True
        assert len(result.steps) == 2
        assert all(s.status == StepStatus.COMPLETED for s in result.steps)

    @pytest.mark.asyncio
    async def test_chain_per_step_strategy(self):
        from prellm.chains.process_chain import ProcessChain

        config = ProcessConfig(
            process="multi-strategy",
            steps=[
                ProcessStep(name="s1", prompt="Classify this", approval=ApprovalMode.AUTO,
                            strategy=DecompositionStrategy.CLASSIFY),
                ProcessStep(name="s2", prompt="Structure this", approval=ApprovalMode.AUTO,
                            strategy=DecompositionStrategy.STRUCTURE),
            ],
        )

        prellm_config = PreLLMConfig(
            small_model=LLMProviderConfig(model="test-small"),
        )
        engine = PreLLM(config=prellm_config)

        strategies_used = []

        async def mock_decompose(query, strategy=None, extra_context=None):
            strategies_used.append(strategy)
            return {
                "strategy": strategy.value if strategy else "classify",
                "original_query": query,
                "classification": None,
                "structure": None,
                "sub_queries": [],
                "missing_fields": [],
                "matched_rule": None,
                "composed_prompt": query,
            }

        engine.decompose_only = mock_decompose

        chain = ProcessChain(config=config, engine=engine)
        result = await chain.execute(dry_run=True)

        assert result.completed is True
        assert strategies_used[0] == DecompositionStrategy.CLASSIFY
        assert strategies_used[1] == DecompositionStrategy.STRUCTURE


# ============================================================
# DecompositionResult Model Tests
# ============================================================

class TestDecompositionResult:
    def test_defaults(self):
        result = DecompositionResult()
        assert result.strategy == DecompositionStrategy.PASSTHROUGH
        assert result.original_query == ""
        assert result.classification is None
        assert result.structure is None
        assert result.sub_queries == []
        assert result.missing_fields == []

    def test_full_result(self):
        result = DecompositionResult(
            strategy=DecompositionStrategy.CLASSIFY,
            original_query="Deploy app",
            classification=ClassificationResult(intent="deploy", confidence=0.9, domain="devops"),
            structure=StructureResult(action="deploy", target="app"),
            missing_fields=["version"],
            matched_rule="production_deploy",
            composed_prompt="Deploy app with version details",
        )
        assert result.classification.intent == "deploy"
        assert result.structure.action == "deploy"
        assert "version" in result.missing_fields


class TestPreLLMResponse:
    def test_defaults(self):
        resp = PreLLMResponse(content="test")
        assert resp.clarified is False
        assert resp.needs_more_context is False
        assert resp.retries == 0
        assert resp.decomposition is None

    def test_with_decomposition(self):
        decomp = DecompositionResult(
            strategy=DecompositionStrategy.CLASSIFY,
            original_query="test",
        )
        resp = PreLLMResponse(
            content="response",
            decomposition=decomp,
            model_used="gpt-4",
            small_model_used="phi3:mini",
        )
        assert resp.decomposition.strategy == DecompositionStrategy.CLASSIFY
        assert resp.model_used == "gpt-4"
        assert resp.small_model_used == "phi3:mini"


class TestDecompositionStrategy:
    def test_all_strategies(self):
        assert DecompositionStrategy.CLASSIFY.value == "classify"
        assert DecompositionStrategy.STRUCTURE.value == "structure"
        assert DecompositionStrategy.SPLIT.value == "split"
        assert DecompositionStrategy.ENRICH.value == "enrich"
        assert DecompositionStrategy.PASSTHROUGH.value == "passthrough"

    def test_from_string(self):
        assert DecompositionStrategy("classify") == DecompositionStrategy.CLASSIFY
        assert DecompositionStrategy("passthrough") == DecompositionStrategy.PASSTHROUGH
