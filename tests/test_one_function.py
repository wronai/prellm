"""Tests for preprocess_and_execute() — the 1-function API like litellm.completion()."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from prellm.core import preprocess_and_execute, preprocess_and_execute_sync
from prellm.models import (
    DecompositionStrategy,
    DomainRule,
    PreLLMResponse,
)


def _mock_litellm_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


class TestPreprocessAndExecute:
    """Core tests for the 1-function API."""

    @pytest.mark.asyncio
    async def test_zero_config(self):
        """Just query — everything else defaults."""
        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # small LLM: classify + compose
                return _mock_litellm_response('{"intent": "general", "confidence": 0.5, "domain": "general"}')
            return _mock_litellm_response("Here is the refactored code.")

        with patch("litellm.acompletion", side_effect=mock_completion):
            result = await preprocess_and_execute("Refaktoryzuj kod")

        assert isinstance(result, PreLLMResponse)
        assert result.content == "Here is the refactored code."
        assert result.small_model_used == "ollama/qwen2.5:3b"

    @pytest.mark.asyncio
    async def test_custom_models(self):
        """Specify both small and large models."""
        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _mock_litellm_response('{"intent": "deploy", "confidence": 0.9, "domain": "devops"}')
            return _mock_litellm_response("Deployed successfully.")

        with patch("litellm.acompletion", side_effect=mock_completion):
            result = await preprocess_and_execute(
                query="Deploy app to production",
                small_llm="phi3:mini",
                large_llm="gpt-4o-mini",
            )

        assert result.content == "Deployed successfully."
        assert result.small_model_used == "phi3:mini"
        assert result.model_used == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_passthrough_strategy(self):
        """Passthrough skips small LLM entirely."""
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_litellm_response("Direct response")
        )):
            result = await preprocess_and_execute(
                query="Simple question",
                strategy="passthrough",
            )

        assert result.content == "Direct response"
        assert result.decomposition is not None
        assert result.decomposition.strategy == DecompositionStrategy.PASSTHROUGH

    @pytest.mark.asyncio
    async def test_strategy_as_enum(self):
        """Strategy can be passed as enum."""
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_litellm_response("Response")
        )):
            result = await preprocess_and_execute(
                query="Test",
                strategy=DecompositionStrategy.PASSTHROUGH,
            )

        assert result.decomposition.strategy == DecompositionStrategy.PASSTHROUGH

    @pytest.mark.asyncio
    async def test_user_context_string(self):
        """User context as a string tag gets injected."""
        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _mock_litellm_response('{"intent": "refactor", "confidence": 0.8, "domain": "development"}')
            return _mock_litellm_response("Context-aware response")

        with patch("litellm.acompletion", side_effect=mock_completion):
            result = await preprocess_and_execute(
                query="Popraw kod",
                user_context="gdansk_embedded_python",
            )

        assert result.content == "Context-aware response"

    @pytest.mark.asyncio
    async def test_user_context_dict(self):
        """User context as a dict gets injected."""
        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _mock_litellm_response('{"intent": "diagnose", "confidence": 0.85, "domain": "devops"}')
            return _mock_litellm_response("K8s diagnosis complete")

        with patch("litellm.acompletion", side_effect=mock_completion):
            result = await preprocess_and_execute(
                query="Zdiagnozuj problem z K8s podami",
                user_context={"cluster": "k8s-prod", "namespace": "backend"},
                strategy="enrich",
            )

        assert result.content == "K8s diagnosis complete"

    @pytest.mark.asyncio
    async def test_with_domain_rules(self):
        """Inline domain rules get matched."""
        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _mock_litellm_response('{"intent": "deploy", "confidence": 0.95, "domain": "devops"}')
            return _mock_litellm_response("Deploy with safety checks")

        with patch("litellm.acompletion", side_effect=mock_completion):
            result = await preprocess_and_execute(
                query="Deploy app to production",
                domain_rules=[{
                    "name": "production_deploy",
                    "keywords": ["deploy", "push"],
                    "intent": "deploy",
                    "required_fields": ["environment", "version"],
                    "severity": "critical",
                }],
            )

        assert result.content == "Deploy with safety checks"
        assert result.decomposition is not None
        assert result.decomposition.matched_rule == "production_deploy"
        assert "environment" in result.decomposition.missing_fields

    @pytest.mark.asyncio
    async def test_with_config_path(self, tmp_path):
        """Load config from YAML file."""
        import yaml

        config_data = {
            "small_model": {"model": "test-small", "max_tokens": 256},
            "large_model": {"model": "test-large", "max_tokens": 1024},
            "default_strategy": "classify",
        }
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _mock_litellm_response('{"intent": "test", "confidence": 0.7, "domain": "general"}')
            return _mock_litellm_response("Config-loaded response")

        with patch("litellm.acompletion", side_effect=mock_completion):
            result = await preprocess_and_execute(
                query="Test query",
                config_path=str(config_file),
            )

        assert result.content == "Config-loaded response"
        assert result.small_model_used == "test-small"
        assert result.model_used == "test-large"

    @pytest.mark.asyncio
    async def test_config_path_with_model_override(self, tmp_path):
        """Config from file but override models."""
        import yaml

        config_data = {
            "small_model": {"model": "config-small"},
            "large_model": {"model": "config-large"},
        }
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _mock_litellm_response('{"intent": "test", "confidence": 0.5, "domain": "general"}')
            return _mock_litellm_response("Overridden response")

        with patch("litellm.acompletion", side_effect=mock_completion):
            result = await preprocess_and_execute(
                query="Test",
                small_llm="override-small",
                large_llm="override-large",
                config_path=str(config_file),
            )

        assert result.small_model_used == "override-small"
        assert result.model_used == "override-large"

    @pytest.mark.asyncio
    async def test_large_llm_failure(self):
        """Large LLM failure returns fallback response."""
        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return _mock_litellm_response('{"intent": "test", "confidence": 0.5, "domain": "general"}')
            raise RuntimeError("All models failed after retries. Last error: connection refused")

        with patch("litellm.acompletion", side_effect=mock_completion):
            result = await preprocess_and_execute(
                query="Test query",
                small_llm="test-small",
                large_llm="test-large",
            )

        assert result.content == "No response from any model."
        assert result.retries == 1

    @pytest.mark.asyncio
    async def test_structure_strategy(self):
        """Structure strategy extracts fields."""
        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_litellm_response('{"intent": "deploy", "confidence": 0.9, "domain": "devops"}')
            elif call_count == 2:
                return _mock_litellm_response('{"action": "deploy", "target": "app", "parameters": {"env": "prod"}}')
            elif call_count == 3:
                return _mock_litellm_response("Composed: Deploy app to prod safely")
            else:
                return _mock_litellm_response("Structured deployment complete")

        with patch("litellm.acompletion", side_effect=mock_completion):
            result = await preprocess_and_execute(
                query="Deploy app to prod",
                strategy="structure",
            )

        assert result.content == "Structured deployment complete"
        assert result.decomposition is not None
        assert result.decomposition.structure is not None
        assert result.decomposition.structure.action == "deploy"

    @pytest.mark.asyncio
    async def test_split_strategy(self):
        """Split strategy breaks query into sub-queries."""
        call_count = 0

        async def mock_completion(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_litellm_response('{"intent": "multi", "confidence": 0.7, "domain": "general"}')
            elif call_count == 2:
                return _mock_litellm_response('{"sub_queries": ["Check status", "Deploy app"]}')
            elif call_count == 3:
                return _mock_litellm_response("Composed multi-task prompt")
            else:
                return _mock_litellm_response("Both tasks completed")

        with patch("litellm.acompletion", side_effect=mock_completion):
            result = await preprocess_and_execute(
                query="Check status and deploy app",
                strategy="split",
            )

        assert result.content == "Both tasks completed"
        assert len(result.decomposition.sub_queries) == 2

    @pytest.mark.asyncio
    async def test_returns_prellm_response(self):
        """Result is always a PreLLMResponse."""
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_litellm_response("OK")
        )):
            result = await preprocess_and_execute("Test", strategy="passthrough")

        assert isinstance(result, PreLLMResponse)
        assert hasattr(result, "content")
        assert hasattr(result, "decomposition")
        assert hasattr(result, "model_used")
        assert hasattr(result, "small_model_used")
        assert hasattr(result, "retries")
        assert hasattr(result, "timestamp")

    @pytest.mark.asyncio
    async def test_kwargs_forwarded(self):
        """Extra kwargs like max_tokens are forwarded to config."""
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_litellm_response("OK")
        )):
            result = await preprocess_and_execute(
                query="Test",
                strategy="passthrough",
                max_tokens=4096,
                temperature=0.9,
            )

        assert result.content == "OK"


class TestPreprocessAndExecuteSync:
    """Tests for the synchronous wrapper."""

    def test_sync_wrapper_works(self):
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_litellm_response("Sync response")
        )):
            result = preprocess_and_execute_sync(
                query="Test",
                strategy="passthrough",
            )

        assert isinstance(result, PreLLMResponse)
        assert result.content == "Sync response"


class TestImportFromPackage:
    """Test that the 1-function API is importable from the package root."""

    def test_import_async(self):
        from prellm import preprocess_and_execute as fn
        assert callable(fn)

    def test_import_sync(self):
        from prellm import preprocess_and_execute_sync as fn
        assert callable(fn)

    def test_version(self):
        import prellm
        assert prellm.__version__ == "0.3.0"

    def test_all_exports(self):
        import prellm
        assert "preprocess_and_execute" in prellm.__all__
        assert "preprocess_and_execute_sync" in prellm.__all__
        assert "PreLLM" in prellm.__all__
        assert "PreLLMResponse" in prellm.__all__
