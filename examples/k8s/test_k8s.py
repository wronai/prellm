"""Tests for K8s debugging example — fully mocked, no LLM needed."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from prellm import preprocess_and_execute
from prellm.models import PreLLMResponse


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


CONFIG_PATH = str(Path(__file__).parent.parent.parent / "configs" / "domains" / "devops_k8s.yaml")


class TestK8sCrashLoopBackOff:
    """Test CrashLoopBackOff diagnosis scenario."""

    @pytest.mark.asyncio
    async def test_crashloop_returns_response(self):
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_response(
                "The CrashLoopBackOff is likely caused by missing environment variables. "
                "Check: kubectl describe pod backend-api -n production"
            )
        )):
            result = await preprocess_and_execute(
                query="Pod backend-api w namespace production restartuje sie z CrashLoopBackOff",
                config_path=CONFIG_PATH,
                strategy="structure",
            )

        assert isinstance(result, PreLLMResponse)
        assert "CrashLoopBackOff" in result.content
        assert result.decomposition is not None

    @pytest.mark.asyncio
    async def test_crashloop_uses_correct_models(self):
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_response("Diagnosis complete")
        )):
            result = await preprocess_and_execute(
                query="Pod crashes with CrashLoopBackOff in production",
                config_path=CONFIG_PATH,
                strategy="structure",
            )

        assert result.small_model_used == "ollama/qwen2.5:3b"


class TestK8sOOMDiagnosis:
    """Test OOM diagnosis with user context."""

    @pytest.mark.asyncio
    async def test_oom_with_context(self):
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_response(
                "OOM detected. With 4GB node RAM, increase memory limits or add nodes."
            )
        )):
            result = await preprocess_and_execute(
                query="Kubernetes pods killed by OOM on RPi cluster",
                config_path=CONFIG_PATH,
                strategy="enrich",
                user_context={
                    "cluster": "rpi-k3s-prod",
                    "namespace": "backend",
                    "node_ram": "4GB",
                },
            )

        assert isinstance(result, PreLLMResponse)
        assert "OOM" in result.content


class TestK8sScaling:
    """Test HPA scaling scenario."""

    @pytest.mark.asyncio
    async def test_hpa_scaling(self):
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_response(
                "apiVersion: autoscaling/v2\nkind: HorizontalPodAutoscaler\n"
                "spec:\n  minReplicas: 2\n  maxReplicas: 10"
            )
        )):
            result = await preprocess_and_execute(
                query="Skonfiguruj autoscaling dla deployment frontend z min 2 max 10 replik",
                config_path=CONFIG_PATH,
            )

        assert "autoscaling" in result.content.lower() or "Replicas" in result.content


class TestK8sConfigLoading:
    """Test that the K8s domain config loads correctly."""

    def test_config_file_exists(self):
        assert Path(CONFIG_PATH).is_file(), f"Config not found: {CONFIG_PATH}"

    def test_config_is_valid_yaml(self):
        import yaml
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        assert "domain_rules" in data
        assert "small_model" in data
        assert "large_model" in data
        assert len(data["domain_rules"]) >= 3

    def test_config_has_k8s_rules(self):
        import yaml
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        rule_names = [r["name"] for r in data["domain_rules"]]
        assert "k8s_debug" in rule_names
        assert "k8s_scale" in rule_names
