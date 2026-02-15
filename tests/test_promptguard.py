"""Tests for PromptGuard core, bias detection, and process chains."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from promptguard.models import (
    AnalysisResult,
    BiasPattern,
    GuardConfig,
    GuardResponse,
    ModelConfig,
    Policy,
    ProcessConfig,
    ProcessStep,
    StepStatus,
    ApprovalMode,
)
from promptguard.analyzers.bias_detector import BiasDetector
from promptguard.analyzers.context_engine import ContextEngine
from promptguard.core import PromptGuard


# === BiasDetector Tests ===

class TestBiasDetector:
    def test_detects_absolute_quantifier_pl(self):
        detector = BiasDetector()
        result = detector.analyze("Zawsze używaj tego testu")
        assert result.needs_clarify is True
        assert any("Absolute quantifier" in p for p in result.detected_patterns)

    def test_detects_absolute_quantifier_en(self):
        detector = BiasDetector()
        result = detector.analyze("Always run all tests")
        assert result.needs_clarify is True

    def test_detects_production_deploy(self):
        detector = BiasDetector()
        result = detector.analyze("Deploy to production now")
        assert result.needs_clarify is True
        assert any("Production deployment" in p for p in result.detected_patterns)

    def test_detects_destructive_db_operation(self):
        detector = BiasDetector()
        result = detector.analyze("Usuń bazę danych klientów")
        assert result.needs_clarify is True
        assert any("Destructive DB" in p for p in result.detected_patterns)

    def test_detects_short_query(self):
        detector = BiasDetector()
        result = detector.analyze("Deploy app")
        assert result.needs_clarify is True
        assert "Query too short" in result.ambiguity_flags[0]

    def test_detects_devops_without_target(self):
        detector = BiasDetector()
        result = detector.analyze("Zdeployuj nową wersję aplikacji")
        assert result.needs_clarify is True
        assert any("without target" in f for f in result.ambiguity_flags)

    def test_clean_query_passes(self):
        detector = BiasDetector()
        result = detector.analyze("Proszę wyświetl listę użytkowników z bazy staging z ostatniego tygodnia")
        assert result.needs_clarify is False

    def test_custom_patterns(self):
        patterns = [BiasPattern(regex=r"custom_word", action="clarify", severity="low")]
        detector = BiasDetector(patterns=patterns)
        result = detector.analyze("Check custom_word status")
        assert result.needs_clarify is True

    def test_no_patterns_match(self):
        detector = BiasDetector(patterns=[])
        result = detector.analyze("This is a perfectly normal and detailed query about checking the server status on staging")
        assert result.needs_clarify is False


# === ContextEngine Tests ===

class TestContextEngine:
    def test_env_context(self, monkeypatch):
        monkeypatch.setenv("CLUSTER", "k8s-prod")
        monkeypatch.setenv("NAMESPACE", "backend")
        engine = ContextEngine([{"env": ["CLUSTER", "NAMESPACE"]}])
        ctx = engine.gather()
        assert ctx["CLUSTER"] == "k8s-prod"
        assert ctx["NAMESPACE"] == "backend"

    def test_enrich_prompt(self, monkeypatch):
        monkeypatch.setenv("CLUSTER", "k8s-prod")
        engine = ContextEngine([{"env": ["CLUSTER"]}])
        result = engine.enrich_prompt("Deploy to {CLUSTER}")
        assert result == "Deploy to k8s-prod"

    def test_extra_context_override(self, monkeypatch):
        monkeypatch.setenv("CLUSTER", "k8s-prod")
        engine = ContextEngine([{"env": ["CLUSTER"]}])
        result = engine.enrich_prompt("Deploy to {CLUSTER}", extra={"CLUSTER": "k8s-staging"})
        assert result == "Deploy to k8s-staging"

    def test_missing_env_graceful(self):
        engine = ContextEngine([{"env": ["NONEXISTENT_VAR_123"]}])
        ctx = engine.gather()
        assert "NONEXISTENT_VAR_123" not in ctx

    def test_system_context(self):
        engine = ContextEngine([{"system": ["hostname", "os", "python"]}])
        ctx = engine.gather()
        assert "hostname" in ctx
        assert "os" in ctx
        assert "python" in ctx


# === PromptGuard Core Tests ===

class TestPromptGuardCore:
    def test_load_inline_config(self):
        config = GuardConfig(policy=Policy.DEVOPS, max_retries=5)
        guard = PromptGuard(config=config)
        assert guard.config.policy == Policy.DEVOPS
        assert guard.config.max_retries == 5

    def test_analyze_only(self):
        guard = PromptGuard(config=GuardConfig())
        result = guard.analyze_only("Deploy to production immediately")
        assert result["needs_clarify"] is True

    def test_analyze_only_clean(self):
        guard = PromptGuard(config=GuardConfig(bias_patterns=[]))
        result = guard.analyze_only("Show me the detailed server metrics from staging environment for last month")
        assert result["needs_clarify"] is False

    @pytest.mark.asyncio
    async def test_call_with_mock_litellm(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Mocked LLM response"

        guard = PromptGuard(config=GuardConfig(bias_patterns=[]))

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
            result = await guard("Show server status on staging for all services", model="gpt-4o-mini")

        assert result.content == "Mocked LLM response"
        assert result.clarified is False

    @pytest.mark.asyncio
    async def test_call_with_clarification(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Clarified response"

        guard = PromptGuard(config=GuardConfig())

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
            result = await guard("Deploy to production", model="gpt-4o-mini")

        assert result.clarified is True
        assert result.content == "Clarified response"

    def test_audit_log(self):
        guard = PromptGuard(config=GuardConfig())
        assert guard.get_audit_log() == []


# === ProcessChain Tests ===

class TestProcessChainConfig:
    def test_process_config_model(self):
        config = ProcessConfig(
            process="test-deploy",
            steps=[
                ProcessStep(name="check", prompt="Check readiness", approval=ApprovalMode.AUTO),
                ProcessStep(name="deploy", prompt="Deploy app", approval=ApprovalMode.MANUAL, rollback=True),
            ],
        )
        assert len(config.steps) == 2
        assert config.steps[1].rollback is True

    def test_step_dependencies(self):
        step = ProcessStep(name="deploy", prompt="Deploy", depends_on=["check", "migrate"])
        assert "check" in step.depends_on


# === Model Validation Tests ===

class TestModels:
    def test_guard_response_defaults(self):
        resp = GuardResponse(content="test")
        assert resp.clarified is False
        assert resp.needs_more_context is False
        assert resp.retries == 0

    def test_analysis_result(self):
        ar = AnalysisResult(
            needs_clarify=True,
            detected_patterns=["bias"],
            original_query="test",
        )
        assert ar.needs_clarify is True

    def test_config_defaults(self):
        config = GuardConfig()
        assert config.policy == Policy.STRICT
        assert config.max_retries == 3
