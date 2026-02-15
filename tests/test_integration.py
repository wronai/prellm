"""Integration tests — CLI dry-run, process chain dry-run, YAML config loading."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from prellm.chains.process_chain import ProcessChain
from prellm.core import prellm
from prellm.models import (
    ApprovalMode,
    GuardConfig,
    Policy,
    ProcessConfig,
    ProcessStep,
    StepStatus,
)


# === YAML Config Loading ===

class TestYAMLConfigLoading:
    def _write_yaml(self, data: dict) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(data, f)
        f.close()
        return f.name

    def test_load_minimal_config(self):
        path = self._write_yaml({"max_retries": 5, "policy": "lenient"})
        guard = prellm(config_path=path)
        assert guard.config.max_retries == 5
        assert guard.config.policy == Policy.LENIENT
        os.unlink(path)

    def test_load_full_config(self):
        path = self._write_yaml({
            "bias_patterns": [
                {"regex": "test_pattern", "action": "clarify", "severity": "high", "description": "Test"}
            ],
            "clarify_template": "Custom: {query}",
            "max_retries": 2,
            "policy": "devops",
            "models": {"fallback": ["llama3", "mistral"], "timeout": 60},
            "context_sources": [{"env": ["HOME"]}],
        })
        guard = prellm(config_path=path)
        assert guard.config.policy == Policy.DEVOPS
        assert len(guard.config.bias_patterns) == 1
        assert guard.config.models.fallback == ["llama3", "mistral"]
        assert guard.config.models.timeout == 60
        os.unlink(path)

    def test_load_empty_config(self):
        path = self._write_yaml({})
        guard = prellm(config_path=path)
        assert guard.config.max_retries == 3  # default
        os.unlink(path)

    def test_configs_rules_yaml(self):
        """Test that the shipped configs/rules.yaml loads correctly."""
        rules_path = Path(__file__).parent.parent / "configs" / "rules.yaml"
        if rules_path.exists():
            guard = prellm(config_path=rules_path)
            assert guard.config.policy == Policy.DEVOPS
            assert len(guard.config.bias_patterns) >= 8


# === Analyze-only (no LLM) ===

class TestAnalyzeOnly:
    def test_production_deploy_detected(self):
        guard = prellm(config=GuardConfig())
        result = guard.analyze_only("Deploy to production now please")
        assert result["needs_clarify"] is True

    def test_db_delete_detected(self):
        guard = prellm(config=GuardConfig())
        result = guard.analyze_only("Usuń bazę danych użytkowników")
        assert result["needs_clarify"] is True

    def test_safe_query_passes(self):
        guard = prellm(config=GuardConfig())
        result = guard.analyze_only(
            "Pokaż status health-checków dla klastra staging z ostatnich 24 godzin"
        )
        assert result["needs_clarify"] is False

    def test_short_devops_query_flagged(self):
        guard = prellm(config=GuardConfig())
        result = guard.analyze_only("restart serwer")
        assert result["needs_clarify"] is True

    def test_scale_operation_detected(self):
        guard = prellm(config=GuardConfig())
        result = guard.analyze_only("Scale up to 10 replicas in production cluster")
        assert result["needs_clarify"] is True

    def test_config_change_detected(self):
        guard = prellm(config=GuardConfig())
        result = guard.analyze_only("Change config for the main service")
        assert result["needs_clarify"] is True

    def test_polish_migration_detected(self):
        guard = prellm(config=GuardConfig())
        result = guard.analyze_only("Migruj bazę na production")
        assert result["needs_clarify"] is True


# === ProcessChain Dry-Run ===

class TestProcessChainDryRun:
    def _make_chain(self, steps: list[ProcessStep]) -> ProcessChain:
        config = ProcessConfig(process="test-process", steps=steps)
        guard = prellm(config=GuardConfig(bias_patterns=[]))
        return ProcessChain(config=config, guard=guard)

    @pytest.mark.asyncio
    async def test_simple_chain_dry_run(self):
        chain = self._make_chain([
            ProcessStep(name="step1", prompt="Check status", approval=ApprovalMode.AUTO),
            ProcessStep(name="step2", prompt="Deploy app", approval=ApprovalMode.AUTO),
        ])
        result = await chain.execute(dry_run=True)
        assert result.completed is True
        assert len(result.steps) == 2
        assert all(s.status == StepStatus.COMPLETED for s in result.steps)

    @pytest.mark.asyncio
    async def test_chain_with_manual_approval_pauses(self):
        chain = self._make_chain([
            ProcessStep(name="check", prompt="Pre-check", approval=ApprovalMode.AUTO),
            ProcessStep(name="deploy", prompt="Deploy to prod", approval=ApprovalMode.MANUAL),
            ProcessStep(name="verify", prompt="Post-verify", approval=ApprovalMode.AUTO),
        ])
        result = await chain.execute(dry_run=False)
        assert result.completed is False
        assert result.steps[0].status == StepStatus.COMPLETED
        assert result.steps[1].status == StepStatus.AWAITING_APPROVAL
        assert len(result.steps) == 2  # chain paused at step 2

    @pytest.mark.asyncio
    async def test_chain_with_approval_callback(self):
        async def auto_approve(step_name: str, prompt: str) -> tuple[bool, str]:
            return True, "ci-bot"

        chain = self._make_chain([
            ProcessStep(name="check", prompt="Pre-check", approval=ApprovalMode.AUTO),
            ProcessStep(name="deploy", prompt="Deploy to prod", approval=ApprovalMode.MANUAL),
        ])

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Deployed successfully"

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
            result = await chain.execute(approval_callback=auto_approve)

        assert result.completed is True
        assert result.steps[1].approved_by == "ci-bot"

    @pytest.mark.asyncio
    async def test_chain_dependency_failure(self):
        chain = self._make_chain([
            ProcessStep(name="deploy", prompt="Deploy", approval=ApprovalMode.AUTO, depends_on=["missing-step"]),
        ])
        result = await chain.execute(dry_run=True)
        assert result.steps[0].status == StepStatus.FAILED
        assert "missing-step" in result.steps[0].error

    @pytest.mark.asyncio
    async def test_chain_context_injection(self, monkeypatch):
        monkeypatch.setenv("CLUSTER", "k8s-test")
        config = ProcessConfig(
            process="ctx-test",
            context_sources=[{"env": ["CLUSTER"]}],
            steps=[ProcessStep(name="check", prompt="Check {CLUSTER}", approval=ApprovalMode.AUTO)],
        )
        guard = prellm(config=GuardConfig(bias_patterns=[]))
        chain = ProcessChain(config=config, guard=guard)
        result = await chain.execute(dry_run=True)
        assert result.completed is True

    @pytest.mark.asyncio
    async def test_chain_extra_context(self):
        chain = self._make_chain([
            ProcessStep(name="step", prompt="Deploy to {env}", approval=ApprovalMode.AUTO),
        ])
        result = await chain.execute(dry_run=True, extra_context={"env": "staging"})
        assert result.completed is True

    @pytest.mark.asyncio
    async def test_chain_timing(self):
        chain = self._make_chain([
            ProcessStep(name="s1", prompt="Step 1", approval=ApprovalMode.AUTO),
            ProcessStep(name="s2", prompt="Step 2", approval=ApprovalMode.AUTO),
        ])
        result = await chain.execute(dry_run=True)
        assert result.total_duration_seconds >= 0
        assert result.finished_at is not None

    @pytest.mark.asyncio
    async def test_chain_audit_trail(self):
        chain = self._make_chain([
            ProcessStep(name="s1", prompt="Do thing", approval=ApprovalMode.AUTO),
        ])

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Done"

        with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
            await chain.execute()

        audit = chain.get_audit_log()
        assert len(audit) >= 1
        assert audit[0]["action"] == "process_step"


# === Deploy YAML Config ===

class TestDeployConfig:
    def test_deploy_yaml_loads(self):
        deploy_path = Path(__file__).parent.parent / "configs" / "deploy.yaml"
        if deploy_path.exists():
            chain = ProcessChain(
                config_path=deploy_path,
                guard=prellm(config=GuardConfig(bias_patterns=[])),
            )
            assert chain.process_config.process == "deploy-production"
            assert len(chain.process_config.steps) == 6
            assert chain.process_config.steps[1].approval == ApprovalMode.MANUAL
            assert chain.process_config.steps[1].rollback is True
            assert "pre-check" in chain.process_config.steps[1].depends_on
