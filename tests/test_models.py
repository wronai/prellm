"""Tests for preLLM v0.2 models, config loading, and validation."""

from __future__ import annotations

import os
import tempfile
import pytest
import yaml

from prellm.models import (
    AuditEntry,
    ClassificationResult,
    DecompositionResult,
    DecompositionStrategy,
    DomainRule,
    LLMProviderConfig,
    PreLLMConfig,
    PreLLMResponse,
    StructureResult,
    Policy,
    ProcessConfig,
    ProcessStep,
    ApprovalMode,
    StepStatus,
)
from prellm.core import PreLLM


class TestModelDefaults:
    def test_llm_provider_config_defaults(self):
        cfg = LLMProviderConfig()
        assert cfg.model == "gpt-4o-mini"
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 2048

    def test_prellm_config_defaults(self):
        cfg = PreLLMConfig()
        assert cfg.default_strategy == DecompositionStrategy.CLASSIFY
        assert cfg.policy == Policy.STRICT
        assert cfg.max_retries == 3
        assert cfg.domain_rules == []
        assert cfg.small_model.model == "phi3:mini"
        assert cfg.large_model.model == "gpt-4o-mini"

    def test_response_defaults(self):
        resp = PreLLMResponse(content="test")
        assert resp.clarified is False
        assert resp.needs_more_context is False
        assert resp.decomposition is None

    def test_decomposition_result_defaults(self):
        dr = DecompositionResult()
        assert dr.strategy == DecompositionStrategy.PASSTHROUGH
        assert dr.sub_queries == []
        assert dr.missing_fields == []
        assert dr.classification is None
        assert dr.structure is None

    def test_classification_result(self):
        cr = ClassificationResult(intent="deploy", confidence=0.95, domain="devops")
        assert cr.intent == "deploy"
        assert cr.confidence == 0.95

    def test_structure_result(self):
        sr = StructureResult(action="deploy", target="app", parameters={"env": "prod"})
        assert sr.action == "deploy"
        assert sr.parameters["env"] == "prod"

    def test_audit_entry(self):
        entry = AuditEntry(
            action="query", model="gpt-4o-mini",
            metadata={"small_model": "phi3:mini", "strategy": "classify"},
        )
        assert entry.metadata["small_model"] == "phi3:mini"
        assert entry.metadata["strategy"] == "classify"


class TestDomainRules:
    def test_rule_with_all_fields(self):
        rule = DomainRule(
            name="prod_deploy",
            keywords=["deploy", "production"],
            intent="deploy",
            required_fields=["target", "environment", "version"],
            severity="critical",
            enrich_template="Enrich: {query}",
        )
        assert rule.severity == "critical"
        assert len(rule.required_fields) == 3

    def test_rule_keyword_matching(self):
        rule = DomainRule(name="test", keywords=["deploy", "push"])
        query = "deploy the app"
        assert any(kw in query.lower() for kw in rule.keywords)

    def test_rule_no_match(self):
        rule = DomainRule(name="test", keywords=["deploy"])
        query = "show me logs"
        assert not any(kw in query.lower() for kw in rule.keywords)


class TestDecompositionStrategy:
    def test_all_strategies_exist(self):
        strategies = [s.value for s in DecompositionStrategy]
        assert "classify" in strategies
        assert "split" in strategies
        assert "structure" in strategies
        assert "enrich" in strategies
        assert "passthrough" in strategies

    def test_strategy_from_string(self):
        assert DecompositionStrategy("structure") == DecompositionStrategy.STRUCTURE
        assert DecompositionStrategy("passthrough") == DecompositionStrategy.PASSTHROUGH


class TestConfigLoading:
    def _write_yaml(self, data: dict) -> str:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(data, f)
        f.close()
        return f.name

    def test_load_minimal_config(self):
        path = self._write_yaml({"max_retries": 5, "policy": "lenient"})
        pre = PreLLM(config_path=path)
        assert pre.config.max_retries == 5
        assert pre.config.policy == Policy.LENIENT
        os.unlink(path)

    def test_load_with_models(self):
        path = self._write_yaml({
            "small_model": {"model": "ollama/qwen2:1.5b", "temperature": 0.05},
            "large_model": {"model": "anthropic/claude-sonnet-4-20250514", "max_tokens": 8192},
        })
        pre = PreLLM(config_path=path)
        assert pre.config.small_model.model == "ollama/qwen2:1.5b"
        assert pre.config.small_model.temperature == 0.05
        assert pre.config.large_model.model == "anthropic/claude-sonnet-4-20250514"
        assert pre.config.large_model.max_tokens == 8192
        os.unlink(path)

    def test_load_with_domain_rules(self):
        path = self._write_yaml({
            "domain_rules": [
                {"name": "deploy", "keywords": ["deploy"], "required_fields": ["env"]},
                {"name": "delete", "keywords": ["delete"], "severity": "critical"},
            ]
        })
        pre = PreLLM(config_path=path)
        assert len(pre.config.domain_rules) == 2
        assert pre.config.domain_rules[0].required_fields == ["env"]
        os.unlink(path)

    def test_load_with_prompts(self):
        path = self._write_yaml({
            "prompts": {
                "classify_prompt": "Classify this: {query}",
                "structure_prompt": "Extract fields from: {query}",
            }
        })
        pre = PreLLM(config_path=path)
        assert "Classify" in pre.config.prompts.classify_prompt
        assert "Extract" in pre.config.prompts.structure_prompt
        os.unlink(path)

    def test_load_empty_config(self):
        path = self._write_yaml({})
        pre = PreLLM(config_path=path)
        assert pre.config.default_strategy == DecompositionStrategy.CLASSIFY
        os.unlink(path)

    def test_shipped_config_loads(self):
        from pathlib import Path
        config_path = Path(__file__).parent.parent / "configs" / "prellm_config.yaml"
        if config_path.exists():
            pre = PreLLM(config_path=config_path)
            assert pre.config.policy == Policy.DEVOPS
            assert len(pre.config.domain_rules) >= 4


class TestProcessModels:
    def test_process_step_with_strategy(self):
        step = ProcessStep(
            name="deploy", prompt="Deploy app",
            strategy=DecompositionStrategy.ENRICH,
            approval=ApprovalMode.MANUAL,
            rollback=True,
        )
        assert step.strategy == DecompositionStrategy.ENRICH
        assert step.rollback is True

    def test_process_config(self):
        cfg = ProcessConfig(
            process="test",
            steps=[
                ProcessStep(name="s1", prompt="p1"),
                ProcessStep(name="s2", prompt="p2", depends_on=["s1"]),
            ],
        )
        assert len(cfg.steps) == 2
        assert "s1" in cfg.steps[1].depends_on
