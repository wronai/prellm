"""Tests for ContextEngine."""

from __future__ import annotations

import pytest
from prellm.analyzers.context_engine import ContextEngine


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

    def test_extra_overrides_env(self, monkeypatch):
        monkeypatch.setenv("CLUSTER", "k8s-prod")
        engine = ContextEngine([{"env": ["CLUSTER"]}])
        result = engine.enrich_prompt("Deploy to {CLUSTER}", extra={"CLUSTER": "k8s-staging"})
        assert result == "Deploy to k8s-staging"

    def test_missing_env_graceful(self):
        engine = ContextEngine([{"env": ["NONEXISTENT_12345"]}])
        ctx = engine.gather()
        assert "NONEXISTENT_12345" not in ctx

    def test_system_context(self):
        engine = ContextEngine([{"system": ["hostname", "os", "python"]}])
        ctx = engine.gather()
        assert "hostname" in ctx
        assert "os" in ctx
        assert "python" in ctx

    def test_empty_sources(self):
        engine = ContextEngine([])
        assert engine.gather() == {}

    def test_multiple_sources(self, monkeypatch):
        monkeypatch.setenv("APP", "myapp")
        engine = ContextEngine([
            {"env": ["APP"]},
            {"system": ["hostname"]},
        ])
        ctx = engine.gather()
        assert "APP" in ctx
        assert "hostname" in ctx
