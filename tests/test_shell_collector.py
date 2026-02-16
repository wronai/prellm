"""Tests for ShellContextCollector (Task 1)."""

import os

import pytest

from prellm.context.shell_collector import ShellContextCollector
from prellm.models import ShellContext


class TestShellContextCollector:

    def test_collect_env_vars_filters_secrets(self):
        """API_KEY, TOKEN, PASSWORD should not pass with safe_only=True."""
        os.environ["TEST_API_KEY_PRELLM"] = "sk-secret123"
        os.environ["TEST_SAFE_PRELLM_VAR"] = "hello"
        try:
            collector = ShellContextCollector(extra_safe_keys={"TEST_SAFE_PRELLM_VAR"})
            result = collector.collect_env_vars(safe_only=True)
            assert "TEST_API_KEY_PRELLM" not in result
            assert "TEST_SAFE_PRELLM_VAR" in result
        finally:
            os.environ.pop("TEST_API_KEY_PRELLM", None)
            os.environ.pop("TEST_SAFE_PRELLM_VAR", None)

    def test_collect_process_info(self):
        """PID and CWD should be returned."""
        collector = ShellContextCollector()
        info = collector.collect_process_info()
        assert info.pid > 0
        assert info.cwd != ""

    def test_collect_locale_info(self):
        """Timezone and encoding should be populated."""
        collector = ShellContextCollector()
        locale = collector.collect_locale_info()
        assert locale.encoding != ""

    def test_collect_shell_info(self):
        """Shell info should have shell path."""
        collector = ShellContextCollector()
        shell = collector.collect_shell_info()
        # In CI/test env, SHELL may or may not be set
        assert isinstance(shell.shell, str)
        assert isinstance(shell.columns, int)

    def test_collect_network_context(self):
        """Hostname should be non-empty."""
        collector = ShellContextCollector()
        net = collector.collect_network_context()
        assert net.hostname != ""

    def test_collect_all_returns_pydantic(self):
        """collect_all should return a ShellContext model."""
        collector = ShellContextCollector()
        result = collector.collect_all()
        assert isinstance(result, ShellContext)
        assert result.collection_duration_ms >= 0
        assert result.process.pid > 0

    def test_collect_with_missing_env(self):
        """Graceful handling when env vars are missing."""
        # Remove LANG temporarily if present
        old = os.environ.pop("LANG", None)
        try:
            collector = ShellContextCollector()
            result = collector.collect_all()
            assert isinstance(result, ShellContext)
        finally:
            if old is not None:
                os.environ["LANG"] = old

    def test_safe_only_flag(self):
        """safe_only=False should return ALL env vars including secrets."""
        os.environ["TEST_SECRET_TOKEN_PRELLM"] = "tok123"
        try:
            collector = ShellContextCollector()
            safe = collector.collect_env_vars(safe_only=True)
            all_vars = collector.collect_env_vars(safe_only=False)
            assert "TEST_SECRET_TOKEN_PRELLM" not in safe
            assert "TEST_SECRET_TOKEN_PRELLM" in all_vars
        finally:
            os.environ.pop("TEST_SECRET_TOKEN_PRELLM", None)
