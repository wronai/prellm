"""Tests for SensitiveDataFilter (Task 2)."""

import pytest

from prellm.context.sensitive_filter import SensitiveDataFilter
from prellm.models import SensitivityLevel


class TestSensitiveDataFilter:

    def test_blocks_api_keys(self):
        """Keys containing API_KEY should be blocked."""
        filt = SensitiveDataFilter()
        data = {"OPENAI_API_KEY": "sk-abc123", "HOME": "/home/user"}
        result = filt.filter_dict(data)
        assert "OPENAI_API_KEY" not in result
        assert "HOME" in result

    def test_masks_database_urls(self):
        """DATABASE_URL should be masked, not blocked."""
        filt = SensitiveDataFilter()
        data = {"DATABASE_URL": "postgres://user:pass@host/db", "HOME": "/home/user"}
        result = filt.filter_dict(data)
        assert "DATABASE_URL" in result
        assert result["DATABASE_URL"] != "postgres://user:pass@host/db"
        assert "***" in result["DATABASE_URL"]

    def test_passes_safe_vars(self):
        """LANG, TERM, SHELL should pass through."""
        filt = SensitiveDataFilter()
        data = {"LANG": "en_US.UTF-8", "TERM": "xterm-256color", "SHELL": "/bin/bash"}
        result = filt.filter_dict(data)
        assert result == data

    def test_detects_token_by_value_pattern(self):
        """Values matching token patterns should be blocked."""
        filt = SensitiveDataFilter()
        assert filt.classify_value("sk-abcdefghijklmnopqrstuvwxyz") == SensitivityLevel.BLOCKED
        assert filt.classify_value("ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789") == SensitivityLevel.BLOCKED
        assert filt.classify_value("just_a_normal_value") == SensitivityLevel.SAFE

    def test_filter_report_tracks_blocked(self):
        """Filter report should track blocked and safe keys."""
        filt = SensitiveDataFilter()
        data = {
            "OPENAI_API_KEY": "sk-test123",
            "SECRET_VALUE": "hidden",
            "HOME": "/home/user",
            "LANG": "en_US",
        }
        filt.filter_dict(data)
        report = filt.get_filter_report()
        assert "OPENAI_API_KEY" in report.blocked_keys
        assert "SECRET_VALUE" in report.blocked_keys
        assert "HOME" in report.safe_keys
        assert report.total_processed == 4

    def test_custom_rules_from_yaml(self, tmp_path):
        """Custom YAML rules should be loaded and applied."""
        rules = tmp_path / "rules.yaml"
        rules.write_text("""
sensitive_keys:
  blocked:
    - "CUSTOM_SECRET"
  safe:
    - "CUSTOM_SAFE_KEY"
sensitive_value_patterns:
  - "mytoken_[a-z]{10,}"
""")
        filt = SensitiveDataFilter(rules_path=str(rules))
        assert filt.classify_key("CUSTOM_SECRET_DATA") == SensitivityLevel.BLOCKED
        assert filt.classify_key("CUSTOM_SAFE_KEY") == SensitivityLevel.SAFE
        assert filt.classify_value("mytoken_abcdefghijk") == SensitivityLevel.BLOCKED

    def test_filter_context_for_large_llm_is_strict(self):
        """filter_context_for_large_llm should recursively remove sensitive data."""
        filt = SensitiveDataFilter()
        context = {
            "query": "deploy app",
            "env": {
                "API_KEY": "secret123",
                "HOME": "/home/user",
            },
            "text_with_token": "use key sk-abcdefghijklmnopqrstuvwxyz123 here",
        }
        result = filt.filter_context_for_large_llm(context)
        assert "API_KEY" not in result.get("env", {})
        assert "HOME" in result.get("env", {})
        assert "[REDACTED]" in result["text_with_token"]

    def test_sanitize_text_redacts_tokens(self):
        """sanitize_text should mask tokens in free text."""
        filt = SensitiveDataFilter()
        text = "Please use key sk-abcdefghijklmnopqrstuvwxyz for the API"
        result = filt.sanitize_text(text)
        assert "sk-abcdef" not in result
        assert "[REDACTED]" in result

    def test_classify_key_sensitivity_levels(self):
        """classify_key should return correct levels."""
        filt = SensitiveDataFilter()
        assert filt.classify_key("HOME") == SensitivityLevel.SAFE
        assert filt.classify_key("OPENAI_API_KEY") == SensitivityLevel.BLOCKED
        assert filt.classify_key("DATABASE_URL") == SensitivityLevel.MASKED
