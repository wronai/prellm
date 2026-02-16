"""Tests for full context pipeline integration (Task 7)."""

import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prellm.context.shell_collector import ShellContextCollector
from prellm.context.sensitive_filter import SensitiveDataFilter
from prellm.context.folder_compressor import FolderCompressor
from prellm.context.schema_generator import ContextSchemaGenerator
from prellm.models import ShellContext, CompressedFolder, ContextSchema


class TestFullContextPipeline:

    def test_end_to_end_with_shell_context(self):
        """Full pipeline: collect → filter → schema should work end-to-end."""
        collector = ShellContextCollector()
        shell_ctx = collector.collect_all()
        assert isinstance(shell_ctx, ShellContext)

        filt = SensitiveDataFilter()
        filtered_env = filt.filter_dict(shell_ctx.env_vars)
        assert isinstance(filtered_env, dict)

        gen = ContextSchemaGenerator()
        schema = gen.generate(shell_context=shell_ctx, sensitive_blocked=len(filt.get_filter_report().blocked_keys))
        assert isinstance(schema, ContextSchema)
        assert schema.platform != ""

    def test_sensitive_data_never_reaches_large_llm(self):
        """Sensitive data should be removed by filter before large LLM."""
        filt = SensitiveDataFilter()
        context = {
            "query": "deploy app",
            "OPENAI_API_KEY": "sk-secret123456789012345678901234",
            "HOME": "/home/user",
            "text": "use sk-abcdefghijklmnopqrstuvwxyz123 key",
        }
        result = filt.filter_context_for_large_llm(context)
        # API key should be gone
        assert "OPENAI_API_KEY" not in result
        # Home should remain
        assert result["HOME"] == "/home/user"
        # Token in text should be redacted
        assert "sk-abcdef" not in result["text"]

    def test_auto_strategy_with_full_context(self):
        """Auto strategy selection should work with a context schema."""
        gen = ContextSchemaGenerator()
        schema = gen.generate()
        assert isinstance(schema, ContextSchema)
        # Schema should be usable for auto strategy prompt
        prompt_section = gen.to_prompt_section(schema)
        assert len(prompt_section) > 10

    def test_folder_compression_included_in_schema(self, tmp_path):
        """Schema should include folder compression data."""
        (tmp_path / "app.py").write_text("def main():\n    pass\n")

        compressor = FolderCompressor()
        compressed = compressor.compress(tmp_path)

        gen = ContextSchemaGenerator()
        schema = gen.generate(folder_compressed=compressed)
        assert schema.project_type == "python"
        assert schema.project_summary is not None

    def test_performance_under_200ms(self):
        """Shell context collection should complete in under 200ms."""
        collector = ShellContextCollector()
        t0 = time.monotonic()
        result = collector.collect_all()
        duration_ms = (time.monotonic() - t0) * 1000
        assert duration_ms < 200, f"Collection took {duration_ms:.0f}ms, expected < 200ms"

    def test_graceful_degradation(self):
        """Pipeline should work even without shell/folder context."""
        gen = ContextSchemaGenerator()
        schema = gen.generate(
            shell_context=None,
            folder_compressed=None,
            user_memory=None,
        )
        assert isinstance(schema, ContextSchema)
        assert schema.execution_env == "cli"
        assert schema.platform != ""

    def test_context_schema_token_cost(self):
        """Schema token cost should be reasonable."""
        gen = ContextSchemaGenerator()
        schema = gen.generate()
        assert schema.schema_token_cost > 0
        assert schema.schema_token_cost < 1000  # minimal schema

    def test_filter_then_schema(self):
        """Filter env vars first, then generate schema — standard flow."""
        collector = ShellContextCollector()
        shell_ctx = collector.collect_all()

        filt = SensitiveDataFilter()
        filtered = filt.filter_dict(shell_ctx.env_vars)
        report = filt.get_filter_report()

        gen = ContextSchemaGenerator()
        # Create a new ShellContext with filtered env vars
        clean_ctx = shell_ctx.model_copy(update={"env_vars": filtered})
        schema = gen.generate(
            shell_context=clean_ctx,
            sensitive_blocked=len(report.blocked_keys),
        )
        assert schema.sensitive_fields_blocked >= 0
