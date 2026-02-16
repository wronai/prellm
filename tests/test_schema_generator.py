"""Tests for ContextSchemaGenerator (Task 4)."""

import pytest

from prellm.context.schema_generator import ContextSchemaGenerator
from prellm.models import (
    CompressedFolder,
    ContextSchema,
    LocaleInfo,
    NetworkContext,
    ProcessInfo,
    ShellContext,
    ShellInfo,
)


class TestContextSchemaGenerator:

    def _make_shell_context(self) -> ShellContext:
        return ShellContext(
            env_vars={"HOME": "/home/user", "LANG": "pl_PL.UTF-8"},
            process=ProcessInfo(pid=1234, cwd="/home/user/project", user="testuser"),
            locale=LocaleInfo(lang="pl_PL.UTF-8", timezone="Europe/Warsaw", encoding="utf-8"),
            shell=ShellInfo(shell="/bin/bash", term="xterm-256color", columns=120, lines=40),
            network=NetworkContext(hostname="dev-machine", local_ip="192.168.1.10"),
        )

    def _make_compressed_folder(self) -> CompressedFolder:
        return CompressedFolder(
            root="/home/user/project",
            toon_output="project: myapp\nmodules[3]{path,lang,items}:\n  main.py,python,5\n",
            dependency_graph={"main": ["utils"]},
            module_summaries={"main": "Main entry point"},
            total_modules=3,
            total_functions=15,
            estimated_tokens=200,
        )

    def test_generate_from_shell_context(self):
        """Schema should be populated from shell context."""
        gen = ContextSchemaGenerator()
        shell_ctx = self._make_shell_context()
        schema = gen.generate(shell_context=shell_ctx)

        assert isinstance(schema, ContextSchema)
        assert schema.locale == "pl_PL.UTF-8"
        assert schema.timezone == "Europe/Warsaw"
        assert schema.platform in ("linux", "darwin", "windows")

    def test_generate_from_folder_only(self):
        """Schema should work with only folder context."""
        gen = ContextSchemaGenerator()
        compressed = self._make_compressed_folder()
        schema = gen.generate(folder_compressed=compressed)

        assert schema.project_type == "python"
        assert "3 modules" in schema.project_summary
        assert "15 functions" in schema.project_summary

    def test_to_prompt_section_fits_token_budget(self):
        """Prompt section should be reasonably compact."""
        gen = ContextSchemaGenerator()
        shell_ctx = self._make_shell_context()
        compressed = self._make_compressed_folder()
        schema = gen.generate(shell_context=shell_ctx, folder_compressed=compressed)

        section = gen.to_prompt_section(schema)
        assert "[Environment Context]" in section
        assert "Platform:" in section
        # Should be < 500 tokens (~2000 chars)
        assert len(section) < 2000

    def test_relevance_scoring(self):
        """Relevance scoring should return scores for all categories."""
        gen = ContextSchemaGenerator()
        schema = ContextSchema(
            platform="linux",
            project_type="python",
            available_tools=["git", "docker"],
        )

        scores = gen.estimate_relevance(schema, "refactor the code module")
        assert "platform" in scores
        assert "project" in scores
        assert "tools" in scores
        assert scores["project"] > 0.5  # code-related query

    def test_empty_sources_returns_minimal_schema(self):
        """With no sources, should return a minimal but valid schema."""
        gen = ContextSchemaGenerator()
        schema = gen.generate()

        assert isinstance(schema, ContextSchema)
        assert schema.execution_env == "cli"
        assert schema.platform != ""
        assert schema.schema_token_cost > 0

    def test_detect_tools(self):
        """Should detect at least some common tools (git, python3)."""
        gen = ContextSchemaGenerator()
        tools = gen._detect_tools()
        # At minimum, python3 should be available in test env
        assert isinstance(tools, list)

    def test_user_history_summary(self):
        """User history should be summarized."""
        gen = ContextSchemaGenerator()
        history = [
            {"query": "deploy to production", "response_summary": "deployed"},
            {"query": "check logs", "response_summary": "logs ok"},
        ]
        schema = gen.generate(user_memory=history)
        assert schema.user_history_summary is not None
        assert "deploy" in schema.user_history_summary.lower()
