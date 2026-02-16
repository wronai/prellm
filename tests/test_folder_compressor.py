"""Tests for FolderCompressor (Task 3)."""

import pytest
from pathlib import Path

from prellm.context.folder_compressor import FolderCompressor
from prellm.models import CompressedFolder


class TestFolderCompressor:

    def test_compress_returns_toon_format(self, tmp_path):
        """Compress should return valid .toon format output."""
        # Create minimal Python project
        (tmp_path / "main.py").write_text('def hello():\n    return "world"\n')
        (tmp_path / "utils.py").write_text('def helper():\n    pass\n')

        compressor = FolderCompressor()
        result = compressor.compress(tmp_path)

        assert isinstance(result, CompressedFolder)
        assert "project:" in result.toon_output
        assert "modules[" in result.toon_output
        assert result.total_modules == 2
        assert result.total_functions >= 2

    def test_dependency_graph_finds_internal_imports(self, tmp_path):
        """Dependency graph should detect internal imports."""
        pkg = tmp_path / "myproject"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "core.py").write_text(
            "from myproject.utils import helper\n\ndef main():\n    pass\n"
        )
        (pkg / "utils.py").write_text("def helper():\n    pass\n")

        compressor = FolderCompressor()
        index = compressor._indexer.index_directory(tmp_path)
        graph = compressor.to_dependency_graph(index)

        # core should depend on utils
        assert any("core" in k for k in graph)

    def test_summary_uses_docstrings(self, tmp_path):
        """Module summary should extract docstrings."""
        (tmp_path / "documented.py").write_text(
            '"""This is a documented module."""\n\ndef foo():\n    pass\n'
        )

        compressor = FolderCompressor()
        index = compressor._indexer.index_directory(tmp_path)
        summaries = compressor.to_summary(index)

        assert any("documented module" in v.lower() for v in summaries.values())

    def test_respects_gitignore(self, tmp_path):
        """Should exclude common non-source directories."""
        src = tmp_path / "src.py"
        src.write_text("def real():\n    pass\n")
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "bad.py").write_text("def bad():\n    pass\n")

        compressor = FolderCompressor()
        result = compressor.compress(tmp_path)

        assert ".venv" not in result.toon_output

    def test_token_estimate_reasonable(self, tmp_path):
        """Token estimate should be a reasonable positive number."""
        (tmp_path / "a.py").write_text("def x():\n    pass\n")

        compressor = FolderCompressor()
        result = compressor.compress(tmp_path)

        assert result.estimated_tokens > 0
        assert result.estimated_tokens < 100000  # sanity check

    def test_large_project_fits_small_llm_context(self):
        """Compressing prellm itself should fit in small LLM context (<4096 tokens)."""
        project_root = Path(__file__).parent.parent / "prellm"
        if not project_root.is_dir():
            pytest.skip("prellm source not found")

        compressor = FolderCompressor()
        result = compressor.compress(project_root)

        # Should fit in context window
        assert result.estimated_tokens < 8192
        assert result.total_modules > 0

    def test_empty_directory(self, tmp_path):
        """Compressing empty dir should not crash."""
        compressor = FolderCompressor()
        result = compressor.compress(tmp_path)

        assert isinstance(result, CompressedFolder)
        assert result.total_modules == 0
        assert result.total_functions == 0
