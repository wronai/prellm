"""Tests for CodebaseIndexer — tree-sitter / regex fallback codebase indexing."""

from __future__ import annotations

import pytest
from pathlib import Path

from prellm.context.codebase_indexer import CodebaseIndexer, CodebaseIndex, CodeSymbol, FileIndex


class TestCodebaseIndexerRegex:
    """Test regex-based fallback indexing (no tree-sitter required)."""

    def test_index_python_file(self, tmp_path):
        src = tmp_path / "example.py"
        src.write_text(
            "import os\n"
            "from pathlib import Path\n"
            "\n"
            "class MyClass:\n"
            "    def method(self):\n"
            "        pass\n"
            "\n"
            "def top_level_func():\n"
            "    pass\n"
            "\n"
            "async def async_func():\n"
            "    pass\n"
        )

        indexer = CodebaseIndexer()
        index = indexer.index_directory(tmp_path)

        assert index.total_files == 1
        assert index.total_symbols >= 3  # class + method + top_level_func + async_func

        names = [s.name for f in index.files for s in f.symbols]
        assert "MyClass" in names
        assert "top_level_func" in names
        assert "async_func" in names

    def test_index_empty_directory(self, tmp_path):
        indexer = CodebaseIndexer()
        index = indexer.index_directory(tmp_path)
        assert index.total_files == 0
        assert index.total_symbols == 0

    def test_index_excludes_dirs(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config.py").write_text("class GitConfig: pass\n")
        (tmp_path / "src.py").write_text("class App: pass\n")

        indexer = CodebaseIndexer()
        index = indexer.index_directory(tmp_path)

        assert index.total_files == 1
        paths = [f.path for f in index.files]
        assert any("src.py" in p for p in paths)
        assert not any(".git" in p for p in paths)

    def test_index_respects_language_filter(self, tmp_path):
        (tmp_path / "app.py").write_text("class App: pass\n")
        (tmp_path / "app.js").write_text("function init() {}\n")

        indexer = CodebaseIndexer(languages=["python"])
        index = indexer.index_directory(tmp_path)

        assert index.total_files == 1
        assert index.files[0].language == "python"

    def test_search_finds_symbols(self, tmp_path):
        (tmp_path / "deploy.py").write_text(
            "def deploy_app():\n    pass\n\n"
            "def rollback_deploy():\n    pass\n\n"
            "class DeployManager:\n    pass\n"
        )

        indexer = CodebaseIndexer()
        index = indexer.index_directory(tmp_path)
        results = indexer.search(index, "deploy")

        assert len(results) >= 2
        names = [s.name for s in results]
        assert "deploy_app" in names
        assert "DeployManager" in names

    def test_search_no_matches(self, tmp_path):
        (tmp_path / "app.py").write_text("class App: pass\n")

        indexer = CodebaseIndexer()
        index = indexer.index_directory(tmp_path)
        results = indexer.search(index, "nonexistent_xyz")

        assert len(results) == 0

    def test_get_context_for_query(self, tmp_path):
        (tmp_path / "app.py").write_text("def process_data():\n    pass\n")

        indexer = CodebaseIndexer()
        index = indexer.index_directory(tmp_path)
        context = indexer.get_context_for_query(index, "process")

        assert "process_data" in context
        assert "1 files" in context

    def test_get_context_no_matches(self, tmp_path):
        (tmp_path / "app.py").write_text("def foo(): pass\n")

        indexer = CodebaseIndexer()
        index = indexer.index_directory(tmp_path)
        context = indexer.get_context_for_query(index, "nonexistent")

        assert "no matches" in context

    def test_extracts_imports(self, tmp_path):
        (tmp_path / "app.py").write_text(
            "import os\nfrom pathlib import Path\n\ndef main(): pass\n"
        )

        indexer = CodebaseIndexer()
        index = indexer.index_directory(tmp_path)

        assert len(index.files) == 1
        assert "import os" in index.files[0].imports
        assert "from pathlib import Path" in index.files[0].imports

    def test_skips_large_files(self, tmp_path):
        large_file = tmp_path / "big.py"
        large_file.write_text("x = 1\n" * 100000)  # ~600KB

        indexer = CodebaseIndexer()
        index = indexer.index_directory(tmp_path, max_file_size_kb=10)

        assert index.total_files == 0


class TestCodeSymbolDataclass:
    def test_code_symbol_defaults(self):
        s = CodeSymbol(name="foo", kind="function", file_path="a.py", line_start=1, line_end=5)
        assert s.signature == ""
        assert s.docstring == ""
        assert s.parent is None

    def test_file_index_defaults(self):
        fi = FileIndex(path="a.py", language="python")
        assert fi.symbols == []
        assert fi.imports == []
        assert fi.loc == 0


class TestCodebaseIndexerImport:
    def test_importable_from_context(self):
        from prellm.context import CodebaseIndexer as CI
        assert CI is CodebaseIndexer
