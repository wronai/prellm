"""Tests for v0.4 context features — RuntimeContext, ContextEngine.gather_runtime(),
SensitiveDataFilter integration, CodebaseIndexer.get_compressed_context()."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from prellm.analyzers.context_engine import ContextEngine
from prellm.models import RuntimeContext, SessionSnapshot


class TestRuntimeContext:
    """Tests for the RuntimeContext model."""

    def test_runtime_context_defaults(self):
        ctx = RuntimeContext()
        assert ctx.env_safe == {}
        assert ctx.process == {}
        assert ctx.locale == {}
        assert ctx.network == {}
        assert ctx.git is None
        assert ctx.system == {}
        assert ctx.sensitive_blocked_count == 0
        assert ctx.token_estimate == 0

    def test_runtime_context_serialization(self):
        ctx = RuntimeContext(
            env_safe={"LANG": "en_US.UTF-8"},
            process={"pid": 1234, "cwd": "/tmp"},
            locale={"lang": "en_US.UTF-8"},
            network={"hostname": "test"},
            system={"os": "Linux"},
            collected_at="2026-02-16T10:00:00",
            sensitive_blocked_count=5,
            token_estimate=100,
        )
        data = ctx.model_dump()
        assert data["env_safe"]["LANG"] == "en_US.UTF-8"
        assert data["sensitive_blocked_count"] == 5

        # Roundtrip
        ctx2 = RuntimeContext.model_validate(data)
        assert ctx2.env_safe == ctx.env_safe
        assert ctx2.token_estimate == 100


class TestContextEngineGatherRuntime:
    """Tests for ContextEngine.gather_runtime()."""

    def test_gather_runtime_returns_model(self):
        engine = ContextEngine()
        runtime = engine.gather_runtime()
        assert isinstance(runtime, RuntimeContext)

    def test_gather_runtime_has_process(self):
        engine = ContextEngine()
        runtime = engine.gather_runtime()
        assert "pid" in runtime.process
        assert "cwd" in runtime.process
        assert runtime.process["pid"] > 0

    def test_gather_runtime_has_locale(self):
        engine = ContextEngine()
        runtime = engine.gather_runtime()
        assert "encoding" in runtime.locale

    def test_gather_runtime_has_network(self):
        engine = ContextEngine()
        runtime = engine.gather_runtime()
        assert "hostname" in runtime.network

    def test_gather_runtime_has_system(self):
        engine = ContextEngine()
        runtime = engine.gather_runtime()
        assert "os" in runtime.system
        assert "python" in runtime.system

    def test_gather_runtime_has_collected_at(self):
        engine = ContextEngine()
        runtime = engine.gather_runtime()
        assert runtime.collected_at != ""
        assert "T" in runtime.collected_at  # ISO format

    def test_gather_runtime_token_estimate(self):
        engine = ContextEngine()
        runtime = engine.gather_runtime()
        assert runtime.token_estimate > 0

    def test_auto_collect_env_filters_sensitive(self, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "sk-secret123456789012345")
        monkeypatch.setenv("LANG", "en_US.UTF-8")
        engine = ContextEngine()
        env_safe, blocked = engine._auto_collect_env()
        assert "MY_API_KEY" not in env_safe
        assert blocked > 0

    def test_auto_collect_env_passes_safe(self, monkeypatch):
        monkeypatch.setenv("LANG", "pl_PL.UTF-8")
        engine = ContextEngine()
        env_safe, _ = engine._auto_collect_env()
        assert "LANG" in env_safe

    def test_gather_process_returns_pid_cwd(self):
        info = ContextEngine._gather_process()
        assert info["pid"] == os.getpid()
        assert info["cwd"] == os.getcwd()

    def test_gather_locale_detects_encoding(self):
        locale = ContextEngine._gather_locale()
        assert "encoding" in locale
        assert locale["encoding"] != ""

    def test_gather_network_no_public_ip(self):
        network = ContextEngine._gather_network()
        assert "hostname" in network
        assert "local_ip" in network
        # Should not contain public IP (no external queries made)


class TestSessionSnapshot:
    """Tests for SessionSnapshot model."""

    def test_session_snapshot_defaults(self):
        snap = SessionSnapshot()
        assert snap.session_id == ""
        assert snap.interactions == []
        assert snap.preferences == {}
        assert snap.runtime_context is None

    def test_session_snapshot_roundtrip_file(self, tmp_path):
        snap = SessionSnapshot(
            session_id="test-123",
            interactions=[{"query": "Deploy", "response_summary": "OK"}],
            preferences={"lang": "pl"},
            created_at="2026-02-16T10:00:00",
            exported_at="2026-02-16T10:01:00",
        )
        path = tmp_path / "session.json"
        snap.to_file(path)
        assert path.exists()

        loaded = SessionSnapshot.from_file(path)
        assert loaded.session_id == "test-123"
        assert len(loaded.interactions) == 1
        assert loaded.preferences["lang"] == "pl"

    def test_session_snapshot_with_runtime(self):
        runtime = RuntimeContext(
            env_safe={"LANG": "pl_PL"},
            process={"pid": 1},
            collected_at="2026-02-16T10:00:00",
        )
        snap = SessionSnapshot(
            session_id="ctx-test",
            runtime_context=runtime,
        )
        data = snap.model_dump()
        assert data["runtime_context"]["env_safe"]["LANG"] == "pl_PL"


class TestCodebaseIndexerCompressedContext:
    """Tests for CodebaseIndexer.get_compressed_context()."""

    def test_get_compressed_context(self, tmp_path):
        # Create a simple Python project
        (tmp_path / "main.py").write_text('"""Main module."""\ndef main():\n    pass\n')
        (tmp_path / "utils.py").write_text('"""Utility functions."""\ndef helper():\n    pass\n')

        from prellm.context.codebase_indexer import CodebaseIndexer
        indexer = CodebaseIndexer()
        ctx = indexer.get_compressed_context(tmp_path, "main", max_tokens=2048)
        assert "[Project:" in ctx
        assert "modules" in ctx

    def test_get_compressed_context_respects_token_limit(self, tmp_path):
        # Create many files
        for i in range(20):
            (tmp_path / f"mod_{i}.py").write_text(f'"""Module {i}."""\ndef func_{i}():\n    pass\n')

        from prellm.context.codebase_indexer import CodebaseIndexer
        indexer = CodebaseIndexer()
        ctx = indexer.get_compressed_context(tmp_path, "func", max_tokens=100)
        assert indexer.estimate_tokens(ctx) <= 150  # some margin

    def test_estimate_tokens(self):
        from prellm.context.codebase_indexer import CodebaseIndexer
        indexer = CodebaseIndexer()
        assert indexer.estimate_tokens("") == 1
        assert indexer.estimate_tokens("hello world") > 0
        # ~4 chars per token
        assert indexer.estimate_tokens("a" * 100) == 26  # 100//4 + 1


class TestV04Imports:
    """Test v0.4 exports are importable."""

    def test_import_runtime_context(self):
        from prellm import RuntimeContext
        assert RuntimeContext is not None

    def test_import_session_snapshot(self):
        from prellm import SessionSnapshot
        assert SessionSnapshot is not None

    def test_import_sensitive_data_filter(self):
        from prellm import SensitiveDataFilter
        assert SensitiveDataFilter is not None

    def test_import_shell_context_collector(self):
        from prellm import ShellContextCollector
        assert ShellContextCollector is not None

    def test_import_folder_compressor(self):
        from prellm import FolderCompressor
        assert FolderCompressor is not None

    def test_version_040(self):
        import prellm
        assert prellm.__version__ == "0.4.3"

    def test_all_v04_exports(self):
        import prellm
        for name in ["RuntimeContext", "SessionSnapshot", "SensitiveDataFilter",
                     "ShellContextCollector", "FolderCompressor"]:
            assert name in prellm.__all__, f"{name} not in __all__"
