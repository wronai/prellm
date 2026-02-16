"""Tests for prellm.trace module — TraceRecorder and markdown generation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from prellm.trace import TraceRecorder, TraceStep, get_current_trace, set_current_trace


class TestTraceStep:
    def test_defaults(self):
        s = TraceStep(name="test")
        assert s.name == "test"
        assert s.step_type == "action"
        assert s.status == "ok"
        assert s.inputs == {}
        assert s.outputs == {}
        assert s.error is None


class TestTraceRecorder:
    def test_start_stop(self):
        rec = TraceRecorder()
        rec.start(query="hello", small_llm="ollama/qwen:7b", large_llm="gpt-4o")
        assert rec.query == "hello"
        assert rec.config["small_llm"] == "ollama/qwen:7b"
        assert get_current_trace() is rec
        rec.stop()
        assert get_current_trace() is None
        assert rec.total_duration_ms > 0

    def test_step_recording(self):
        rec = TraceRecorder()
        rec.start(query="test")
        rec.step(name="Step 1", step_type="config", outputs={"model": "gpt-4o"})
        rec.step(name="Step 2", step_type="llm_call", duration_ms=1234.5,
                 inputs={"prompt": "hello"}, outputs={"response": "world"})
        rec.step(name="Step 3", step_type="action", status="error", error="timeout")
        rec.stop()

        assert len(rec.steps) == 3
        assert rec.steps[0].name == "Step 1"
        assert rec.steps[1].duration_ms == 1234.5
        assert rec.steps[2].status == "error"
        assert rec.steps[2].error == "timeout"

    def test_set_result(self):
        rec = TraceRecorder()
        rec.start(query="test")
        rec.set_result(content="Hello world", model_used="gpt-4o", retries=0)
        rec.stop()

        assert rec.result_summary["content"] == "Hello world"
        assert rec.result_summary["model_used"] == "gpt-4o"

    def test_to_markdown(self):
        rec = TraceRecorder()
        rec.start(query="Deploy app", small_llm="ollama/qwen:7b", large_llm="gpt-4o", strategy="classify")
        rec.step(name="Configuration", step_type="config",
                 description="Resolved params.",
                 outputs={"small_llm": "ollama/qwen:7b", "large_llm": "gpt-4o"})
        rec.step(name="Pipeline: classify", step_type="llm_call",
                 description="LLM classification step",
                 outputs={"classification": {"intent": "deploy", "confidence": 0.9}},
                 duration_ms=500)
        rec.step(name="PreprocessorAgent", step_type="agent",
                 inputs={"query": "Deploy app"},
                 outputs={"executor_input": "Deploy app"},
                 duration_ms=600)
        rec.step(name="ExecutorAgent", step_type="llm_call",
                 inputs={"executor_input": "Deploy app"},
                 outputs={"content_preview": "Here are the steps..."},
                 duration_ms=2000)
        rec.set_result(content="Here are the steps to deploy...", model_used="gpt-4o")
        rec.stop()

        md = rec.to_markdown()

        # Check structure
        assert "# preLLM Execution Trace" in md
        assert "**Query**: `Deploy app`" in md
        assert "## Configuration" in md
        assert "## Decision Path" in md
        assert "## Result" in md
        assert "## Summary" in md
        # Check config table
        assert "| `small_llm` | `ollama/qwen:7b` |" in md
        # Check steps
        assert "### Step 1: Configuration" in md
        assert "### Step 2: Pipeline: classify" in md
        assert "### Step 3: PreprocessorAgent" in md
        assert "### Step 4: ExecutorAgent" in md
        # Check duration in summary table
        assert "500ms" in md
        assert "2000ms" in md
        # Check result
        assert "Here are the steps to deploy" in md

    def test_to_stdout(self):
        rec = TraceRecorder()
        rec.start(query="Hello world", small_llm="s", large_llm="l", strategy="classify")
        rec.step(name="Config", step_type="config")
        rec.step(name="Classify", step_type="llm_call", duration_ms=100,
                 outputs={"intent": "greeting"})
        rec.set_result(content="Hi there!", model_used="l")
        rec.stop()

        out = rec.to_stdout()

        assert "preLLM Trace" in out
        assert "Hello world" in out
        assert "Config" in out
        assert "Classify" in out
        assert "100ms" in out
        assert "Hi there!" in out

    def test_save_creates_file(self, tmp_path: Path):
        rec = TraceRecorder(output_dir=tmp_path / ".prellm")
        rec.start(query="Test save")
        rec.step(name="Step 1", step_type="config")
        rec.stop()

        filepath = rec.save()

        assert filepath.exists()
        assert filepath.suffix == ".md"
        assert filepath.parent == tmp_path / ".prellm"
        content = filepath.read_text()
        assert "# preLLM Execution Trace" in content
        assert "Test save" in content

    def test_save_custom_dir(self, tmp_path: Path):
        rec = TraceRecorder()
        rec.start(query="Custom dir test")
        rec.stop()

        custom = tmp_path / "custom_traces"
        filepath = rec.save(output_dir=custom)

        assert filepath.exists()
        assert filepath.parent == custom

    def test_empty_trace(self):
        rec = TraceRecorder()
        rec.start(query="empty")
        rec.stop()

        md = rec.to_markdown()
        assert "# preLLM Execution Trace" in md
        assert "empty" in md

        out = rec.to_stdout()
        assert "preLLM Trace" in out


class TestContextVar:
    def test_set_and_get(self):
        assert get_current_trace() is None
        rec = TraceRecorder()
        set_current_trace(rec)
        assert get_current_trace() is rec
        set_current_trace(None)
        assert get_current_trace() is None

    def test_start_sets_context(self):
        rec = TraceRecorder()
        rec.start(query="test")
        assert get_current_trace() is rec
        rec.stop()
        assert get_current_trace() is None


class TestMarkdownEdgeCases:
    def test_long_content_truncated(self):
        rec = TraceRecorder()
        rec.start(query="test")
        # Need enough keys to exceed 2000 chars in JSON after sanitization
        big_outputs = {f"key_{i}": "x" * 200 for i in range(20)}
        rec.step(name="Big output", outputs=big_outputs)
        rec.stop()

        md = rec.to_markdown()
        assert "truncated" in md

    def test_pydantic_model_in_outputs(self):
        """Outputs with Pydantic-like objects should be serialized."""
        rec = TraceRecorder()
        rec.start(query="test")
        rec.step(name="Step", outputs={"path": Path("/tmp/test"), "number": 42})
        rec.stop()

        md = rec.to_markdown()
        assert "/tmp/test" in md
        assert "42" in md

    def test_result_content_preview(self):
        long_content = "A" * 1000
        rec = TraceRecorder()
        rec.start(query="test")
        rec.set_result(content=long_content)
        rec.stop()

        md = rec.to_markdown()
        assert "..." in md  # Should be truncated at 500 chars
        assert f"({len(long_content)} chars)" in md

    def test_error_step_in_markdown(self):
        rec = TraceRecorder()
        rec.start(query="test")
        rec.step(name="Broken", status="error", error="Connection refused")
        rec.stop()

        md = rec.to_markdown()
        assert "❌" in md
        assert "Connection refused" in md

    def test_skipped_step_in_markdown(self):
        rec = TraceRecorder()
        rec.start(query="test")
        rec.step(name="Skipped", status="skipped")
        rec.stop()

        md = rec.to_markdown()
        assert "⏭️" in md
