"""Execution trace recorder — generates readable markdown documentation of the decision path.

Records each step of the preLLM pipeline (classification, structure extraction,
domain matching, LLM calls, etc.) with inputs, outputs, timing, and rationale.

Output:
  - Markdown file saved to .prellm/ in the working directory
  - Printed to stdout for immediate inspection
"""

from __future__ import annotations

import json
import os
import shutil
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ─── Context var for current trace (thread/async safe) ───────────────────────

_current_trace: ContextVar["TraceRecorder | None"] = ContextVar("_current_trace", default=None)


def get_current_trace() -> "TraceRecorder | None":
    """Get the active trace recorder for the current execution context."""
    return _current_trace.get()


def set_current_trace(trace: "TraceRecorder | None") -> None:
    """Set the active trace recorder for the current execution context."""
    _current_trace.set(trace)


# ─── Data classes ────────────────────────────────────────────────────────────


@dataclass
class TraceStep:
    """A single recorded step in the execution trace."""
    name: str
    step_type: str = "action"  # "config", "llm_call", "pipeline_step", "agent", "action", "result"
    description: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    status: str = "ok"  # "ok", "error", "skipped"
    error: str | None = None
    children: list["TraceStep"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceRecorder:
    """Records execution trace and generates markdown documentation."""
    query: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    steps: list[TraceStep] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    result_summary: dict[str, Any] = field(default_factory=dict)
    output_dir: Path = field(default_factory=lambda: Path(".prellm"))

    def start(self, query: str, **config: Any) -> None:
        """Start recording a trace."""
        self.query = query
        self.start_time = time.time()
        self.config = config
        set_current_trace(self)

    def stop(self) -> None:
        """Stop recording."""
        self.end_time = time.time()
        set_current_trace(None)

    def step(
        self,
        name: str,
        step_type: str = "action",
        description: str = "",
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
        status: str = "ok",
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TraceStep:
        """Record a single execution step."""
        s = TraceStep(
            name=name,
            step_type=step_type,
            description=description,
            inputs=inputs or {},
            outputs=outputs or {},
            duration_ms=duration_ms,
            status=status,
            error=error,
            metadata=metadata or {},
        )
        self.steps.append(s)
        return s

    def set_result(self, **kwargs: Any) -> None:
        """Record the final result summary."""
        self.result_summary = kwargs

    @property
    def total_duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    # ─── Markdown generation ─────────────────────────────────────────────

    def _generate_markdown_header(self) -> list[str]:
        """Generate markdown header section."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return [
            f"# preLLM Execution Trace",
            f"",
            f"> **Query**: `{self.query}`",
            f"> **Timestamp**: {ts}",
            f"> **Total duration**: {self.total_duration_ms:.0f}ms",
            f"",
        ]
    
    def _generate_markdown_config(self) -> list[str]:
        """Generate markdown configuration section."""
        if not self.config:
            return []
        
        lines = [
            f"## Configuration",
            f"",
            f"| Parameter | Value |",
            f"|---|---|",
        ]
        
        for key, val in self.config.items():
            lines.append(f"| `{key}` | `{val}` |")
        
        lines.append(f"")
        return lines
    
    def _generate_markdown_step_details(self, step: TraceStep, step_num: int) -> list[str]:
        """Generate markdown details for a single step."""
        icon = _step_icon(step.status)
        type_badge = f"`{step.step_type}`"
        
        lines = [
            f"### Step {step_num}: {step.name} {icon}",
            f"",
        ]
        
        if step.description:
            lines.extend([step.description, f""])
        
        lines.extend([
            f"- **Type**: {type_badge}",
            f"- **Status**: {step.status}",
        ])
        
        if step.duration_ms > 0:
            lines.append(f"- **Duration**: {step.duration_ms:.0f}ms")
        
        if step.error:
            lines.append(f"- **Error**: `{step.error}`")
        
        lines.append(f"")
        
        # Inputs
        if step.inputs:
            lines.extend([
                f"<details>",
                f"<summary>Inputs</summary>",
                f"",
                f"```json",
                _safe_json(step.inputs),
                f"```",
                f"</details>",
                f"",
            ])
        
        # Outputs
        if step.outputs:
            lines.extend([
                f"<details>",
                f"<summary>Outputs</summary>",
                f"",
                f"```json",
                _safe_json(step.outputs),
                f"```",
                f"</details>",
                f"",
            ])
        
        # Metadata
        if step.metadata:
            for mk, mv in step.metadata.items():
                lines.append(f"- **{mk}**: `{mv}`")
            lines.append(f"")
        
        lines.extend([f"---", f""])
        return lines
    
    def _generate_markdown_decision_path(self) -> list[str]:
        """Generate markdown decision path section."""
        lines = [f"## Decision Path", f""]
        
        for i, step in enumerate(self.steps, 1):
            lines.extend(self._generate_markdown_step_details(step, i))
        
        return lines
    
    def _generate_markdown_result(self) -> list[str]:
        """Generate markdown result section."""
        if not self.result_summary:
            return []
        
        lines = [f"## Result", f""]
        
        content = self.result_summary.get("content", "")
        if content:
            lines.extend([
                f"**Response** ({len(content)} chars):",
                f"",
                f"```",
                content,
                f"```",
                f"",
            ])
        
        for key, val in self.result_summary.items():
            if key != "content":
                lines.append(f"- **{key}**: `{val}`")
        
        lines.append(f"")
        return lines
    
    def _generate_markdown_summary(self) -> list[str]:
        """Generate markdown summary section."""
        lines = [
            f"## Summary",
            f"",
            f"| # | Step | Type | Duration | Status |",
            f"|---|---|---|---|---|",
        ]
        
        for i, step in enumerate(self.steps, 1):
            dur = f"{step.duration_ms:.0f}ms" if step.duration_ms > 0 else "—"
            lines.append(f"| {i} | {step.name} | `{step.step_type}` | {dur} | {_step_icon(step.status)} {step.status} |")
        
        lines.extend([
            f"",
            f"**Total**: {self.total_duration_ms:.0f}ms",
            f"",
        ])
        
        return lines
    
    def to_markdown(self) -> str:
        """Generate full markdown trace document."""
        sections = [
            self._generate_markdown_header(),
            self._generate_markdown_config(),
            self._generate_markdown_decision_path(),
            self._generate_markdown_result(),
            self._generate_markdown_summary(),
        ]
        
        # Combine all sections
        lines = []
        for section in sections:
            lines.extend(section)
        
        return "\n".join(lines)

    def _collect_trace_data(self) -> dict:
        """Collect and organize data from trace steps for visualization."""
        classification = None
        matched_rule = None
        composed_prompt = None
        executor_input = None
        final_content = self.result_summary.get("content", "")
        prep_ms = 0.0
        exec_ms = 0.0

        pipeline_steps: list[TraceStep] = []
        for s in self.steps:
            if s.step_type == "config":
                continue
            if s.name.startswith("Pipeline:"):
                pipeline_steps.append(s)
                out = s.outputs
                if "classification" in out:
                    classification = out["classification"]
                if "matched_rule" in out:
                    matched_rule = out["matched_rule"]
                if "composed_prompt" in out:
                    composed_prompt = out["composed_prompt"]
            elif s.step_type == "agent":
                prep_ms = s.duration_ms
                executor_input = s.outputs.get("executor_input", "")
            elif s.name.startswith("ExecutorAgent"):
                exec_ms = s.duration_ms

        return {
            "classification": classification,
            "matched_rule": matched_rule,
            "composed_prompt": composed_prompt,
            "executor_input": executor_input,
            "final_content": final_content,
            "prep_ms": prep_ms,
            "exec_ms": exec_ms,
            "pipeline_steps": pipeline_steps,
        }

    def _generate_header(self, W: int) -> list[str]:
        """Generate header section of terminal output."""
        small = self.config.get("small_llm", "?")
        large = self.config.get("large_llm", "?")
        strategy = self.config.get("strategy", "?")
        
        lines = [
            "",
            f"{'═' * W}",
            f"  🧠 preLLM Trace",
            f"{'─' * W}",
            f"  Query:    {self.query}",
            f"  Strategy: {strategy}",
            f"  Models:   {small} (small) → {large} (large)",
            f"{'═' * W}",
        ]
        return lines
    
    def _generate_decision_tree(self, W: int) -> list[str]:
        """Generate decision tree visualization."""
        lines = [
            "",
            f"  📊 Decision Tree",
            f"  {'─' * (W - 4)}",
            "",
        ]
        
        # Collect data from steps for the tree
        data = self._collect_trace_data()
        classification = data["classification"]
        matched_rule = data["matched_rule"]
        composed_prompt = data["composed_prompt"]
        executor_input = data["executor_input"]
        final_content = data["final_content"]
        prep_ms = data["prep_ms"]
        exec_ms = data["exec_ms"]
        pipeline_steps = data["pipeline_steps"]
        
        small = self.config.get("small_llm", "?")
        large = self.config.get("large_llm", "?")
        strategy = self.config.get("strategy", "?")
        
        # USER node
        lines.extend([
            f"  👤 USER",
            f"  │",
            f"  │  \"{self.query}\"",
            f"  │",
            f"  ▼",
        ])
        
        # Small LLM node
        lines.extend([
            f"  🤖 Small LLM: {small}",
            f"  │  Strategy: {strategy} | Time: {prep_ms:.0f}ms",
            f"  │",
        ])
        
        # Show pipeline sub-steps
        for i, ps in enumerate(pipeline_steps):
            is_last = (i == len(pipeline_steps) - 1)
            icon = _step_icon(ps.status)
            step_name = ps.name.replace("Pipeline: ", "")
            connector = "└" if is_last else "├"
            cont = " " if is_last else "│"
            
            lines.append(f"  │  {connector}── {icon} {step_name}")
            
            # Show key output for this sub-step
            for key, val in ps.outputs.items():
                val_str = _format_tree_value(val)
                lines.append(f"  │  {cont}   {key}: {val_str}")
            
            if ps.error:
                lines.append(f"  │  {cont}   ✗ {ps.error}")
        
        lines.append(f"  │")
        
        # Show query transformation
        if composed_prompt:
            prompt_text = _extract_prompt_text(composed_prompt)
            if prompt_text and prompt_text != self.query:
                lines.append(f"  │  📝 Composed prompt:")
                for pline in _wrap_text(prompt_text, W - 10):
                    lines.append(f"  │     {pline}")
                lines.append(f"  │")
        
        lines.extend([
            f"  ▼",
            f"  🧠 Large LLM: {large}",
            f"  │  Time: {exec_ms:.0f}ms",
            f"  │",
            f"  ▼",
            f"  📋 RESULT",
            f"  {'─' * (W - 4)}",
        ])
        
        return lines
    
    def _generate_response_section(self, W: int) -> list[str]:
        """Generate response content section."""
        lines = [""]
        
        data = self._collect_trace_data()
        final_content = data["final_content"]
        
        if final_content:
            lines.extend([
                f"  📄 Response ({len(final_content)} chars):",
                f"  {'─' * (W - 4)}",
            ])
            # Show full content with proper indentation
            for cline in final_content.split("\n"):
                lines.append(f"  {cline}")
            lines.append(f"  {'─' * (W - 4)}")
        
        return lines
    
    def _generate_timing_breakdown(self, W: int) -> list[str]:
        """Generate timing breakdown section."""
        total_ms = self.total_duration_ms
        data = self._collect_trace_data()
        prep_ms = data["prep_ms"]
        exec_ms = data["exec_ms"]
        
        lines = [
            f"  ⏱  Timing Breakdown",
            f"  {'─' * (W - 4)}",
        ]
        
        timing_entries = []
        if prep_ms > 0:
            timing_entries.append(("Small LLM (preprocess)", prep_ms))
        if exec_ms > 0:
            timing_entries.append(("Large LLM (execute)", exec_ms))
        overhead = total_ms - prep_ms - exec_ms
        if overhead > 50:
            timing_entries.append(("Overhead (context/io)", overhead))
        
        # Use max of wall-clock and sum-of-steps as denominator
        sum_ms = sum(ms for _, ms in timing_entries)
        denom = max(total_ms, sum_ms, 1)
        bar_width = min(max(W - 55, 10), 40)
        
        for label, ms in timing_entries:
            pct = (ms / denom * 100)
            filled = max(0, min(bar_width, int(bar_width * ms / denom)))
            bar = "#" * filled + "." * (bar_width - filled)
            lines.append(f"  {label:<28s} [{bar}] {ms:>7.0f}ms ({pct:4.1f}%)")
        
        lines.extend([
            f"  {'─' * (W - 4)}",
            f"  {'Total:':<28s} {'':>{bar_width + 2}s} {total_ms:>7.0f}ms",
        ])
        
        return lines
    
    def _generate_step_log(self, W: int) -> list[str]:
        """Generate step log section."""
        lines = [
            "",
            f"  📝 Step Log",
            f"  {'─' * (W - 4)}",
        ]
        
        for i, s in enumerate(self.steps, 1):
            icon = _step_icon(s.status)
            dur = f" ({s.duration_ms:.0f}ms)" if s.duration_ms > 0 else ""
            type_tag = f"[{s.step_type}]"
            lines.append(f"  {i:>2}. {icon} {s.name}{dur}  {type_tag}")
            if s.error:
                lines.append(f"      ✗ {s.error}")
        
        return lines
    
    def _generate_footer(self, W: int) -> list[str]:
        """Generate footer section."""
        total_ms = self.total_duration_ms
        model = self.result_summary.get("model_used", "")
        small_model = self.result_summary.get("small_model_used", "")
        retries = self.result_summary.get("retries", 0)
        
        lines = [
            "",
            f"{'═' * W}",
            f"  Small: {small_model} | Large: {model} | Retries: {retries} | Total: {total_ms:.0f}ms",
            f"{'═' * W}",
            "",
        ]
        return lines
    
    def to_stdout(self) -> str:
        """Generate rich terminal trace with decision tree visualization."""
        W = min(shutil.get_terminal_size(fallback=(100, 24)).columns, 120)
        
        # Build all sections
        sections = [
            self._generate_header(W),
            self._generate_decision_tree(W),
            self._generate_response_section(W),
            self._generate_timing_breakdown(W),
            self._generate_step_log(W),
            self._generate_footer(W),
        ]
        
        # Combine all sections
        lines = []
        for section in sections:
            lines.extend(section)
        
        return "\n".join(lines)

    def save(self, output_dir: Path | str | None = None) -> Path:
        """Save markdown trace to .prellm/ directory.

        Returns:
            Path to the saved file.
        """
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = self.query[:40].replace(" ", "_").replace("/", "_")
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        filename = f"trace_{ts}_{slug}.md"
        filepath = out / filename

        md = self.to_markdown()
        filepath.write_text(md, encoding="utf-8")
        return filepath


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _step_icon(status: str) -> str:
    return {"ok": "✅", "error": "❌", "skipped": "⏭️"}.get(status, "🔄")


def _safe_json(obj: Any, max_len: int = 50000) -> str:
    """Serialize to JSON with truncation for large values."""
    try:
        text = json.dumps(_sanitize(obj), indent=2, ensure_ascii=False, default=str)
        if len(text) > max_len:
            return text[:max_len] + "\n... (truncated)"
        return text
    except (TypeError, ValueError):
        return str(obj)[:max_len]


def _sanitize(obj: Any, depth: int = 0) -> Any:
    """Sanitize an object for JSON serialization (handle Pydantic, Path, etc.)."""
    if depth > 5:
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj[:10000] if len(obj) > 10000 else obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v, depth + 1) for v in obj[:20]]
    if hasattr(obj, "model_dump"):
        try:
            return _sanitize(obj.model_dump(), depth + 1)
        except Exception:
            return str(obj)[:5000]
    return str(obj)[:5000]


def _compact_value(val: Any, max_len: int = 500) -> str:
    """Compact value for terminal output."""
    if isinstance(val, dict):
        try:
            s = json.dumps(val, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            s = str(val)
    elif isinstance(val, str):
        s = val
    else:
        s = str(val)
    s = s.replace("\n", " ")
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s


def _format_tree_value(val: Any) -> str:
    """Format a value for display in the decision tree — no truncation."""
    if isinstance(val, dict):
        try:
            return json.dumps(val, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(val)
    elif isinstance(val, str):
        return val.replace("\n", " ")
    return str(val)


def _extract_prompt_text(composed: Any) -> str:
    """Extract the prompt text from a composed_prompt output (dict or str)."""
    if isinstance(composed, dict):
        return str(composed.get("composed_prompt", composed))
    return str(composed)


def _wrap_text(text: str, width: int) -> list[str]:
    """Word-wrap text to fit within a given width."""
    if width < 20:
        width = 20
    words = text.replace("\n", " ").split()
    result_lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        if current_len + len(word) + (1 if current else 0) > width:
            if current:
                result_lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + (1 if len(current) > 1 else 0)
    if current:
        result_lines.append(" ".join(current))
    return result_lines or [""]
