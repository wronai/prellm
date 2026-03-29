"""Terminal trace formatter — generates rich terminal output with decision tree."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from prellm.trace.utils import (
    _step_icon,
    _format_tree_value,
    _extract_prompt_text,
    _wrap_text,
)

if TYPE_CHECKING:
    from prellm.trace.models import TraceRecorder, TraceStep


def _collect_trace_data(recorder: "TraceRecorder") -> dict:
    """Collect and organize data from trace steps for visualization."""
    classification = None
    matched_rule = None
    composed_prompt = None
    executor_input = None
    final_content = recorder.result_summary.get("content", "")
    prep_ms = 0.0
    exec_ms = 0.0

    pipeline_steps: list[TraceStep] = []
    for s in recorder.steps:
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


def _generate_header(recorder: "TraceRecorder", W: int) -> list[str]:
    """Generate header section of terminal output."""
    small = recorder.config.get("small_llm", "?")
    large = recorder.config.get("large_llm", "?")
    strategy = recorder.config.get("strategy", "?")

    lines = [
        "",
        f"{'═' * W}",
        f"  🧠 preLLM Trace",
        f"{'─' * W}",
        f"  Query:    {recorder.query}",
        f"  Strategy: {strategy}",
        f"  Models:   {small} (small) → {large} (large)",
        f"{'═' * W}",
    ]
    return lines


def _generate_decision_tree(recorder: "TraceRecorder", W: int) -> list[str]:
    """Generate decision tree visualization."""
    lines = [
        "",
        f"  📊 Decision Tree",
        f"  {'─' * (W - 4)}",
        "",
    ]

    data = _collect_trace_data(recorder)
    classification = data["classification"]
    matched_rule = data["matched_rule"]
    composed_prompt = data["composed_prompt"]
    executor_input = data["executor_input"]
    final_content = data["final_content"]
    prep_ms = data["prep_ms"]
    exec_ms = data["exec_ms"]
    pipeline_steps = data["pipeline_steps"]

    small = recorder.config.get("small_llm", "?")
    large = recorder.config.get("large_llm", "?")
    strategy = recorder.config.get("strategy", "?")

    # USER node
    lines.extend([
        f"  👤 USER",
        f"  │",
        f"  │  \"{recorder.query}\"",
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
        if prompt_text and prompt_text != recorder.query:
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


def _generate_response_section(recorder: "TraceRecorder", W: int) -> list[str]:
    """Generate response content section."""
    lines = [""]

    data = _collect_trace_data(recorder)
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


def _generate_timing_breakdown(recorder: "TraceRecorder", W: int) -> list[str]:
    """Generate timing breakdown section."""
    total_ms = recorder.total_duration_ms
    data = _collect_trace_data(recorder)
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


def _generate_step_log(recorder: "TraceRecorder", W: int) -> list[str]:
    """Generate step log section."""
    lines = [
        "",
        f"  📝 Step Log",
        f"  {'─' * (W - 4)}",
    ]

    for i, s in enumerate(recorder.steps, 1):
        icon = _step_icon(s.status)
        dur = f" ({s.duration_ms:.0f}ms)" if s.duration_ms > 0 else ""
        type_tag = f"[{s.step_type}]"
        lines.append(f"  {i:>2}. {icon} {s.name}{dur}  {type_tag}")
        if s.error:
            lines.append(f"      ✗ {s.error}")

    return lines


def _generate_footer(recorder: "TraceRecorder", W: int) -> list[str]:
    """Generate footer section."""
    total_ms = recorder.total_duration_ms
    model = recorder.result_summary.get("model_used", "")
    small_model = recorder.result_summary.get("small_model_used", "")
    retries = recorder.result_summary.get("retries", 0)

    lines = [
        "",
        f"{'═' * W}",
        f"  Small: {small_model} | Large: {model} | Retries: {retries} | Total: {total_ms:.0f}ms",
        f"{'═' * W}",
        "",
    ]
    return lines


def generate_terminal_output(recorder: "TraceRecorder") -> str:
    """Generate rich terminal trace with decision tree visualization."""
    W = min(shutil.get_terminal_size(fallback=(100, 24)).columns, 120)

    sections = [
        _generate_header(recorder, W),
        _generate_decision_tree(recorder, W),
        _generate_response_section(recorder, W),
        _generate_timing_breakdown(recorder, W),
        _generate_step_log(recorder, W),
        _generate_footer(recorder, W),
    ]

    lines = []
    for section in sections:
        lines.extend(section)

    return "\n".join(lines)
