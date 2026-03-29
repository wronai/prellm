"""Markdown trace formatter — generates markdown documentation of execution trace."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from prellm.trace.utils import _step_icon, _safe_json

if TYPE_CHECKING:
    from prellm.trace.models import TraceRecorder, TraceStep


def _generate_markdown_header(recorder: "TraceRecorder") -> list[str]:
    """Generate markdown header section."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [
        f"# preLLM Execution Trace",
        f"",
        f"> **Query**: `{recorder.query}`",
        f"> **Timestamp**: {ts}",
        f"> **Total duration**: {recorder.total_duration_ms:.0f}ms",
        f"",
    ]


def _generate_markdown_config(recorder: "TraceRecorder") -> list[str]:
    """Generate markdown configuration section."""
    if not recorder.config:
        return []

    lines = [
        f"## Configuration",
        f"",
        f"| Parameter | Value |",
        f"|---|---|",
    ]

    for key, val in recorder.config.items():
        lines.append(f"| `{key}` | `{val}` |")

    lines.append(f"")
    return lines


def _generate_markdown_step_details(step: "TraceStep", step_num: int) -> list[str]:
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


def _generate_markdown_decision_path(recorder: "TraceRecorder") -> list[str]:
    """Generate markdown decision path section."""
    lines = [f"## Decision Path", f""]

    for i, step in enumerate(recorder.steps, 1):
        lines.extend(_generate_markdown_step_details(step, i))

    return lines


def _generate_markdown_result(recorder: "TraceRecorder") -> list[str]:
    """Generate markdown result section."""
    if not recorder.result_summary:
        return []

    lines = [f"## Result", f""]

    content = recorder.result_summary.get("content", "")
    if content:
        lines.extend([
            f"**Response** ({len(content)} chars):",
            f"",
            f"```",
            content,
            f"```",
            f"",
        ])

    for key, val in recorder.result_summary.items():
        if key != "content":
            lines.append(f"- **{key}**: `{val}`")

    lines.append(f"")
    return lines


def _generate_markdown_summary(recorder: "TraceRecorder") -> list[str]:
    """Generate markdown summary section."""
    lines = [
        f"## Summary",
        f"",
        f"| # | Step | Type | Duration | Status |",
        f"|---|---|---|---|---|",
    ]

    for i, step in enumerate(recorder.steps, 1):
        dur = f"{step.duration_ms:.0f}ms" if step.duration_ms > 0 else "—"
        lines.append(f"| {i} | {step.name} | `{step.step_type}` | {dur} | {_step_icon(step.status)} {step.status} |")

    lines.extend([
        f"",
        f"**Total**: {recorder.total_duration_ms:.0f}ms",
        f"",
    ])

    return lines


def generate_markdown(recorder: "TraceRecorder") -> str:
    """Generate full markdown trace document."""
    sections = [
        _generate_markdown_header(recorder),
        _generate_markdown_config(recorder),
        _generate_markdown_decision_path(recorder),
        _generate_markdown_result(recorder),
        _generate_markdown_summary(recorder),
    ]

    lines = []
    for section in sections:
        lines.extend(section)

    return "\n".join(lines)
