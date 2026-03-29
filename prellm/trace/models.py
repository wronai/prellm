"""Trace data models — dataclasses for execution trace recording."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
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
        self.start_time = __import__("time").time()
        self.config = config
        set_current_trace(self)

    def stop(self) -> None:
        """Stop recording."""
        self.end_time = __import__("time").time()
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

    def save(self, output_dir: Path | str | None = None) -> Path:
        """Save markdown trace to .prellm/ directory.

        Returns:
            Path to the saved file.
        """
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = self.query[:40].replace(" ", "_").replace("/", "_")
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        filename = f"trace_{ts}_{slug}.md"
        filepath = out / filename

        # Import here to avoid circular dependency
        from prellm.trace.markdown import generate_markdown
        md = generate_markdown(self)
        filepath.write_text(md, encoding="utf-8")
        return filepath
