"""prellm/trace package — Execution trace recording and formatting.

This package contains the trace recording functionality for prellm,
split into focused submodules by concern.
"""

# Re-export main public API
from prellm.trace.models import (
    TraceStep,
    TraceRecorder,
    get_current_trace,
    set_current_trace,
)
from prellm.trace.markdown import generate_markdown
from prellm.trace.terminal import generate_terminal_output

# Attach methods to TraceRecorder for backward compatibility
TraceRecorder.to_markdown = lambda self: generate_markdown(self)
TraceRecorder.to_stdout = lambda self: generate_terminal_output(self)

__all__ = [
    "TraceStep",
    "TraceRecorder",
    "get_current_trace",
    "set_current_trace",
    "generate_markdown",
    "generate_terminal_output",
]
