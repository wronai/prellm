"""Execution trace recorder — generates readable markdown documentation of the decision path.

This module is a thin wrapper around prellm.trace package for backward compatibility.
All implementation has been moved to the prellm.trace package.

Records each step of the preLLM pipeline (classification, structure extraction,
domain matching, LLM calls, etc.) with inputs, outputs, timing, and rationale.

Output:
  - Markdown file saved to .prellm/ in the working directory
  - Printed to stdout for immediate inspection
"""

from __future__ import annotations

# Re-export all public API from the new package
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
]
