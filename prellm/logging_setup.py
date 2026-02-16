"""Centralized nfo logging configuration for preLLM.

Provides markdown-formatted terminal output so users can see exactly
what happens during query preprocessing, LLM calls, and pipeline execution.

Usage:
    from prellm.logging_setup import setup_logging, get_logger

    setup_logging("DEBUG")          # call once at startup
    logger = get_logger("prellm")   # per-module logger
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from nfo.configure import configure
from nfo.logger import Logger
from nfo.sinks import MarkdownSink
from nfo.terminal import TerminalSink

_logger: Optional[Logger] = None


def setup_logging(
    level: str = "INFO",
    markdown_file: str | None = None,
    terminal_format: str = "markdown",
) -> Logger:
    """Initialize nfo logging for the entire preLLM project.

    Args:
        level: Log level — DEBUG, INFO, WARNING, ERROR.
        markdown_file: Optional path to write markdown log file (e.g. "prellm.log.md").
        terminal_format: Terminal sink format — "markdown", "color", "toon", "ascii".

    Returns:
        Configured nfo Logger instance.
    """
    global _logger

    if _logger is not None:
        return _logger

    sinks = [
        TerminalSink(
            format=terminal_format,
            stream=sys.stderr,
            show_args=True,
            show_return=True,
            show_duration=True,
            show_traceback=True,
        ),
    ]

    if markdown_file:
        sinks.append(MarkdownSink(file_path=markdown_file))

    _logger = configure(
        name="prellm",
        level=level.upper(),
        sinks=sinks,
        bridge_stdlib=True,
        propagate_stdlib=False,
        env_prefix="PRELLM_NFO_",
        version=_get_version(),
        force=True,
    )

    # Suppress noisy litellm provider warnings and "Provider List" stdout spam
    import logging as _logging
    for _name in ("LiteLLM", "litellm", "httpx", "httpcore"):
        _logging.getLogger(_name).setLevel(_logging.WARNING)
    try:
        import litellm
        litellm.suppress_debug_info = True
    except ImportError:
        pass

    return _logger


def get_logger(name: str = "prellm") -> Logger:
    """Get or create the nfo logger.

    If setup_logging() hasn't been called yet, initializes with defaults
    from environment (PRELLM_LOG_LEVEL).
    """
    global _logger
    if _logger is None:
        level = os.getenv("PRELLM_LOG_LEVEL", "INFO").upper()
        md_file = os.getenv("PRELLM_NFO_LOG_FILE", None)
        fmt = os.getenv("PRELLM_NFO_FORMAT", "markdown")
        setup_logging(level=level, markdown_file=md_file, terminal_format=fmt)
    return _logger  # type: ignore[return-value]


def _get_version() -> str:
    """Read preLLM version without circular imports."""
    try:
        from prellm import __version__
        return __version__
    except Exception:
        return "unknown"
