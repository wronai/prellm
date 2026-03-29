"""Trace output utilities — helper functions for trace formatting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
    """Format a value for display in the decision tree."""
    if isinstance(val, dict):
        try:
            return json.dumps(val, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(val)
    elif isinstance(val, str):
        return val.replace("\n", " ")
    return str(val)


def _extract_prompt_text(composed: Any) -> str:
    """Extract the prompt text from a composed_prompt output."""
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
