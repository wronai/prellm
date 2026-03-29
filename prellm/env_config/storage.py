"""Env file storage helpers for environment configuration commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from getv import EnvStore as _EnvStore
except ImportError:
    _EnvStore = None


def _get_env_store_class() -> Any | None:
    """Return the optional getv EnvStore implementation if available."""
    return _EnvStore


def _write_with_env_store(env_store: Any, path: Path, entries: dict[str, str]) -> None:
    """Persist env entries using getv's EnvStore backend."""
    store = env_store(path)
    store.update(entries)
    store.save()


def _read_plain_env_lines(path: Path, entries: dict[str, str]) -> tuple[list[str], set[str]]:
    """Read a plain .env file while preserving comment and unknown lines."""
    existing_lines: list[str] = []
    existing_keys: set[str] = set()

    if not path.is_file():
        return existing_lines, existing_keys

    with open(path) as f:
        for line in f:
            raw = line.rstrip("\n")
            stripped = raw.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key, _, _ = stripped.partition("=")
                key = key.strip()
                if key in entries:
                    existing_lines.append(f"{key}={entries[key]}")
                    existing_keys.add(key)
                    continue
            existing_lines.append(raw)

    return existing_lines, existing_keys


def _render_plain_env_content(existing_lines: list[str], comments: list[str] | None = None) -> str:
    """Render the final .env file content for the plain-file backend."""
    rendered_lines: list[str] = []
    if comments:
        for comment in comments:
            rendered_lines.append(f"# {comment}")
        if existing_lines and not existing_lines[0].startswith("#"):
            rendered_lines.append("")

    rendered_lines.extend(existing_lines)
    return "\n".join(rendered_lines) + "\n"


def _write_plain_env_file(path: Path, entries: dict[str, str], comments: list[str] | None = None) -> None:
    """Persist env entries using a plain-text .env file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines, existing_keys = _read_plain_env_lines(path, entries)

    for key, value in entries.items():
        if key not in existing_keys:
            existing_lines.append(f"{key}={value}")

    content = _render_plain_env_content(existing_lines, comments)
    with open(path, "w") as f:
        f.write(content)


def write_env_file(path: Path, entries: dict[str, str], comments: list[str] | None = None) -> None:
    """Write entries to a .env file, preserving existing comments."""
    env_store = _get_env_store_class()
    if env_store is not None:
        _write_with_env_store(env_store, path, entries)
        return

    _write_plain_env_file(path, entries, comments)
