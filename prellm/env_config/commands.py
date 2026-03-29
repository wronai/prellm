"""Config commands for CLI (set, get, list operations)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prellm.env_config.models import EnvConfig


def _resolve_config_path(global_: bool = False) -> Path:
    """Resolve config .env file path."""
    if global_:
        return Path.home() / ".prellm" / ".env"
    return Path(".env")


def _read_env_file(path: Path) -> dict[str, str]:
    """Read .env file into ordered dict."""
    try:
        from getv import EnvStore
        _HAS_GETV = True
    except ImportError:
        _HAS_GETV = False

    if _HAS_GETV:
        if not path.is_file():
            return {}
        return EnvStore(path, auto_create=False).as_dict()

    entries: dict[str, str] = {}
    if not path.is_file():
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key:
                entries[key] = value
    return entries


def _write_env_file(path: Path, entries: dict[str, str], comments: list[str] | None = None) -> None:
    """Write entries to .env file, preserving existing comments."""
    try:
        from getv import EnvStore
        _HAS_GETV = True
    except ImportError:
        _HAS_GETV = False

    if _HAS_GETV:
        store = EnvStore(path)
        store.update(entries)
        store.save()
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines: list[str] = []
    existing_keys: set[str] = set()
    if path.is_file():
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
    for key, value in entries.items():
        if key not in existing_keys:
            existing_lines.append(f"{key}={value}")
    with open(path, "w") as f:
        if comments:
            for c in comments:
                f.write(f"# {c}\n")
            if existing_lines and not existing_lines[0].startswith("#"):
                f.write("\n")
        f.write("\n".join(existing_lines) + "\n")


def config_set(key: str, value: str, global_: bool = False) -> tuple[str, Path]:
    """Set a config value persistently in .env file."""
    from prellm.env_config.utils import resolve_alias

    env_var = resolve_alias(key)
    path = _resolve_config_path(global_)
    entries = _read_env_file(path)
    entries[env_var] = value
    _write_env_file(path, entries)
    os.environ[env_var] = value
    return env_var, path


def config_get(key: str, global_: bool = False) -> tuple[str, str | None, str]:
    """Get a config value from env var → project .env → global .env."""
    from prellm.env_config.utils import resolve_alias

    env_var = resolve_alias(key)

    val = os.environ.get(env_var)
    if val:
        return env_var, val, "environment"

    project_env = _read_env_file(Path(".env"))
    if env_var in project_env:
        return env_var, project_env[env_var], ".env (project)"

    global_env = _read_env_file(Path.home() / ".prellm" / ".env")
    if env_var in global_env:
        return env_var, global_env[env_var], "~/.prellm/.env (global)"

    return env_var, None, "not set"


def config_list(global_: bool = False, show_secrets: bool = False) -> dict[str, dict[str, str]]:
    """List all config values from .env files and environment."""
    from prellm.env_config.constants import CONFIG_ALIASES, SECRET_KEYS
    from prellm.env_config.utils import mask_value

    result: dict[str, dict[str, str]] = {}
    global_env = _read_env_file(Path.home() / ".prellm" / ".env")
    project_env = _read_env_file(Path(".env"))

    reverse_alias: dict[str, str] = {}
    for alias, env_var in CONFIG_ALIASES.items():
        if env_var not in reverse_alias:
            reverse_alias[env_var] = alias

    all_vars = set(CONFIG_ALIASES.values()) | set(global_env.keys()) | set(project_env.keys())
    for var in sorted(all_vars):
        val = os.environ.get(var) or project_env.get(var) or global_env.get(var)
        if val is None:
            continue
        source = "environment"
        if var in project_env:
            source = ".env"
        elif var in global_env:
            source = "~/.prellm/.env"
        result[var] = {
            "value": val if show_secrets else mask_value(var, val),
            "raw_value": val,
            "source": source,
            "alias": reverse_alias.get(var, ""),
        }

    return result
