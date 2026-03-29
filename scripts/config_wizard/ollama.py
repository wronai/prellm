"""Ollama diagnostics and model management."""

from __future__ import annotations

import subprocess
import urllib.error
import urllib.request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config_wizard.ui import ok, warn, fail, info, ask_yn


def to_ollama_full(name: str) -> str:
    """Convert model name to full ollama/ prefix format."""
    return name if name.startswith("ollama/") else f"ollama/{name}"


def strip_ollama_prefix(name: str) -> str:
    """Strip ollama/ prefix from model name."""
    return name[len("ollama/") :] if name.startswith("ollama/") else name


def fetch_ollama_models(base_url: str) -> list[str] | None:
    """Return list of installed Ollama models or None if unreachable."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            import json

            data = json.loads(resp.read())
            return [m.get("name", "").strip() for m in data.get("models", []) if m.get("name")]
    except urllib.error.URLError:
        return None
    except Exception:
        return None


def check_ollama(base_url: str) -> list[str] | None:
    """Check if Ollama is reachable and list available models."""
    from config_wizard.ui import ok, warn, fail, info

    print()
    print(f"  Checking Ollama at {base_url} ...")
    models = fetch_ollama_models(base_url)
    if models is None:
        fail(f"Ollama not reachable at {base_url}")
        info("Start with: ollama serve")
        return None
    if models:
        ok(f"Ollama running — {len(models)} models: {', '.join(models[:5])}")
    else:
        warn("Ollama running but no models pulled. Run: ollama pull qwen2.5:3b")
    return models


def build_ollama_options(installed_raw: list[str], recommended: list[str]) -> list[tuple[str, str]]:
    """Build list of Ollama model options with installation status."""
    options: list[tuple[str, str]] = []
    seen: set[str] = set()

    for model in installed_raw:
        full = to_ollama_full(model)
        options.append((f"{full} (installed)", full))
        seen.add(full)

    for model in recommended:
        full = to_ollama_full(model)
        if full not in seen:
            options.append((f"{full} (not installed)", full))
            seen.add(full)

    options.append(("other (specify)", "other"))
    return options


def option_index_for_value(options: list[tuple[str, str]], value: str, default: int = 0) -> int:
    """Find index of option with given value."""
    for idx, opt in enumerate(options):
        if opt[1] == value:
            return idx
    return default


def install_ollama_model(raw: str) -> bool:
    """Attempt to install an Ollama model via CLI."""
    from config_wizard.ui import ok, fail

    try:
        result = subprocess.run(["ollama", "pull", raw], check=False)
    except FileNotFoundError:
        fail("Ollama CLI not found in PATH")
        return False

    if result.returncode == 0:
        ok(f"Installed: {raw}")
        return True
    fail(f"Failed to install: {raw}")
    return False


def validate_ollama_model(base_url: str, model: str, installed_raw: list[str], reachable: bool) -> bool:
    """Validate Ollama model is installed, offer to install if not."""
    from config_wizard.ui import ok, warn, info, ask_yn

    if not model.startswith("ollama/") or not reachable:
        if model.startswith("ollama/") and not reachable:
            warn("Ollama not reachable; skipping model validation")
        return True
    raw = strip_ollama_prefix(model)
    if raw in installed_raw:
        ok(f"Ollama model installed: {model}")
        return True
    warn(f"Ollama model not installed: {model}")
    info(f"Install with: ollama pull {raw}")
    if ask_yn("Install now with `ollama pull`?", default=False):
        if install_ollama_model(raw):
            installed_raw.append(raw)
            return True
    return ask_yn("Continue anyway?", default=True)
