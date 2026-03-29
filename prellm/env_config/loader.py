"""Environment file loading utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prellm.env_config.models import EnvConfig

logger = logging.getLogger("prellm.env_config")

try:
    from getv import EnvStore
    _HAS_GETV = True
except ImportError:
    _HAS_GETV = False


def _load_getv_defaults() -> None:
    """Load getv app defaults if available."""
    if not _HAS_GETV:
        return
    try:
        from getv import AppDefaults
        from getv.integrations.pydantic_env import load_profile_into_env
        defaults = AppDefaults("prellm")
        llm_profile = defaults.get("llm")
        if llm_profile:
            load_profile_into_env("llm", llm_profile)
            logger.debug(f"Loaded getv default LLM profile: {llm_profile}")
    except Exception:
        pass


def _get_env_candidates(path: str | Path | None) -> list[Path]:
    """Get list of candidate .env files to load."""
    if path:
        return [Path(path)]
    return [Path(".env"), Path.home() / ".prellm" / ".env"]


def _load_env_file_with_getv(candidate: Path) -> bool:
    """Load .env file using getv EnvStore. Returns True if successful."""
    if not _HAS_GETV:
        return False
    try:
        store = EnvStore(candidate, auto_create=False)
        for key, value in store.items():
            if key and value and key not in os.environ:
                os.environ[key] = value
        return True
    except Exception:
        return False


def _parse_env_line(line: str) -> tuple[str, str] | None:
    """Parse a single .env line. Returns (key, value) or None if invalid."""
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, _, value = line.partition("=")
    key = key.strip()
    value = value.strip().strip("'\"")
    if key and value:
        return key, value
    return None


def _load_env_file_manually(candidate: Path) -> None:
    """Load .env file using manual parsing."""
    try:
        with open(candidate) as f:
            for line in f:
                parsed = _parse_env_line(line)
                if parsed:
                    key, value = parsed
                    if key not in os.environ:
                        os.environ[key] = value
    except Exception:
        pass


def load_dotenv_if_available(path: str | Path | None = None) -> None:
    """Load .env file if it exists. No dependency on python-dotenv."""
    _load_getv_defaults()
    for candidate in _get_env_candidates(path):
        if candidate and candidate.is_file():
            logger.debug(f"Loading .env from {candidate}")
            if _load_env_file_with_getv(candidate):
                return
            _load_env_file_manually(candidate)
            return


def get_env_config(dotenv_path: str | Path | None = None) -> "EnvConfig":
    """Read all config from environment variables (LiteLLM-compatible)."""
    from prellm.env_config.models import EnvConfig
    from prellm.env_config.constants import PROVIDER_KEY_MAP

    load_dotenv_if_available(dotenv_path)

    # Resolve providers
    providers: dict[str, dict[str, Any]] = {}
    for name, info in PROVIDER_KEY_MAP.items():
        key = os.getenv(info["key_var"], "") if info["key_var"] else ""
        base = os.getenv(info["base_var"], info["default_base"])
        providers[name] = {
            "has_key": bool(key),
            "key_var": info["key_var"],
            "base_url": base,
        }

    # Parse fallbacks
    fallback_str = os.getenv("PRELLM_FALLBACKS", "")
    fallbacks = [f.strip() for f in fallback_str.split(",") if f.strip()] if fallback_str else []

    # Parse budget
    budget_str = os.getenv("PRELLM_MONTHLY_BUDGET", "")
    monthly_budget = float(budget_str) if budget_str else None

    return EnvConfig(
        master_key=os.getenv("LITELLM_MASTER_KEY", None) or None,
        small_model=os.getenv("PRELLM_SMALL_DEFAULT", os.getenv("SMALL_MODEL", "ollama/qwen2.5:3b")),
        large_model=os.getenv("PRELLM_LARGE_DEFAULT", os.getenv("LARGE_MODEL", "gpt-4o-mini")),
        strategy=os.getenv("PRELLM_STRATEGY", "auto"),
        fallbacks=fallbacks,
        host=os.getenv("PRELLM_HOST", "0.0.0.0"),
        port=int(os.getenv("PRELLM_PORT", "8080")),
        config_path=os.getenv("PRELLM_CONFIG", None) or None,
        log_level=os.getenv("PRELLM_LOG_LEVEL", "info"),
        monthly_budget=monthly_budget,
        max_tokens=int(os.getenv("PRELLM_MAX_TOKENS", "4096")),
        timeout=int(os.getenv("PRELLM_TIMEOUT", "30")),
        providers=providers,
    )
