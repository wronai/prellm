"""Environment configuration — LiteLLM-compatible .env loading.

Reads standard LiteLLM env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
plus preLLM-specific vars (PRELLM_SMALL_DEFAULT, PRELLM_LARGE_DEFAULT).

Usage:
    from prellm.env_config import get_env_config, check_providers

    cfg = get_env_config()
    print(cfg.small_model)   # "ollama/qwen2.5:3b"
    print(cfg.large_model)   # "gpt-4o-mini"
    print(cfg.master_key)    # "sk-prellm-1234" or None

    status = check_providers()
    # {"ollama": {"status": "ok", ...}, "anthropic": {"status": "no_key"}, ...}
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from getv import EnvStore
    from getv.security import mask_value as _getv_mask, is_sensitive_key as _getv_is_sensitive
    _HAS_GETV = True
except ImportError:
    _HAS_GETV = False

logger = logging.getLogger("prellm.env_config")

# LiteLLM-compatible provider env vars
PROVIDER_KEY_MAP: dict[str, dict[str, str]] = {
    "openai": {"key_var": "OPENAI_API_KEY", "base_var": "OPENAI_BASE_URL", "default_base": "https://api.openai.com/v1"},
    "anthropic": {"key_var": "ANTHROPIC_API_KEY", "base_var": "ANTHROPIC_BASE_URL", "default_base": "https://api.anthropic.com/v1"},
    "groq": {"key_var": "GROQ_API_KEY", "base_var": "GROQ_BASE_URL", "default_base": "https://api.groq.com/openai/v1"},
    "mistral": {"key_var": "MISTRAL_API_KEY", "base_var": "MISTRAL_BASE_URL", "default_base": "https://api.mistral.ai/v1"},
    "azure": {"key_var": "AZURE_API_KEY", "base_var": "AZURE_API_BASE", "default_base": ""},
    "ollama": {"key_var": "", "base_var": "OLLAMA_API_BASE", "default_base": "http://localhost:11434"},
    "openrouter": {"key_var": "OPENROUTER_API_KEY", "base_var": "OPENROUTER_API_BASE", "default_base": "https://openrouter.ai/api/v1"},
    "deepseek": {"key_var": "DEEPSEEK_API_KEY", "base_var": "DEEPSEEK_API_BASE", "default_base": "https://api.deepseek.com/v1"},
    "together_ai": {"key_var": "TOGETHERAI_API_KEY", "base_var": "TOGETHERAI_API_BASE", "default_base": "https://api.together.xyz/v1"},
    "gemini": {"key_var": "GEMINI_API_KEY", "base_var": "GEMINI_API_BASE", "default_base": "https://generativelanguage.googleapis.com/v1beta"},
    "bedrock": {"key_var": "AWS_ACCESS_KEY_ID", "base_var": "", "default_base": ""},
    "moonshot": {"key_var": "MOONSHOT_API_KEY", "base_var": "MOONSHOT_API_BASE", "default_base": "https://api.moonshot.cn/v1"},
}

# Provider → LiteLLM model prefix mapping (for model string resolution)
PROVIDER_MODEL_PREFIX: dict[str, str] = {
    "openai": "",             # gpt-4o-mini (no prefix needed)
    "anthropic": "anthropic/",
    "groq": "groq/",
    "mistral": "mistral/",
    "azure": "azure/",
    "ollama": "ollama/",
    "openrouter": "openrouter/",
    "deepseek": "deepseek/",
    "together_ai": "together_ai/",
    "gemini": "gemini/",
    "bedrock": "bedrock/",
    "moonshot": "openrouter/",  # moonshot models via openrouter: openrouter/moonshotai/kimi-k2.5
}


@dataclass
class EnvConfig:
    """Resolved environment configuration."""
    # Auth
    master_key: str | None = None

    # Models
    small_model: str = "ollama/qwen2.5:3b"
    large_model: str = "gpt-4o-mini"
    strategy: str = "auto"
    fallbacks: list[str] = field(default_factory=list)

    # Server
    host: str = "0.0.0.0"
    port: int = 8080

    # Config
    config_path: str | None = None
    log_level: str = "info"

    # Budget
    monthly_budget: float | None = None
    max_tokens: int = 4096
    timeout: int = 30

    # Providers (resolved)
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_dotenv_if_available(path: str | Path | None = None) -> None:
    """Load .env file if it exists. No dependency on python-dotenv — just basic parsing."""
    candidates = [path] if path else [".env", Path.home() / ".prellm" / ".env"]

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            logger.debug(f"Loading .env from {candidate}")
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    if key and value and key not in os.environ:
                        os.environ[key] = value
            return


def get_env_config(dotenv_path: str | Path | None = None) -> EnvConfig:
    """Read all config from environment variables (LiteLLM-compatible).

    Priority: CLI args > env vars > .env file > defaults
    """
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


def check_providers(env: EnvConfig | None = None) -> dict[str, dict[str, Any]]:
    """Check which providers are configured and reachable.

    Returns dict of provider_name → {status, key_var, base_url, detail}.
    """
    cfg = env or get_env_config()
    results: dict[str, dict[str, Any]] = {}

    for name, info in cfg.providers.items():
        if name == "ollama":
            # Ollama: check base URL reachability
            base = info["base_url"]
            results[name] = {
                "status": "configured",
                "base_url": base,
                "detail": f"Base URL: {base} (no key required)",
            }
        elif info["has_key"]:
            results[name] = {
                "status": "configured",
                "key_var": info["key_var"],
                "base_url": info["base_url"],
                "detail": f"{info['key_var']} set",
            }
        else:
            results[name] = {
                "status": "no_key",
                "key_var": info["key_var"],
                "detail": f"{info['key_var']} not set (skip)",
            }

    return results


# ============================================================
# Persistent config management (for CLI `prellm config set/get`)
# ============================================================

# Aliases: user-friendly key → env var name
CONFIG_ALIASES: dict[str, str] = {
    # API keys
    "openai-key": "OPENAI_API_KEY",
    "anthropic-key": "ANTHROPIC_API_KEY",
    "groq-key": "GROQ_API_KEY",
    "mistral-key": "MISTRAL_API_KEY",
    "azure-key": "AZURE_API_KEY",
    "openrouter-key": "OPENROUTER_API_KEY",
    "deepseek-key": "DEEPSEEK_API_KEY",
    "together-key": "TOGETHERAI_API_KEY",
    "gemini-key": "GEMINI_API_KEY",
    "aws-key": "AWS_ACCESS_KEY_ID",
    "aws-secret": "AWS_SECRET_ACCESS_KEY",
    "moonshot-key": "MOONSHOT_API_KEY",
    "master-key": "LITELLM_MASTER_KEY",
    # Models
    "small-model": "PRELLM_SMALL_DEFAULT",
    "large-model": "PRELLM_LARGE_DEFAULT",
    "model": "PRELLM_LARGE_DEFAULT",          # shortcut alias
    # Settings
    "strategy": "PRELLM_STRATEGY",
    "host": "PRELLM_HOST",
    "port": "PRELLM_PORT",
    "log-level": "PRELLM_LOG_LEVEL",
    "budget": "PRELLM_MONTHLY_BUDGET",
    "max-tokens": "PRELLM_MAX_TOKENS",
    "timeout": "PRELLM_TIMEOUT",
    "fallbacks": "PRELLM_FALLBACKS",
    # Base URLs
    "openai-base": "OPENAI_BASE_URL",
    "ollama-base": "OLLAMA_API_BASE",
    "openrouter-base": "OPENROUTER_API_BASE",
    "azure-base": "AZURE_API_BASE",
}

# Keys that should be masked in output
_SECRET_KEYS = {
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY",
    "AZURE_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "TOGETHERAI_API_KEY",
    "GEMINI_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "MOONSHOT_API_KEY",
    "LITELLM_MASTER_KEY",
}


def _resolve_config_path(global_: bool = False) -> Path:
    """Resolve config .env file path.

    global_=True  → ~/.prellm/.env  (user-wide)
    global_=False → .env             (project-local, preferred)
    """
    if global_:
        p = Path.home() / ".prellm" / ".env"
    else:
        p = Path(".env")
    return p


def _read_env_file(path: Path) -> dict[str, str]:
    """Read .env file into ordered dict, preserving comments and order.

    Delegates to getv.EnvStore when available.
    """
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
    """Write entries to .env file, preserving existing comments.

    Delegates to getv.EnvStore when available.
    """
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


def resolve_alias(key: str) -> str:
    """Resolve a user-friendly alias to an env var name.

    Accepts: "openrouter-key", "OPENROUTER_API_KEY", "openrouter_api_key"
    """
    if key in CONFIG_ALIASES:
        return CONFIG_ALIASES[key]
    # Try uppercase
    upper = key.upper().replace("-", "_")
    if upper in _SECRET_KEYS or upper.startswith("PRELLM_") or upper.startswith("OPENAI_") or upper.startswith("OLLAMA_"):
        return upper
    # Check if it's already a valid env var name
    if key == key.upper() and "_" in key:
        return key
    return CONFIG_ALIASES.get(key, key.upper().replace("-", "_"))


def mask_value(key: str, value: str) -> str:
    """Mask secret values for display.

    Delegates to getv.security when available.
    """
    if _HAS_GETV:
        if _getv_is_sensitive(key):
            return _getv_mask(value, visible_chars=4)
        return value
    if key in _SECRET_KEYS and len(value) > 8:
        return value[:4] + "..." + value[-4:]
    return value


def config_set(key: str, value: str, global_: bool = False) -> tuple[str, Path]:
    """Set a config value persistently in .env file.

    Returns (resolved_env_var_name, path_written_to).
    """
    env_var = resolve_alias(key)
    path = _resolve_config_path(global_)
    entries = _read_env_file(path)
    entries[env_var] = value
    _write_env_file(path, entries)
    # Also set in current process
    os.environ[env_var] = value
    return env_var, path


def config_get(key: str, global_: bool = False) -> tuple[str, str | None, str]:
    """Get a config value. Checks: env var → project .env → global .env → None.

    Returns (resolved_env_var_name, value_or_None, source).
    """
    env_var = resolve_alias(key)

    # 1. Current env
    val = os.environ.get(env_var)
    if val:
        return env_var, val, "environment"

    # 2. Project .env
    project_env = _read_env_file(Path(".env"))
    if env_var in project_env:
        return env_var, project_env[env_var], ".env (project)"

    # 3. Global .env
    global_env = _read_env_file(Path.home() / ".prellm" / ".env")
    if env_var in global_env:
        return env_var, global_env[env_var], "~/.prellm/.env (global)"

    return env_var, None, "not set"


def config_list(global_: bool = False, show_secrets: bool = False) -> dict[str, dict[str, str]]:
    """List all config values from .env files and environment.

    Returns dict of env_var → {value, source, alias, masked_value}.
    """
    result: dict[str, dict[str, str]] = {}

    # Collect from all sources
    global_env = _read_env_file(Path.home() / ".prellm" / ".env")
    project_env = _read_env_file(Path(".env"))

    # Build reverse alias map
    reverse_alias: dict[str, str] = {}
    for alias, env_var in CONFIG_ALIASES.items():
        if env_var not in reverse_alias:
            reverse_alias[env_var] = alias

    # All known vars
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


async def check_providers_live(env: EnvConfig | None = None) -> dict[str, dict[str, Any]]:
    """Check providers with live connectivity tests."""
    import httpx

    cfg = env or get_env_config()
    results = check_providers(cfg)

    async with httpx.AsyncClient(timeout=5.0) as client:
        # Test Ollama
        ollama_base = cfg.providers.get("ollama", {}).get("base_url", "http://localhost:11434")
        try:
            resp = await client.get(f"{ollama_base}/api/tags")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                results["ollama"]["status"] = "ok"
                results["ollama"]["models"] = model_names
                results["ollama"]["detail"] = f"{len(models)} models available"
            else:
                results["ollama"]["status"] = "error"
                results["ollama"]["detail"] = f"HTTP {resp.status_code}"
        except Exception as e:
            results["ollama"]["status"] = "unreachable"
            results["ollama"]["detail"] = str(e)

        # Test remote providers (just check API key format, no actual call)
        for name in ["openai", "anthropic", "groq", "mistral"]:
            if results.get(name, {}).get("status") == "configured":
                results[name]["status"] = "ok"

    return results
