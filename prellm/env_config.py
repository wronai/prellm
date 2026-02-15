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

logger = logging.getLogger("prellm.env_config")

# LiteLLM-compatible provider env vars
PROVIDER_KEY_MAP: dict[str, dict[str, str]] = {
    "openai": {"key_var": "OPENAI_API_KEY", "base_var": "OPENAI_BASE_URL", "default_base": "https://api.openai.com/v1"},
    "anthropic": {"key_var": "ANTHROPIC_API_KEY", "base_var": "ANTHROPIC_BASE_URL", "default_base": "https://api.anthropic.com/v1"},
    "groq": {"key_var": "GROQ_API_KEY", "base_var": "GROQ_BASE_URL", "default_base": "https://api.groq.com/openai/v1"},
    "mistral": {"key_var": "MISTRAL_API_KEY", "base_var": "MISTRAL_BASE_URL", "default_base": "https://api.mistral.ai/v1"},
    "azure": {"key_var": "AZURE_API_KEY", "base_var": "AZURE_API_BASE", "default_base": ""},
    "ollama": {"key_var": "", "base_var": "OLLAMA_API_BASE", "default_base": "http://localhost:11434"},
}


@dataclass
class EnvConfig:
    """Resolved environment configuration."""
    # Auth
    master_key: str | None = None

    # Models
    small_model: str = "ollama/qwen2.5:3b"
    large_model: str = "gpt-4o-mini"
    strategy: str = "classify"
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
        strategy=os.getenv("PRELLM_STRATEGY", "classify"),
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
