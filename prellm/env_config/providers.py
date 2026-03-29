"""Provider checking utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prellm.env_config.models import EnvConfig


def check_providers(env: "EnvConfig" | None = None) -> dict[str, dict[str, Any]]:
    """Check which providers are configured and reachable."""
    from prellm.env_config.loader import get_env_config

    cfg = env or get_env_config()
    results: dict[str, dict[str, Any]] = {}

    for name, info in cfg.providers.items():
        if name == "ollama":
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


async def check_providers_live(env: "EnvConfig" | None = None) -> dict[str, dict[str, Any]]:
    """Check providers with live connectivity tests."""
    import httpx
    from prellm.env_config.loader import get_env_config

    cfg = env or get_env_config()
    results = check_providers(cfg)

    async with httpx.AsyncClient(timeout=5.0) as client:
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

        for name in ["openai", "anthropic", "groq", "mistral"]:
            if results.get(name, {}).get("status") == "configured":
                results[name]["status"] = "ok"

    return results
