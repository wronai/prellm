"""Environment configuration — LiteLLM-compatible .env loading.

This module is a thin wrapper around prellm.env_config package for backward compatibility.
All implementation has been moved to the prellm.env_config package.

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

# Re-export all public API from the new package
from prellm.env_config.models import EnvConfig
from prellm.env_config.loader import load_dotenv_if_available, get_env_config
from prellm.env_config.providers import check_providers, check_providers_live
from prellm.env_config.commands import config_set, config_get, config_list
from prellm.env_config.utils import resolve_alias, mask_value
from prellm.env_config.constants import PROVIDER_KEY_MAP, PROVIDER_MODEL_PREFIX, CONFIG_ALIASES, SECRET_KEYS

__all__ = [
    "EnvConfig",
    "load_dotenv_if_available",
    "get_env_config",
    "check_providers",
    "check_providers_live",
    "config_set",
    "config_get",
    "config_list",
    "resolve_alias",
    "mask_value",
    "PROVIDER_KEY_MAP",
    "PROVIDER_MODEL_PREFIX",
    "CONFIG_ALIASES",
    "SECRET_KEYS",
]
