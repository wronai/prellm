"""prellm/env_config package — Environment configuration management.

This package handles LiteLLM-compatible .env loading, provider configuration,
and persistent config management for the prellm CLI.
"""

from prellm.env_config.models import EnvConfig
from prellm.env_config.loader import load_dotenv_if_available, get_env_config
from prellm.env_config.providers import check_providers, check_providers_live
from prellm.env_config.commands import config_set, config_get, config_list
from prellm.env_config.utils import resolve_alias, mask_value, _resolve_config_path

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
    "_resolve_config_path",
]
