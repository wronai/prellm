"""Utility functions for environment configuration."""

from __future__ import annotations

try:
    from getv.security import mask_value as _getv_mask, is_sensitive_key as _getv_is_sensitive
    _HAS_GETV = True
except ImportError:
    _HAS_GETV = False

from prellm.env_config.constants import CONFIG_ALIASES, SECRET_KEYS


def resolve_alias(key: str) -> str:
    """Resolve a user-friendly alias to an env var name."""
    if key in CONFIG_ALIASES:
        return CONFIG_ALIASES[key]
    upper = key.upper().replace("-", "_")
    if upper in SECRET_KEYS or upper.startswith("PRELLM_") or upper.startswith("OPENAI_") or upper.startswith("OLLAMA_"):
        return upper
    if key == key.upper() and "_" in key:
        return key
    return CONFIG_ALIASES.get(key, key.upper().replace("-", "_"))


def mask_value(key: str, value: str) -> str:
    """Mask secret values for display."""
    if _HAS_GETV:
        if _getv_is_sensitive(key):
            return _getv_mask(value, visible_chars=4)
        return value
    if key in SECRET_KEYS and len(value) > 8:
        return value[:4] + "..." + value[-4:]
    return value
