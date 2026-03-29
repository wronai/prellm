"""Environment configuration constants and mappings."""

from __future__ import annotations

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

# Provider → LiteLLM model prefix mapping
PROVIDER_MODEL_PREFIX: dict[str, str] = {
    "openai": "",
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
    "moonshot": "openrouter/",
}

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
    "model": "PRELLM_LARGE_DEFAULT",
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
SECRET_KEYS = {
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY",
    "AZURE_API_KEY", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "TOGETHERAI_API_KEY",
    "GEMINI_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "MOONSHOT_API_KEY",
    "LITELLM_MASTER_KEY",
}
