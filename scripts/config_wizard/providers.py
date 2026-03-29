"""Provider configuration handlers."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config_wizard.ui import ok, ask, ask_choice
    from config_wizard.ollama import check_ollama, build_ollama_options, option_index_for_value, validate_ollama_model


def mask_key(key: str) -> str:
    """Mask API key for display."""
    if len(key) > 12:
        return key[:8] + "..." + key[-4:]
    return key[:4] + "..."


def check_api_key_format(provider: str, key: str) -> bool:
    """Validate API key format."""
    from config_wizard.ui import ok, warn

    patterns = {
        "openai": r"^sk-[A-Za-z0-9_-]{20,}$",
        "anthropic": r"^sk-ant-[A-Za-z0-9_-]{20,}$",
        "groq": r"^gsk_[A-Za-z0-9_-]{20,}$",
        "openrouter": r"^sk-or-v1-[A-Za-z0-9_-]{20,}$",
        "mistral": r"^[A-Za-z0-9_-]{20,}$",
    }
    pattern = patterns.get(provider)
    if not pattern:
        return True
    if re.match(pattern, key):
        ok(f"{provider} API key format valid")
        return True
    else:
        warn(f"{provider} API key format looks unusual (expected: {pattern})")
        return True  # Don't block, just warn


def check_port_available(host: str, port: int) -> bool:
    """Check if port is available for the server."""
    import socket
    from config_wizard.ui import ok, warn

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host if host != "0.0.0.0" else "127.0.0.1", port))
            if result == 0:
                warn(f"Port {port} is already in use")
                return False
            else:
                ok(f"Port {port} is available")
                return True
    except Exception:
        return True


def _configure_openai(config: dict) -> None:
    """Configure OpenAI provider."""
    from config_wizard.ui import ask, ask_choice, ok

    api_key = ask("OpenAI API Key", required=True)
    config["OPENAI_API_KEY"] = api_key
    check_api_key_format("openai", api_key)

    openai_models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "other"]
    model = ask_choice("Select model:", openai_models, default=0)
    if model == "other":
        model = ask("Enter model name:")
    config["PRELLM_LARGE_DEFAULT"] = model


def _configure_anthropic(config: dict) -> None:
    """Configure Anthropic provider."""
    from config_wizard.ui import ask, ask_choice, ok

    api_key = ask("Anthropic API Key", required=True)
    config["ANTHROPIC_API_KEY"] = api_key
    check_api_key_format("anthropic", api_key)

    anthropic_models = [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-haiku-4-20250514",
        "other",
    ]
    model = ask_choice("Select model:", anthropic_models, default=0)
    if model == "other":
        model = ask("Enter model name (e.g., claude-3-5-sonnet-20241022):")
    config["PRELLM_LARGE_DEFAULT"] = f"anthropic/{model}"


def _configure_groq(config: dict) -> None:
    """Configure Groq provider."""
    from config_wizard.ui import ask, ask_choice, ok

    api_key = ask("Groq API Key", required=True)
    config["GROQ_API_KEY"] = api_key
    check_api_key_format("groq", api_key)

    groq_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "other",
    ]
    model = ask_choice("Select model:", groq_models, default=0)
    if model == "other":
        model = ask("Enter model name:")
    config["PRELLM_LARGE_DEFAULT"] = f"groq/{model}"


def _configure_openrouter(config: dict) -> None:
    """Configure OpenRouter provider."""
    from config_wizard.ui import ask, ask_choice, ok

    api_key = ask("OpenRouter API Key", required=True)
    config["OPENROUTER_API_KEY"] = api_key
    check_api_key_format("openrouter", api_key)

    openrouter_models = [
        "qwen/qwen3-vl-32b-instruct",
        "deepseek/deepseek-chat",
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "meta-llama/llama-3.3-70b-instruct",
        "google/gemini-pro-1.5",
        "other",
    ]
    model = ask_choice("Select model:", openrouter_models, default=0)
    if model == "other":
        model = ask("Enter model name (e.g., qwen/qwen3-vl-32b-instruct):")
    config["PRELLM_LARGE_DEFAULT"] = f"openrouter/{model}"


def _configure_ollama_large(config: dict) -> None:
    """Configure Ollama large model."""
    from config_wizard.ui import ask, ask_choice
    from config_wizard.ollama import check_ollama, build_ollama_options, option_index_for_value, validate_ollama_model, to_ollama_full

    if not config.get("OLLAMA_API_BASE"):
        ollama_base = ask("Ollama API base URL", "http://localhost:11434")
        config["OLLAMA_API_BASE"] = ollama_base
        installed_raw = check_ollama(ollama_base)
        ollama_reachable = installed_raw is not None
        installed_raw = installed_raw or []
    else:
        installed_raw = check_ollama(config["OLLAMA_API_BASE"]) or []
        ollama_reachable = True

    large_recommended = [
        "llama3:latest",
        "qwen2.5:7b",
        "qwen2.5:14b",
        "mistral:7b",
        "codellama:13b",
    ]
    large_options = build_ollama_options(installed_raw, large_recommended)
    large_default = option_index_for_value(large_options, "ollama/qwen2.5:7b", default=0)

    while True:
        model_choice = ask_choice("Select large model:", large_options, default=large_default)
        if model_choice == "other":
            model = ask("Enter model name (e.g., llama3:latest)")
            model_full = to_ollama_full(model)
        else:
            model_full = model_choice

        config["PRELLM_LARGE_DEFAULT"] = model_full
        if validate_ollama_model(config["OLLAMA_API_BASE"], model_full, installed_raw, ollama_reachable):
            break


def _configure_other_provider(config: dict) -> None:
    """Configure other provider."""
    from config_wizard.ui import ask

    config["PRELLM_LARGE_DEFAULT"] = ask(
        "Enter full model name (e.g., mistral/mistral-small-latest):", required=True
    )
    key_name = ask("API key env var name (e.g., MISTRAL_API_KEY, or empty to skip)")
    if key_name:
        key_val = ask(f"{key_name}", required=True)
        config[key_name] = key_val
