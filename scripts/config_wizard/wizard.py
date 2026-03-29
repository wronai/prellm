"""Main wizard flow for preLLM configuration."""

from __future__ import annotations

import secrets
import sys
from datetime import datetime
from pathlib import Path

from config_wizard.ui import GREEN, BLUE, YELLOW, NC, ok, warn, info, ask, ask_yn, ask_choice
from config_wizard.ollama import check_ollama, build_ollama_options, option_index_for_value, validate_ollama_model
from config_wizard.providers import (
    mask_key,
    check_port_available,
    _configure_openai,
    _configure_anthropic,
    _configure_groq,
    _configure_openrouter,
    _configure_ollama_large,
    _configure_other_provider,
)


def _run_diagnostics() -> tuple[dict, list, bool]:
    """Run initial diagnostics and return config dict and installed models."""
    config: dict = {}

    # Small LLM setup
    print(f"{YELLOW}--- Step 1/6: Small LLM (Local Preprocessing) ---{NC}")
    print("The small model runs locally for fast query preprocessing.")
    print()

    ollama_base = ask("Ollama API base URL", "http://localhost:11434")
    config["OLLAMA_API_BASE"] = ollama_base
    installed_raw = check_ollama(ollama_base)
    ollama_reachable = installed_raw is not None
    installed_raw = installed_raw or []

    return config, installed_raw, ollama_reachable


def _configure_small_llm(config: dict, installed_raw: list, ollama_reachable: bool) -> None:
    """Configure small LLM settings."""
    small_recommended = [
        "qwen2.5:3b",
        "phi3:latest",
        "llama3.2:3b",
        "gemma:2b",
        "tinyllama:latest",
    ]
    small_options = build_ollama_options(installed_raw, small_recommended)
    small_default = option_index_for_value(small_options, "ollama/qwen2.5:3b", default=0)

    while True:
        small_choice = ask_choice("Select your small (local) model:", small_options, default=small_default)
        if small_choice == "other":
            config["PRELLM_SMALL_DEFAULT"] = ask("Enter small model (e.g., ollama/phi3:latest)")
        else:
            config["PRELLM_SMALL_DEFAULT"] = small_choice

        if validate_ollama_model(config["OLLAMA_API_BASE"], config["PRELLM_SMALL_DEFAULT"], installed_raw, ollama_reachable):
            break

    ok(f"Small model: {config['PRELLM_SMALL_DEFAULT']}")


def _configure_large_llm(config: dict) -> None:
    """Configure large LLM provider and model."""
    print()
    print(f"{YELLOW}--- Step 2/6: Large LLM (Remote Execution) ---{NC}")
    print("The large model handles final query execution.")
    print()

    providers = [
        "OpenAI (GPT-4o, GPT-4o-mini)",
        "Anthropic (Claude 3.5 Sonnet, Opus)",
        "Groq (fast, cheap - Llama, Mixtral)",
        "OpenRouter (unified API - Qwen, DeepSeek, etc.)",
        "Ollama (local large model — $0 cost)",
        "Other provider",
    ]

    provider = ask_choice("Select your large LLM provider:", providers, default=0)

    if "OpenAI" in provider:
        _configure_openai(config)
    elif "Anthropic" in provider:
        _configure_anthropic(config)
    elif "Groq" in provider:
        _configure_groq(config)
    elif "OpenRouter" in provider:
        _configure_openrouter(config)
    elif "Ollama" in provider:
        _configure_ollama_large(config)
    else:
        _configure_other_provider(config)

    ok(f"Large model: {config['PRELLM_LARGE_DEFAULT']}")


def _configure_strategy_and_server(config: dict) -> None:
    """Configure strategy and server settings."""
    # Strategy
    print()
    print(f"{YELLOW}--- Step 3/6: Default Strategy ---{NC}")
    strategies = [
        "classify - Classify and route query to appropriate handler",
        "structure - Add structure/JSON schema to query",
        "split - Break complex queries into sub-queries",
        "enrich - Add context and examples",
        "passthrough - No preprocessing (raw pass-through)",
    ]
    strategy = ask_choice("Select default preprocessing strategy:", strategies, default=0)
    config["PRELLM_STRATEGY"] = strategy.split(" - ")[0]
    ok(f"Strategy: {config['PRELLM_STRATEGY']}")

    # Server
    print()
    print(f"{YELLOW}--- Step 4/6: Server Settings ---{NC}")

    auth = ask_yn("Enable API authentication?", default=False)
    if auth:
        master_key = ask("Enter master API key (or leave empty for auto-generated)")
        if not master_key:
            master_key = "sk-" + secrets.token_urlsafe(32)
            ok(f"Generated key: {mask_key(master_key)}")
        config["LITELLM_MASTER_KEY"] = master_key
    else:
        config["LITELLM_MASTER_KEY"] = ""
        info("Auth disabled (dev mode). Set LITELLM_MASTER_KEY later for production.")

    server_host = ask("Server host", "0.0.0.0")
    config["PRELLM_HOST"] = server_host

    server_port = ask("Server port", "8080")
    config["PRELLM_PORT"] = server_port
    check_port_available(server_host, int(server_port))

    config_path = ask("YAML config file path (optional)", "")
    config["PRELLM_CONFIG"] = config_path


def _configure_budget_and_limits(config: dict) -> None:
    """Configure budget and limits."""
    print()
    print(f"{YELLOW}--- Step 5/6: Budget & Limits ---{NC}")

    budget = ask("Monthly budget in USD (empty = unlimited)", "")
    config["PRELLM_MONTHLY_BUDGET"] = budget

    max_tokens = ask("Max tokens per request", "4096")
    config["PRELLM_MAX_TOKENS"] = max_tokens

    timeout = ask("Request timeout in seconds", "30")
    config["PRELLM_TIMEOUT"] = timeout

    print()
    info("Fallback models are tried in order if the primary large model fails.")
    fallbacks = ask(
        "Fallback models (comma-separated, or empty)",
        "",
    )
    config["PRELLM_FALLBACKS"] = fallbacks


def _configure_logging(config: dict) -> None:
    """Configure logging settings."""
    print()
    print(f"{YELLOW}--- Step 6/6: Logging ---{NC}")

    log_levels = ["info", "debug", "warning", "error"]
    log_level = ask_choice("Log level:", log_levels, default=0)
    config["PRELLM_LOG_LEVEL"] = log_level


def _generate_config_file(config: dict) -> None:
    """Generate the .env configuration file."""
    print()
    print(f"{BLUE}--- Generating .env file ---{NC}")

    timestamp = datetime.now().isoformat()

    env_content = f"""# ============================================================
# preLLM Configuration — Generated by `make config`
# ============================================================
# 100% compatible with existing LiteLLM .env files.
# Your LiteLLM .env works immediately — just add PRELLM_SMALL_DEFAULT.
#
# Generated: {timestamp}

# ========== PRELLM PROXY AUTH ==========
# Master key for API authentication (like LiteLLM proxy)
# Leave empty to disable auth (dev mode)
LITELLM_MASTER_KEY={config.get('LITELLM_MASTER_KEY', '')}

# ========== SMALL LLM (local preprocessing) ==========
# Ollama runs locally — no API key needed
OLLAMA_API_BASE={config.get('OLLAMA_API_BASE', 'http://localhost:11434')}

# ========== LARGE LLM PROVIDERS (remote execution) ==========
# Fill in ONLY the providers you use. LiteLLM routes automatically.

# Anthropic Claude
ANTHROPIC_API_KEY={config.get('ANTHROPIC_API_KEY', '')}
# ANTHROPIC_BASE_URL=https://api.anthropic.com/v1

# OpenAI / GPT
OPENAI_API_KEY={config.get('OPENAI_API_KEY', '')}
# OPENAI_BASE_URL=https://api.openai.com/v1

# Groq (fast + cheap)
GROQ_API_KEY={config.get('GROQ_API_KEY', '')}
# GROQ_BASE_URL=https://api.groq.com/openai/v1

# Mistral
MISTRAL_API_KEY={config.get('MISTRAL_API_KEY', '')}
# MISTRAL_BASE_URL=https://api.mistral.ai/v1

# OpenRouter
OPENROUTER_API_KEY={config.get('OPENROUTER_API_KEY', '')}
# OPENROUTER_API_BASE=https://openrouter.ai/api/v1

# Azure OpenAI
# AZURE_API_KEY=
# AZURE_API_BASE=
# AZURE_API_VERSION=2024-02-01

# Google Vertex / Gemini
# GOOGLE_APPLICATION_CREDENTIALS=
# VERTEX_PROJECT=
# VERTEX_LOCATION=

# AWS Bedrock
# AWS_ACCESS_KEY_ID=
# AWS_SECRET_ACCESS_KEY=
# AWS_REGION_NAME=

# ========== PRELLM DEFAULTS ==========
# Default model pair (LiteLLM model naming)
PRELLM_SMALL_DEFAULT={config['PRELLM_SMALL_DEFAULT']}
PRELLM_LARGE_DEFAULT={config['PRELLM_LARGE_DEFAULT']}

# Default decomposition strategy: classify|structure|split|enrich|passthrough
PRELLM_STRATEGY={config['PRELLM_STRATEGY']}

# Fallback chain (comma-separated, tried in order if primary fails)
PRELLM_FALLBACKS={config.get('PRELLM_FALLBACKS', '')}

# ========== PRELLM SERVER ==========
PRELLM_HOST={config['PRELLM_HOST']}
PRELLM_PORT={config['PRELLM_PORT']}

# YAML config file (optional, overrides env defaults)
PRELLM_CONFIG={config.get('PRELLM_CONFIG', '')}

# ========== BUDGET & LIMITS ==========
PRELLM_MONTHLY_BUDGET={config.get('PRELLM_MONTHLY_BUDGET', '')}
PRELLM_MAX_TOKENS={config.get('PRELLM_MAX_TOKENS', '4096')}
PRELLM_TIMEOUT={config.get('PRELLM_TIMEOUT', '30')}

# ========== LOGGING ==========
PRELLM_LOG_LEVEL={config.get('PRELLM_LOG_LEVEL', 'info')}
"""

    # Write file
    env_path = Path(".env")
    if env_path.exists():
        backup = Path(".env.backup")
        backup.write_text(env_path.read_text())
        ok(f"Backed up existing .env → .env.backup")

    env_path.write_text(env_content)
    ok(f"Created: {env_path.absolute()}")


def _show_final_summary(config: dict) -> None:
    """Show final configuration summary."""
    print()
    print(f"{BLUE}--- Configuration Summary ---{NC}")
    print(f"  Small model:  {config['PRELLM_SMALL_DEFAULT']}")
    print(f"  Large model:  {config['PRELLM_LARGE_DEFAULT']}")
    print(f"  Strategy:     {config['PRELLM_STRATEGY']}")
    print(f"  Server:       {config['PRELLM_HOST']}:{config['PRELLM_PORT']}")
    if config.get("LITELLM_MASTER_KEY"):
        print(f"  Auth:         enabled ({mask_key(config['LITELLM_MASTER_KEY'])})")
    else:
        print(f"  Auth:         disabled (dev mode)")
    if config.get("PRELLM_MONTHLY_BUDGET"):
        print(f"  Budget:       ${config['PRELLM_MONTHLY_BUDGET']}/month")
    print(f"  Max tokens:   {config.get('PRELLM_MAX_TOKENS', '4096')}")
    print(f"  Timeout:      {config.get('PRELLM_TIMEOUT', '30')}s")
    if config.get("PRELLM_FALLBACKS"):
        print(f"  Fallbacks:    {config['PRELLM_FALLBACKS']}")
    print(f"  Log level:    {config.get('PRELLM_LOG_LEVEL', 'info')}")

    # Count configured providers
    configured = []
    for name, key in [
        ("OpenAI", "OPENAI_API_KEY"),
        ("Anthropic", "ANTHROPIC_API_KEY"),
        ("Groq", "GROQ_API_KEY"),
        ("Mistral", "MISTRAL_API_KEY"),
        ("OpenRouter", "OPENROUTER_API_KEY"),
    ]:
        if config.get(key):
            configured.append(name)
    if config.get("OLLAMA_API_BASE"):
        configured.append("Ollama")
    print(f"  Providers:    {', '.join(configured) if configured else 'none'}")

    print()
    print(f"{GREEN}{'=' * 60}{NC}")
    print(f"{GREEN}  Configuration Complete!{NC}")
    print(f"{GREEN}{'=' * 60}{NC}")
    print()
    print("Next steps:")
    print("  1. Verify:      make doctor")
    print("  2. Start:       prellm serve")
    print("  3. Test:        prellm query 'Hello world'")
    print("  4. Examples:    make examples")
    print()


def main() -> None:
    """Main configuration wizard."""
    print(f"{BLUE}{'=' * 60}{NC}")
    print(f"{BLUE}  preLLM Interactive Configuration Wizard{NC}")
    print(f"{BLUE}{'=' * 60}{NC}")
    print()
    print("This wizard creates a complete .env file with all preLLM settings.")
    print("Each step includes diagnostics to verify your configuration.")
    print()

    # Run diagnostics and get initial config
    config, installed_raw, ollama_reachable = _run_diagnostics()

    # Configure small LLM
    _configure_small_llm(config, installed_raw, ollama_reachable)

    # Configure large LLM
    _configure_large_llm(config)

    # Configure strategy and server
    _configure_strategy_and_server(config)

    # Configure budget and limits
    _configure_budget_and_limits(config)

    # Configure logging
    _configure_logging(config)

    # Generate config file
    _generate_config_file(config)

    # Show final summary
    _show_final_summary(config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(1)
