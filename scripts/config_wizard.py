#!/usr/bin/env python3
"""Interactive preLLM configuration wizard with diagnostics."""

import os
import re
import secrets
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# Colors
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BLUE = "\033[34m"
DIM = "\033[2m"
NC = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✔ {msg}{NC}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠ {msg}{NC}")


def fail(msg: str) -> None:
    print(f"  {RED}✘ {msg}{NC}")


def info(msg: str) -> None:
    print(f"  {DIM}{msg}{NC}")


def ask(question: str, default: str = "", required: bool = False) -> str:
    """Ask user for input with optional default."""
    if default:
        prompt = f"{question} [{default}]: "
    else:
        prompt = f"{question}: "

    while True:
        response = input(prompt).strip()
        if response:
            return response
        if default:
            return default
        if not required:
            return ""
        print("  (required)")


def ask_yn(question: str, default: bool = False) -> bool:
    """Ask yes/no question."""
    suffix = "[Y/n]" if default else "[y/N]"
    response = input(f"{question} {suffix}: ").strip().lower()
    if not response:
        return default
    return response in ("y", "yes", "tak", "t")


def ask_choice(question: str, options: list, default: int = 0) -> str:
    """Ask user to choose from options.

    Options can be strings or (label, value) tuples.
    """
    print(f"\n{question}")
    labels = []
    values = []
    for opt in options:
        if isinstance(opt, tuple):
            label, value = opt
        else:
            label = opt
            value = opt
        labels.append(label)
        values.append(value)

    for i, label in enumerate(labels, 1):
        marker = " (default)" if i - 1 == default else ""
        print(f"  {i}. {label}{marker}")

    while True:
        response = input("Choice [1-{}]: ".format(len(options))).strip()
        if not response:
            return values[default]
        try:
            idx = int(response) - 1
            if 0 <= idx < len(options):
                return values[idx]
        except ValueError:
            pass
        print("  (invalid choice)")


# ============================================================
# Diagnostics
# ============================================================

def check_ollama(base_url: str) -> list[str] | None:
    """Check if Ollama is reachable and list available models."""
    print()
    print(f"  Checking Ollama at {base_url} ...")
    models = fetch_ollama_models(base_url)
    if models is None:
        fail(f"Ollama not reachable at {base_url}")
        info("Start with: ollama serve")
        return None
    if models:
        ok(f"Ollama running — {len(models)} models: {', '.join(models[:5])}")
    else:
        warn("Ollama running but no models pulled. Run: ollama pull qwen2.5:3b")
    return models


def fetch_ollama_models(base_url: str) -> list[str] | None:
    """Return list of installed Ollama models (raw names) or None if unreachable."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            import json

            data = json.loads(resp.read())
            return [m.get("name", "").strip() for m in data.get("models", []) if m.get("name")]
    except urllib.error.URLError:
        return None
    except Exception:
        return None


def to_ollama_full(name: str) -> str:
    return name if name.startswith("ollama/") else f"ollama/{name}"


def strip_ollama_prefix(name: str) -> str:
    return name[len("ollama/") :] if name.startswith("ollama/") else name


def build_ollama_options(installed_raw: list[str], recommended: list[str]) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    seen: set[str] = set()

    for model in installed_raw:
        full = to_ollama_full(model)
        options.append((f"{full} (installed)", full))
        seen.add(full)

    for model in recommended:
        full = to_ollama_full(model)
        if full not in seen:
            options.append((f"{full} (not installed)", full))
            seen.add(full)

    options.append(("other (specify)", "other"))
    return options


def option_index_for_value(options: list[tuple[str, str]], value: str, default: int = 0) -> int:
    for idx, opt in enumerate(options):
        if opt[1] == value:
            return idx
    return default


def install_ollama_model(raw: str) -> bool:
    """Attempt to install an Ollama model via CLI."""
    try:
        result = subprocess.run(["ollama", "pull", raw], check=False)
    except FileNotFoundError:
        fail("Ollama CLI not found in PATH")
        return False

    if result.returncode == 0:
        ok(f"Installed: {raw}")
        return True
    fail(f"Failed to install: {raw}")
    return False


def validate_ollama_model(base_url: str, model: str, installed_raw: list[str], reachable: bool) -> bool:
    if not model.startswith("ollama/") or not reachable:
        if model.startswith("ollama/") and not reachable:
            warn("Ollama not reachable; skipping model validation")
        return True
    raw = strip_ollama_prefix(model)
    if raw in installed_raw:
        ok(f"Ollama model installed: {model}")
        return True
    warn(f"Ollama model not installed: {model}")
    info(f"Install with: ollama pull {raw}")
    if ask_yn("Install now with `ollama pull`?", default=False):
        if install_ollama_model(raw):
            installed_raw.append(raw)
            return True
    return ask_yn("Continue anyway?", default=True)


def check_api_key_format(provider: str, key: str) -> bool:
    """Validate API key format."""
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


def mask_key(key: str) -> str:
    """Mask API key for display."""
    if len(key) > 12:
        return key[:8] + "..." + key[-4:]
    return key[:4] + "..."


# ============================================================
# Main wizard
# ============================================================

def main():
    print(f"{BLUE}{'=' * 60}{NC}")
    print(f"{BLUE}  preLLM Interactive Configuration Wizard{NC}")
    print(f"{BLUE}{'=' * 60}{NC}")
    print()
    print("This wizard creates a complete .env file with all preLLM settings.")
    print("Each step includes diagnostics to verify your configuration.")
    print()

    config = {}

    # ========== Step 1: Small LLM ==========
    print(f"{YELLOW}--- Step 1/6: Small LLM (Local Preprocessing) ---{NC}")
    print("The small model runs locally for fast query preprocessing.")
    print()

    ollama_base = ask("Ollama API base URL", "http://localhost:11434")
    config["OLLAMA_API_BASE"] = ollama_base
    installed_raw = check_ollama(ollama_base)
    ollama_reachable = installed_raw is not None
    installed_raw = installed_raw or []

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

        if validate_ollama_model(ollama_base, config["PRELLM_SMALL_DEFAULT"], installed_raw, ollama_reachable):
            break

    ok(f"Small model: {config['PRELLM_SMALL_DEFAULT']}")

    # ========== Step 2: Large LLM ==========
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
        api_key = ask("OpenAI API Key", required=True)
        config["OPENAI_API_KEY"] = api_key
        check_api_key_format("openai", api_key)

        openai_models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "other"]
        model = ask_choice("Select model:", openai_models, default=0)
        if model == "other":
            model = ask("Enter model name:")
        config["PRELLM_LARGE_DEFAULT"] = model

    elif "Anthropic" in provider:
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

    elif "Groq" in provider:
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

    elif "OpenRouter" in provider:
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

    elif "Ollama" in provider:
        if not config.get("OLLAMA_API_BASE"):
            ollama_base = ask("Ollama API base URL", "http://localhost:11434")
            config["OLLAMA_API_BASE"] = ollama_base
            installed_raw = check_ollama(ollama_base)
            ollama_reachable = installed_raw is not None
            installed_raw = installed_raw or []

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
            if validate_ollama_model(ollama_base, model_full, installed_raw, ollama_reachable):
                break

    else:
        config["PRELLM_LARGE_DEFAULT"] = ask(
            "Enter full model name (e.g., mistral/mistral-small-latest):", required=True
        )
        key_name = ask("API key env var name (e.g., MISTRAL_API_KEY, or empty to skip)")
        if key_name:
            key_val = ask(f"{key_name}", required=True)
            config[key_name] = key_val

    ok(f"Large model: {config['PRELLM_LARGE_DEFAULT']}")

    # ========== Step 3: Strategy ==========
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

    # ========== Step 4: Server ==========
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

    # YAML config path
    config_path = ask("YAML config file path (optional)", "")
    config["PRELLM_CONFIG"] = config_path

    # ========== Step 5: Budget & Limits ==========
    print()
    print(f"{YELLOW}--- Step 5/6: Budget & Limits ---{NC}")

    budget = ask("Monthly budget in USD (empty = unlimited)", "")
    config["PRELLM_MONTHLY_BUDGET"] = budget

    max_tokens = ask("Max tokens per request", "4096")
    config["PRELLM_MAX_TOKENS"] = max_tokens

    timeout = ask("Request timeout in seconds", "30")
    config["PRELLM_TIMEOUT"] = timeout

    # Fallbacks
    print()
    info("Fallback models are tried in order if the primary large model fails.")
    fallbacks = ask(
        "Fallback models (comma-separated, or empty)",
        "",
    )
    config["PRELLM_FALLBACKS"] = fallbacks

    # ========== Step 6: Logging ==========
    print()
    print(f"{YELLOW}--- Step 6/6: Logging ---{NC}")

    log_levels = ["info", "debug", "warning", "error"]
    log_level = ask_choice("Log level:", log_levels, default=0)
    config["PRELLM_LOG_LEVEL"] = log_level

    # ============================================================
    # Generate .env
    # ============================================================
    print()
    print(f"{BLUE}--- Generating .env file ---{NC}")

    timestamp = datetime.now().isoformat()

    env_content = f"""\
# ============================================================
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

    # ============================================================
    # Final diagnostics summary
    # ============================================================
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(1)
