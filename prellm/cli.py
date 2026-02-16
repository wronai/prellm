"""preLLM CLI — small LLM preprocessing before large LLM execution.

Usage:
    prellm "Deploy app to prod" --small ollama/qwen2.5:3b --large gpt-4o-mini
    prellm "Refaktoryzuj kod" --strategy structure --json
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="prellm",
    help="preLLM — Small LLM preprocessing before large LLM execution. Like litellm.completion() but with decomposition.",
    no_args_is_help=True,
)


def _init_logging() -> None:
    """Initialize nfo logging from .env config (called once per CLI invocation)."""
    from prellm.env_config import get_env_config
    from prellm.logging_setup import setup_logging

    env = get_env_config()
    setup_logging(level=env.log_level)


@app.command()
def query(
    prompt: str = typer.Argument(..., help="The prompt/query to preprocess and execute"),
    small: Optional[str] = typer.Option(None, "--small", "-s", help="Small LLM for preprocessing (default: from .env)"),
    large: Optional[str] = typer.Option(None, "--large", "-l", help="Large LLM for execution (default: from .env)"),
    strategy: Optional[str] = typer.Option(None, "--strategy", "-S", help="Strategy: classify|structure|split|enrich|passthrough (default: from .env)"),
    context: Optional[str] = typer.Option(None, "--context", "-C", help="User context tag (e.g. 'gdansk_embedded_python')"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Optional YAML config file"),
    memory: Optional[Path] = typer.Option(None, "--memory", "-m", help="Path to UserMemory database"),
    codebase: Optional[Path] = typer.Option(None, "--codebase", help="Path to codebase root for context indexing"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    trace: bool = typer.Option(False, "--trace", "-t", help="Generate markdown execution trace (.prellm/)"),
    trace_dir: Optional[Path] = typer.Option(None, "--trace-dir", help="Trace output directory (default: .prellm)"),
    env_file: Optional[Path] = typer.Option(None, "--env-file", help="Path to .env file (default: .env)"),
):
    """Preprocess a query with small LLM, then execute with large LLM."""
    from prellm.core import preprocess_and_execute
    from prellm.env_config import get_env_config
    from prellm.trace import TraceRecorder
    from prellm.budget import get_budget_tracker

    _init_logging()

    env = get_env_config(str(env_file) if env_file else None)
    effective_small = small or env.small_model
    effective_large = large or env.large_model
    effective_strategy = strategy or env.strategy

    # Initialize budget tracker if configured
    if env.monthly_budget:
        get_budget_tracker(monthly_limit=env.monthly_budget)

    # Start trace if requested
    recorder = None
    if trace:
        recorder = TraceRecorder(output_dir=Path(trace_dir) if trace_dir else Path(".prellm"))
        recorder.start(
            query=prompt,
            small_llm=effective_small,
            large_llm=effective_large,
            strategy=effective_strategy,
        )

    result = asyncio.run(preprocess_and_execute(
        query=prompt,
        small_llm=effective_small,
        large_llm=effective_large,
        strategy=effective_strategy,
        user_context=context,
        config_path=str(config) if config else None,
        memory_path=str(memory) if memory else None,
        codebase_path=str(codebase) if codebase else None,
    ))

    # Stop trace and output
    if recorder:
        recorder.stop()
        # Print compact trace to stdout
        typer.echo(recorder.to_stdout())
        # Save full markdown trace
        filepath = recorder.save()
        typer.echo(f"📄 Trace saved: {filepath}")

    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"\U0001f9e0 preLLM [{effective_small} \u2192 {effective_large}]")
        typer.echo(f"{'='*60}")
        if result.decomposition and result.decomposition.classification:
            c = result.decomposition.classification
            typer.echo(f"   Intent: {c.intent} (confidence: {c.confidence:.2f})")
        if result.decomposition and result.decomposition.matched_rule:
            typer.echo(f"   Rule: {result.decomposition.matched_rule}")
        if result.decomposition and result.decomposition.missing_fields:
            typer.echo(f"   \u26a0\ufe0f  Missing: {', '.join(result.decomposition.missing_fields)}")
        typer.echo(f"{'='*60}")
        typer.echo(f"\n{result.content}")
        typer.echo(f"\n{'='*60}")
        typer.echo(f"   Small: {result.small_model_used} | Large: {result.model_used} | Retries: {result.retries}")
        typer.echo(f"{'='*60}")


@app.command()
def process(
    config: Path = typer.Argument(..., help="Path to process chain YAML"),
    guard_config: Path = typer.Option("rules.yaml", "--guard-config", "-g", help="Path to guard YAML config"),
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="Analyze steps without calling LLM"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    env: Optional[str] = typer.Option(None, "--env", "-e", help="Environment override (e.g., production)"),
):
    """Execute a DevOps process chain."""
    from prellm.chains.process_chain import ProcessChain

    chain = ProcessChain(config_path=config, guard_config_path=guard_config)

    extra = {}
    if env:
        extra["env"] = env

    result = asyncio.run(chain.execute(extra_context=extra, dry_run=dry_run))

    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"🔗 Process: {result.process_name}")
        typer.echo(f"   Status: {'✅ Completed' if result.completed else '⏸️  Incomplete'}")
        typer.echo(f"   Duration: {result.total_duration_seconds:.2f}s")
        typer.echo(f"{'='*60}")
        for step in result.steps:
            icon = {
                "completed": "✅",
                "failed": "❌",
                "awaiting_approval": "⏳",
                "rolled_back": "↩️",
            }.get(step.status.value, "🔄")
            typer.echo(f"   {icon} {step.step_name}: {step.status.value} ({step.duration_seconds:.2f}s)")
            if step.error:
                typer.echo(f"      Error: {step.error}")
        typer.echo(f"{'='*60}")


@app.command()
def decompose(
    query: str = typer.Argument(..., help="The prompt/query to decompose"),
    config: Path = typer.Option("configs/prellm_config.yaml", "--config", "-c", help="Path to preLLM v0.2 YAML config"),
    strategy: str = typer.Option("classify", "--strategy", "-s", help="Decomposition strategy: classify|structure|split|enrich|passthrough"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """[v0.2] Decompose a query using small LLM without calling the large model."""
    from prellm.core import PreLLM
    from prellm.models import DecompositionStrategy

    engine = PreLLM(config_path=config)
    strat = DecompositionStrategy(strategy)

    result = asyncio.run(engine.decompose_only(query, strategy=strat))

    if json_output:
        typer.echo(json.dumps(result, indent=2, default=str))
    else:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"\U0001f9e0 preLLM Decomposition [{strategy}]")
        typer.echo(f"{'='*60}")
        typer.echo(f"   Original: {result['original_query']}")
        if result.get('classification'):
            c = result['classification']
            typer.echo(f"   Intent: {c['intent']} (confidence: {c['confidence']:.2f})")
            typer.echo(f"   Domain: {c['domain']}")
        if result.get('structure'):
            s = result['structure']
            typer.echo(f"   Action: {s['action']}, Target: {s['target']}")
        if result.get('sub_queries'):
            typer.echo(f"   Sub-queries: {result['sub_queries']}")
        if result.get('missing_fields'):
            typer.echo(f"   \u26a0\ufe0f  Missing: {', '.join(result['missing_fields'])}")
        if result.get('matched_rule'):
            typer.echo(f"   Matched rule: {result['matched_rule']}")
        typer.echo(f"   Composed: {result.get('composed_prompt', '')[:200]}")
        typer.echo(f"{'='*60}")


@app.command()
def init(
    output: Path = typer.Option("prellm_config.yaml", "--output", "-o", help="Output path for config"),
    devops: bool = typer.Option(False, "--devops", help="Include DevOps-specific domain rules and context sources"),
):
    """Generate a starter preLLM config file."""
    import yaml

    config = {
        "small_model": {"model": "phi3:mini", "fallback": ["qwen2:1.5b"], "max_tokens": 512, "temperature": 0.0},
        "large_model": {"model": "gpt-4o-mini", "fallback": ["llama3"], "max_tokens": 2048},
        "default_strategy": "classify",
        "policy": "devops" if devops else "strict",
        "domain_rules": [
            {"name": "production_deploy", "keywords": ["deploy", "push", "release"],
             "intent": "deploy", "required_fields": ["environment_details", "version"],
             "severity": "critical", "strategy": "structure"},
            {"name": "database_operation", "keywords": ["delete", "drop", "migrate"],
             "intent": "database", "required_fields": ["target_database", "backup_confirmed"],
             "severity": "critical", "strategy": "structure"},
        ] if devops else [],
        "context_sources": [
            {"env": ["CLUSTER", "NAMESPACE", "GIT_SHA", "ENV"]},
            {"git": ["branch", "short_sha", "last_commit_msg"]},
            {"system": ["hostname", "os"]},
        ] if devops else [],
    }

    with open(output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    typer.echo(f"\u2705 Config written to {output}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-H", help="Bind host"),
    port: int = typer.Option(8080, "--port", "-p", help="Bind port"),
    small: Optional[str] = typer.Option(None, "--small", "-s", help="Override small LLM (default: from .env)"),
    large: Optional[str] = typer.Option(None, "--large", "-l", help="Override large LLM (default: from .env)"),
    strategy: Optional[str] = typer.Option(None, "--strategy", "-S", help="Override strategy (default: from .env)"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file"),
    env_file: Optional[Path] = typer.Option(None, "--env-file", help="Path to .env file (default: .env)"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev mode)"),
):
    """Start the OpenAI-compatible API server.

    Reads config from .env file (LiteLLM-compatible). CLI args override .env values.

    Example:
        prellm serve
        prellm serve --small ollama/qwen2.5:3b --large gpt-4o-mini --port 8080
    """
    import uvicorn
    from prellm.env_config import get_env_config
    from prellm.server import create_app

    env = get_env_config(str(env_file) if env_file else None)
    _init_logging()

    effective_small = small or env.small_model
    effective_large = large or env.large_model
    effective_strategy = strategy or env.strategy
    effective_host = host
    effective_port = port

    create_app(
        small_model=effective_small,
        large_model=effective_large,
        strategy=effective_strategy,
        config_path=str(config) if config else env.config_path,
        dotenv_path=str(env_file) if env_file else None,
    )

    auth_status = "ON (LITELLM_MASTER_KEY)" if env.master_key else "OFF (no key set)"

    typer.echo(f"\n\U0001f9e0 preLLM API Server")
    typer.echo(f"   http://{effective_host}:{effective_port}")
    typer.echo(f"   Small: {effective_small} | Large: {effective_large}")
    typer.echo(f"   Strategy: {effective_strategy} | Auth: {auth_status}")
    typer.echo(f"   Endpoints: /v1/chat/completions, /v1/batch, /v1/models, /health")
    typer.echo(f"{'='*60}\n")

    uvicorn.run(
        "prellm.server:app",
        host=effective_host,
        port=effective_port,
        reload=reload,
        log_level=env.log_level,
    )


def _doctor_check_config(env) -> list[str]:
    """Format configuration summary lines."""
    lines = [
        f"   Small LLM:  {env.small_model}",
        f"   Large LLM:  {env.large_model}",
        f"   Strategy:   {env.strategy}",
        f"   Server:     {env.host}:{env.port}",
        f"   Auth:       {'ON' if env.master_key else 'OFF (no LITELLM_MASTER_KEY)'}",
    ]
    if env.config_path:
        lines.append(f"   Config:     {env.config_path}")
    if env.fallbacks:
        lines.append(f"   Fallbacks:  {', '.join(env.fallbacks)}")
    if env.monthly_budget:
        lines.append(f"   Budget:     ${env.monthly_budget:.2f}/month")
    return lines


def _doctor_check_providers(env, live: bool = False) -> list[str]:
    """Check providers and return formatted lines."""
    from prellm.env_config import check_providers

    if live:
        import asyncio
        from prellm.env_config import check_providers_live
        results = asyncio.run(check_providers_live(env))
    else:
        results = check_providers(env)

    lines = []
    for name, info in results.items():
        status = info["status"]
        icon = "\u2713" if status in ("ok", "configured") else ("\u2717" if status == "no_key" else "!")
        lines.append(f"   {icon} {name.upper():12s} {info['detail']}")
        if "models" in info:
            lines.append(f"     Models: {', '.join(info['models'][:5])}")
    return lines


def _doctor_check_files(env_file: Path | None) -> list[str]:
    """Check config files and return formatted lines."""
    lines = []
    env_path = Path(str(env_file)) if env_file else Path(".env")
    if env_path.is_file():
        lines.append(f"   \u2713 {env_path} (loaded)")
    else:
        lines.append(f"   \u2717 {env_path} (not found \u2014 run: cp .env.example .env)")

    example_path = Path(".env.example")
    if example_path.is_file():
        lines.append(f"   \u2713 .env.example (available)")
    else:
        lines.append(f"   \u2717 .env.example (not found)")

    config_yaml = Path("configs/prellm_config.yaml")
    if config_yaml.is_file():
        lines.append(f"   \u2713 {config_yaml}")
    return lines


@app.command()
def doctor(
    env_file: Optional[Path] = typer.Option(None, "--env-file", help="Path to .env file"),
    live: bool = typer.Option(False, "--live", help="Test live connectivity to providers"),
):
    """Check configuration and provider connectivity.

    Validates .env config, API keys, and optionally tests live connections.

    Example:
        prellm doctor
        prellm doctor --live
    """
    from prellm.env_config import get_env_config

    env = get_env_config(str(env_file) if env_file else None)

    typer.echo(f"\n\U0001f9e0 preLLM Doctor")
    typer.echo(f"{'='*60}")

    typer.echo(f"\n\U0001f4cb Configuration:")
    for line in _doctor_check_config(env):
        typer.echo(line)

    typer.echo(f"\n\U0001f50c Providers:")
    for line in _doctor_check_providers(env, live=live):
        typer.echo(line)

    typer.echo(f"\n\U0001f4c4 Files:")
    for line in _doctor_check_files(env_file):
        typer.echo(line)

    typer.echo(f"\n{'='*60}")
    typer.echo(f"\u2705 Doctor complete. Use --live to test connectivity.\n")


# ============================================================
# Config management
# ============================================================

config_app = typer.Typer(
    name="config",
    help="Manage preLLM configuration — API keys, models, defaults.",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")


@config_app.command("set")
def config_set_cmd(
    key: str = typer.Argument(..., help="Config key (e.g. openrouter-key, model, small-model, strategy)"),
    value: str = typer.Argument(..., help="Value to set"),
    global_: bool = typer.Option(False, "--global", "-g", help="Save to ~/.prellm/.env (user-wide) instead of project .env"),
):
    """Set a config value persistently.

    Saves to .env (project) or ~/.prellm/.env (--global).

    Examples:
        prellm config set openrouter-key sk-or-v1-abc123
        prellm config set model openrouter/moonshotai/kimi-k2.5
        prellm config set small-model ollama/qwen2.5:3b
        prellm config set strategy structure
        prellm config set openrouter-key sk-or-v1-abc123 --global
    """
    from prellm.env_config import config_set, mask_value, resolve_alias

    env_var, path = config_set(key, value, global_=global_)
    masked = mask_value(env_var, value)
    typer.echo(f"\u2705 {env_var}={masked}")
    typer.echo(f"   Saved to: {path}")


@config_app.command("get")
def config_get_cmd(
    key: str = typer.Argument(..., help="Config key (e.g. openrouter-key, model, small-model)"),
    raw: bool = typer.Option(False, "--raw", "-r", help="Show unmasked value"),
):
    """Get a config value.

    Examples:
        prellm config get openrouter-key
        prellm config get model
        prellm config get small-model
    """
    from prellm.env_config import config_get, mask_value

    env_var, val, source = config_get(key)
    if val is None:
        typer.echo(f"\u2717 {env_var} — not set")
        typer.echo(f"   Set with: prellm config set {key} <value>")
        raise typer.Exit(1)
    displayed = val if raw else mask_value(env_var, val)
    typer.echo(f"{env_var}={displayed}")
    typer.echo(f"   Source: {source}")


def _format_config_sections(entries: dict) -> dict[str, list[str]]:
    """Group config entries into categorized sections for display."""
    sections: dict[str, list[str]] = {
        "\U0001f511 API Keys": [],
        "\U0001f916 Models": [],
        "\u2699\ufe0f  Settings": [],
        "\U0001f4cb Other": [],
    }
    for var, info in entries.items():
        alias = f" ({info['alias']})" if info["alias"] else ""
        line = f"   {var}{alias} = {info['value']}  [{info['source']}]"
        if "API_KEY" in var or "SECRET" in var or var == "LITELLM_MASTER_KEY":
            sections["\U0001f511 API Keys"].append(line)
        elif "MODEL" in var or "DEFAULT" in var:
            sections["\U0001f916 Models"].append(line)
        elif var.startswith("PRELLM_"):
            sections["\u2699\ufe0f  Settings"].append(line)
        else:
            sections["\U0001f4cb Other"].append(line)
    return sections


@config_app.command("list")
def config_list_cmd(
    raw: bool = typer.Option(False, "--raw", "-r", help="Show unmasked secret values"),
):
    """List all configured values.

    Example:
        prellm config list
        prellm config list --raw
    """
    from prellm.env_config import config_list

    entries = config_list(show_secrets=raw)
    if not entries:
        typer.echo("No config values set.")
        typer.echo("   Set with: prellm config set <key> <value>")
        typer.echo("   Example:  prellm config set openrouter-key sk-or-v1-abc123")
        return

    typer.echo(f"\n\U0001f9e0 preLLM Configuration")
    typer.echo(f"{'='*60}")

    for title, lines in _format_config_sections(entries).items():
        if lines:
            typer.echo(f"\n{title}:")
            for line in lines:
                typer.echo(line)

    typer.echo(f"\n{'='*60}")


@config_app.command("show")
def config_show_cmd():
    """Show effective configuration (resolved from all sources).

    Example:
        prellm config show
    """
    from prellm.env_config import get_env_config, mask_value

    env = get_env_config()
    typer.echo(f"\n\U0001f9e0 preLLM Effective Configuration")
    typer.echo(f"{'='*60}")
    typer.echo(f"   Small LLM:     {env.small_model}")
    typer.echo(f"   Large LLM:     {env.large_model}")
    typer.echo(f"   Strategy:      {env.strategy}")
    typer.echo(f"   Server:        {env.host}:{env.port}")
    typer.echo(f"   Auth:          {'ON' if env.master_key else 'OFF'}")
    typer.echo(f"   Log level:     {env.log_level}")
    typer.echo(f"   Max tokens:    {env.max_tokens}")
    typer.echo(f"   Timeout:       {env.timeout}s")
    if env.fallbacks:
        typer.echo(f"   Fallbacks:     {', '.join(env.fallbacks)}")
    if env.monthly_budget:
        typer.echo(f"   Budget:        ${env.monthly_budget:.2f}/month")
    if env.config_path:
        typer.echo(f"   Config file:   {env.config_path}")

    typer.echo(f"\n\U0001f50c Providers:")
    for name, info in env.providers.items():
        if info["has_key"] or name == "ollama":
            typer.echo(f"   \u2713 {name.upper():14s} {info.get('base_url', '')}")
        else:
            typer.echo(f"   \u2717 {name.upper():14s} ({info.get('key_var', '')} not set)")

    typer.echo(f"\n{'='*60}")


@config_app.command("init-env")
def config_init_env(
    global_: bool = typer.Option(False, "--global", "-g", help="Create ~/.prellm/.env instead of project .env"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing file"),
):
    """Generate a starter .env file with all available settings.

    Example:
        prellm config init-env
        prellm config init-env --global
    """
    from prellm.env_config import _resolve_config_path

    path = _resolve_config_path(global_)
    if path.is_file() and not force:
        typer.echo(f"\u26a0\ufe0f  {path} already exists. Use --force to overwrite.")
        raise typer.Exit(1)

    path.parent.mkdir(parents=True, exist_ok=True)
    template = """\
# preLLM Configuration
# Generated by: prellm config init-env
# Docs: https://github.com/wronai/prellm

# ── Models ──────────────────────────────────────────────
PRELLM_SMALL_DEFAULT=ollama/qwen2.5:3b
PRELLM_LARGE_DEFAULT=gpt-4o-mini
PRELLM_STRATEGY=classify

# ── API Keys (uncomment and fill in) ───────────────────
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GROQ_API_KEY=gsk_...
# MISTRAL_API_KEY=...
# OPENROUTER_API_KEY=sk-or-v1-...
# DEEPSEEK_API_KEY=...
# TOGETHERAI_API_KEY=...
# GEMINI_API_KEY=...
# MOONSHOT_API_KEY=...

# ── Azure OpenAI ───────────────────────────────────────
# AZURE_API_KEY=...
# AZURE_API_BASE=https://your-resource.openai.azure.com
# AZURE_API_VERSION=2024-02-01

# ── AWS Bedrock ────────────────────────────────────────
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
# AWS_REGION_NAME=us-east-1

# ── Ollama (local) ─────────────────────────────────────
# OLLAMA_API_BASE=http://localhost:11434

# ── OpenRouter ─────────────────────────────────────────
# OPENROUTER_API_BASE=https://openrouter.ai/api/v1

# ── Server ─────────────────────────────────────────────
# PRELLM_HOST=0.0.0.0
# PRELLM_PORT=8080
# PRELLM_LOG_LEVEL=info
# LITELLM_MASTER_KEY=sk-prellm-...

# ── Limits ─────────────────────────────────────────────
# PRELLM_MAX_TOKENS=4096
# PRELLM_TIMEOUT=30
# PRELLM_MONTHLY_BUDGET=50.00
# PRELLM_FALLBACKS=ollama/llama3:8b,gpt-4o-mini
"""
    with open(path, "w") as f:
        f.write(template)

    typer.echo(f"\u2705 Created {path}")
    typer.echo(f"   Edit the file to add your API keys.")


# ============================================================
# Budget
# ============================================================

@app.command()
def budget(
    reset: bool = typer.Option(False, "--reset", help="Reset current month's budget"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Show LLM API spend tracking and budget status.

    Example:
        prellm budget
        prellm budget --json
        prellm budget --reset
    """
    from prellm.budget import get_budget_tracker
    from prellm.env_config import get_env_config

    env = get_env_config()
    tracker = get_budget_tracker(monthly_limit=env.monthly_budget)

    if reset:
        tracker.reset()
        typer.echo("✅ Budget reset for current month.")
        return

    summary = tracker.summary()

    if json_output:
        typer.echo(json.dumps(summary, indent=2, default=str))
        return

    typer.echo(f"\n💰 preLLM Budget")
    typer.echo(f"{'='*60}")
    typer.echo(f"   Month:      {summary['month']}")
    typer.echo(f"   Spent:      ${summary['total_cost']:.4f}")
    if summary['monthly_limit'] is not None:
        typer.echo(f"   Limit:      ${summary['monthly_limit']:.2f}")
        typer.echo(f"   Remaining:  ${summary['remaining']:.4f}")
        pct = (summary['total_cost'] / summary['monthly_limit'] * 100) if summary['monthly_limit'] > 0 else 0
        bar_len = 30
        filled = int(bar_len * min(pct, 100) / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        typer.echo(f"   Usage:      [{bar}] {pct:.1f}%")
    else:
        typer.echo(f"   Limit:      not set (PRELLM_MONTHLY_BUDGET)")
    typer.echo(f"   Requests:   {summary['requests']}")

    if summary['by_model']:
        typer.echo(f"\n   By model:")
        for model, cost in sorted(summary['by_model'].items(), key=lambda x: -x[1]):
            typer.echo(f"     {model}: ${cost:.4f}")

    typer.echo(f"\n{'='*60}")


# ============================================================
# Models
# ============================================================

@app.command()
def models(
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Filter by provider (e.g. openrouter, ollama, openai)"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search model name"),
):
    """List popular model pairs and provider examples.

    Examples:
        prellm models
        prellm models --provider openrouter
        prellm models --search kimi
    """
    from prellm.model_catalog import list_model_pairs, list_openrouter_models

    pairs = list_model_pairs(provider=provider, search=search)
    or_models = list_openrouter_models(provider=provider, search=search)

    typer.echo(f"\n\U0001f916 preLLM Model Pairs")
    typer.echo(f"{'='*60}")

    if pairs:
        typer.echo(f"\n{'Name':<25s} {'Small LLM':<30s} {'Large LLM':<45s} {'Cost':>6s}")
        typer.echo(f"{'-'*25} {'-'*30} {'-'*45} {'-'*6}")
        for m in pairs:
            typer.echo(f"{m['name']:<25s} {m['small']:<30s} {m['large']:<45s} {m['cost']:>6s}")

    if or_models:
        typer.echo(f"\n\U0001f310 OpenRouter Models (use with --large):")
        for m in or_models:
            typer.echo(f"   {m['model_id']}")
            typer.echo(f"      {m['description']}")

    typer.echo(f"\n\U0001f4a1 Usage:")
    typer.echo(f'   prellm "Deploy app" --large openrouter/moonshotai/kimi-k2.5')
    typer.echo(f"   prellm config set model openrouter/moonshotai/kimi-k2.5")
    typer.echo(f"   prellm config set openrouter-key sk-or-v1-abc123")
    typer.echo(f"\n{'='*60}")


if __name__ == "__main__":
    app()
