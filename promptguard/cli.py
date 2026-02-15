"""PromptGuard CLI — run prompts, execute process chains, and manage configs."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="promptguard",
    help="PromptGuard — LLM prompt middleware for bias detection, standardization, and DevOps process chains.",
    no_args_is_help=True,
)


@app.command()
def run(
    query: str = typer.Argument(..., help="The prompt/query to process"),
    config: Path = typer.Option("rules.yaml", "--config", "-c", help="Path to YAML config"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="LLM model to use"),
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="Analyze only, don't call LLM"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Run a single query through PromptGuard."""
    from promptguard.core import PromptGuard

    guard = PromptGuard(config_path=config)

    if dry_run:
        result = guard.analyze_only(query)
        if json_output:
            typer.echo(json.dumps(result, indent=2, default=str))
        else:
            typer.echo(f"🔍 Analysis for: {query}")
            typer.echo(f"   Needs clarification: {result['needs_clarify']}")
            typer.echo(f"   Patterns detected: {result['patterns']}")
            typer.echo(f"   Ambiguity flags: {result['ambiguity_flags']}")
            typer.echo(f"   Enriched query: {result['enriched']}")
        return

    result = asyncio.run(_run_guard(guard, query, model))

    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        status = "✅ clarified" if result.clarified else "📝 direct"
        typer.echo(f"\n{'='*60}")
        typer.echo(f"🛡️  PromptGuard [{status}] via {result.model_used}")
        typer.echo(f"{'='*60}")
        typer.echo(f"\n{result.content}")
        if result.analysis and result.analysis.detected_patterns:
            typer.echo(f"\n⚠️  Detected: {', '.join(result.analysis.detected_patterns)}")
        typer.echo(f"\n{'='*60}")


@app.command()
def process(
    config: Path = typer.Argument(..., help="Path to process chain YAML"),
    guard_config: Path = typer.Option("rules.yaml", "--guard-config", "-g", help="Path to guard YAML config"),
    dry_run: bool = typer.Option(False, "--dry-run", "-d", help="Analyze steps without calling LLM"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    env: Optional[str] = typer.Option(None, "--env", "-e", help="Environment override (e.g., production)"),
):
    """Execute a DevOps process chain."""
    from promptguard.chains.process_chain import ProcessChain

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
def analyze(
    query: str = typer.Argument(..., help="Query to analyze for bias/ambiguity"),
    config: Path = typer.Option("rules.yaml", "--config", "-c", help="Path to YAML config"),
):
    """Analyze a query without calling any LLM (bias detection + ambiguity check)."""
    from promptguard.core import PromptGuard

    guard = PromptGuard(config_path=config)
    result = guard.analyze_only(query)

    typer.echo(f"\n🔍 Analysis: {query}")
    typer.echo(f"   Needs clarification: {'⚠️  YES' if result['needs_clarify'] else '✅ NO'}")
    if result["patterns"]:
        typer.echo(f"   Patterns: {', '.join(result['patterns'])}")
    if result["ambiguity_flags"]:
        typer.echo(f"   Flags: {', '.join(result['ambiguity_flags'])}")
    if result["readability"] is not None:
        typer.echo(f"   Readability: {result['readability']:.1f}")
    typer.echo(f"   Enriched: {result['enriched']}")


@app.command()
def init(
    output: Path = typer.Option("rules.yaml", "--output", "-o", help="Output path for config"),
    devops: bool = typer.Option(False, "--devops", help="Include DevOps-specific patterns"),
):
    """Generate a starter rules.yaml config file."""
    import yaml

    config = {
        "bias_patterns": [
            {"regex": "(zawsze|always)\\s+\\w+", "action": "clarify", "severity": "medium",
             "description": "Absolute quantifier"},
            {"regex": "(tylko|only|just)\\s+\\w+", "action": "clarify", "severity": "low",
             "description": "Exclusive quantifier"},
        ],
        "clarify_template": "[KONTEKST]: Podaj szczegóły lub alternatywy dla: {query}",
        "max_retries": 3,
        "policy": "strict",
        "models": {
            "fallback": ["gpt-4o-mini", "llama3"],
            "timeout": 30,
            "max_tokens": 2048,
        },
    }

    if devops:
        config["bias_patterns"].extend([
            {"regex": "(deploy|zdeployuj)\\s+(na|to)\\s+(prod|production)", "action": "clarify",
             "severity": "critical", "description": "Production deployment without context"},
            {"regex": "(delete|drop|remove|usuń)\\s+(database|db|baz)", "action": "clarify",
             "severity": "critical", "description": "Destructive DB operation"},
            {"regex": "(restart|reboot|kill)\\s+(server|service|pod)", "action": "clarify",
             "severity": "high", "description": "Service disruption"},
            {"regex": "(scale|skaluj)\\s+(down|up|to\\s+\\d+)", "action": "clarify",
             "severity": "high", "description": "Scaling operation"},
        ])
        config["context_sources"] = [
            {"env": ["CLUSTER", "NAMESPACE", "GIT_SHA", "ENV"]},
            {"git": ["branch", "short_sha", "last_commit_msg"]},
            {"system": ["hostname", "os"]},
        ]

    with open(output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    typer.echo(f"✅ Config written to {output}")


async def _run_guard(guard, query: str, model: str):
    return await guard(query, model=model)


if __name__ == "__main__":
    app()
