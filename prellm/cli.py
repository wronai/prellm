"""Prellm CLI — run prompts, execute process chains, and manage configs."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="prellm",
    help="preLLM — Small LLM decomposition middleware for prompt preprocessing, DevOps process chains, and multi-provider orchestration.",
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
    """Run a single query through Prellm."""
    from prellm.core import prellm

    guard = prellm(config_path=config)

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
        typer.echo(f"🛡️  Prellm [{status}] via {result.model_used}")
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
def analyze(
    query: str = typer.Argument(..., help="Query to analyze for bias/ambiguity"),
    config: Path = typer.Option("rules.yaml", "--config", "-c", help="Path to YAML config"),
):
    """Analyze a query without calling any LLM (bias detection + ambiguity check)."""
    from prellm.core import prellm

    guard = prellm(config_path=config)
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
    output: Path = typer.Option("rules.yaml", "--output", "-o", help="Output path for config"),
    devops: bool = typer.Option(False, "--devops", help="Include DevOps-specific patterns"),
    v2: bool = typer.Option(False, "--v2", help="Generate preLLM v0.2 config instead of v0.1"),
):
    """Generate a starter config file (v0.1 rules.yaml or v0.2 prellm_config.yaml)."""
    import yaml

    if v2:
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
        out = output if str(output) != "rules.yaml" else Path("prellm_config.yaml")
    else:
        config = {
            "bias_patterns": [
                {"regex": "(zawsze|always)\\s+\\w+", "action": "clarify", "severity": "medium",
                 "description": "Absolute quantifier"},
                {"regex": "(tylko|only|just)\\s+\\w+", "action": "clarify", "severity": "low",
                 "description": "Exclusive quantifier"},
            ],
            "clarify_template": "[KONTEKST]: Podaj szczeg\u00f3\u0142y lub alternatywy dla: {query}",
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
                {"regex": "(delete|drop|remove|usu\u0144)\\s+(database|db|baz)", "action": "clarify",
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
        out = output

    with open(out, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    typer.echo(f"\u2705 Config written to {out}")


async def _run_guard(guard, query: str, model: str):
    return await guard(query, model=model)


if __name__ == "__main__":
    app()
