#!/usr/bin/env python3
"""preLLM Quick Start (v0.3.8) — runnable examples for all major use cases.

Usage:
    pip install prellm
    ollama serve && ollama pull qwen2.5:3b
    python examples/quick_start.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path


async def example_zero_config():
    """Simplest possible usage — one line, default models."""
    from prellm import preprocess_and_execute

    result = await preprocess_and_execute("Refaktoryzuj kod")
    print(f"[zero-config] {result.content[:100]}...")
    print(f"  model: {result.model_used}")
    print(f"  small: {result.small_model_used}")


async def example_strategy():
    """Strategy-based preprocessing (classify, structure, split, enrich)."""
    from prellm import preprocess_and_execute

    result = await preprocess_and_execute(
        query="Deploy backend v2.3 to production with rollback",
        small_llm="ollama/qwen2.5:3b",
        large_llm="gpt-4o-mini",
        strategy="structure",
        user_context={"team": "backend", "env": "production"},
    )
    print(f"[strategy=structure] {result.content[:100]}...")
    if result.decomposition:
        print(f"  classification: {result.decomposition.classification}")
        print(f"  composed_prompt: {result.decomposition.composed_prompt[:80]}...")


async def example_pipeline():
    """Named pipeline — 4-step preprocessing for maximum quality."""
    from prellm import preprocess_and_execute

    result = await preprocess_and_execute(
        query="Deploy backend v2.3 to production with rollback",
        small_llm="ollama/qwen2.5:3b",
        large_llm="gpt-4o-mini",
        pipeline="dual_agent_full",
    )
    print(f"[pipeline=dual_agent_full] {result.content[:100]}...")


async def example_ollama_local():
    """Both models run locally via Ollama — $0.00 cost."""
    from prellm import preprocess_and_execute

    result = await preprocess_and_execute(
        query="Explain Python decorators with examples",
        small_llm="ollama/qwen2.5:3b",
        large_llm="ollama/llama3:8b",
        strategy="classify",
    )
    print(f"[ollama local] {result.content[:100]}...")


async def example_hybrid_ollama_openai():
    """Hybrid: local preprocessing + cloud execution."""
    from prellm import preprocess_and_execute

    result = await preprocess_and_execute(
        query="Review this Python function for security issues",
        small_llm="ollama/qwen2.5:3b",
        large_llm="gpt-4o-mini",
        strategy="enrich",
    )
    print(f"[hybrid ollama→openai] {result.content[:100]}...")


async def example_hybrid_ollama_anthropic():
    """Hybrid: local preprocessing + Anthropic execution."""
    from prellm import preprocess_and_execute

    result = await preprocess_and_execute(
        query="Write Kubernetes deployment manifest for a Python app",
        small_llm="ollama/phi3:mini",
        large_llm="anthropic/claude-sonnet-4-20250514",
        strategy="structure",
    )
    print(f"[hybrid ollama→anthropic] {result.content[:100]}...")


async def example_domain_rules():
    """Domain rules catch missing safety-critical fields."""
    from prellm import preprocess_and_execute

    result = await preprocess_and_execute(
        query="Usuń bazę danych klientów",
        small_llm="ollama/qwen2.5:3b",
        large_llm="gpt-4o-mini",
        strategy="structure",
        domain_rules=[{
            "name": "destructive_db",
            "keywords": ["delete", "drop", "usuń", "remove"],
            "required_fields": ["target_database", "backup_confirmed"],
            "severity": "critical",
        }],
    )
    print(f"[domain-rules] {result.content[:100]}...")
    if result.decomposition and result.decomposition.missing_fields:
        print(f"  missing fields: {result.decomposition.missing_fields}")


async def example_with_memory():
    """UserMemory enriches queries with interaction history."""
    from prellm import preprocess_and_execute, UserMemory

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "memory.db"
        memory = UserMemory(path=db_path)
        await memory.add_interaction("Deploy v1", "Deployed to staging OK", {"env": "staging"})
        memory.close()

        result = await preprocess_and_execute(
            query="Deploy v2 to staging",
            memory_path=str(db_path),
        )
        print(f"[with-memory] {result.content[:100]}...")


async def example_with_codebase():
    """CodebaseIndexer enriches queries with relevant source symbols."""
    from prellm import preprocess_and_execute

    result = await preprocess_and_execute(
        query="Refactor the pipeline execution logic",
        codebase_path="prellm/",
    )
    print(f"[with-codebase] {result.content[:100]}...")


def example_sync():
    """Synchronous version — for scripts, notebooks, non-async code."""
    from prellm import preprocess_and_execute_sync

    result = preprocess_and_execute_sync(
        "Explain Docker networking",
        small_llm="ollama/qwen2.5:3b",
        large_llm="gpt-4o-mini",
    )
    print(f"[sync] {result.content[:100]}...")


async def example_openai_sdk_client():
    """Use preLLM server from any OpenAI SDK client."""
    print("[openai-sdk] Start preLLM server first:")
    print("  prellm serve --port 8080 --small ollama/qwen2.5:3b --large gpt-4o-mini")
    print("  Then:")
    print("  client = openai.OpenAI(base_url='http://localhost:8080/v1', api_key='any')")
    print("  response = client.chat.completions.create(model='prellm:default', messages=[...])")


async def main():
    """Run all examples (requires LLM providers to be configured)."""
    print("=" * 60)
    print("  preLLM Quick Start (v0.3.8)")
    print("=" * 60)
    print()
    print("NOTE: Requires LLM providers. For Ollama: ollama serve && ollama pull qwen2.5:3b")
    print()

    examples = [
        ("Zero Config", example_zero_config),
        ("Strategy-Based", example_strategy),
        ("Named Pipeline", example_pipeline),
        ("Ollama Local ($0)", example_ollama_local),
        ("Hybrid Ollama→OpenAI", example_hybrid_ollama_openai),
        ("Hybrid Ollama→Anthropic", example_hybrid_ollama_anthropic),
        ("Domain Rules", example_domain_rules),
        ("With UserMemory", example_with_memory),
        ("With Codebase Context", example_with_codebase),
        ("Sync Version", None),
        ("OpenAI SDK Client", example_openai_sdk_client),
    ]

    for name, fn in examples:
        print(f"\n--- {name} ---")
        try:
            if fn is None:
                example_sync()
            else:
                await fn()
        except Exception as e:
            print(f"  (skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    asyncio.run(main())
