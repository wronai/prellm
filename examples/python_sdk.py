"""preLLM Python SDK Examples (v0.3.8) — 1-function API, pipelines, memory, codebase context.

Usage:
    python examples/python_sdk.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path


# ═══════════════════════════════════════════════
# Example 1: One-Function API (recommended)
# ═══════════════════════════════════════════════

async def example_one_function():
    """The simplest way to use preLLM — like litellm.completion()."""
    from prellm import preprocess_and_execute

    # Zero-config
    result = await preprocess_and_execute(
        query="Explain Docker networking in 3 sentences",
    )
    print(f"[1a] Content: {result.content[:100]}...")
    print(f"     Model: {result.model_used}")
    print()

    # Full control
    result = await preprocess_and_execute(
        query="Deploy app to production",
        small_llm="ollama/qwen2.5:3b",
        large_llm="gpt-4o-mini",
        strategy="structure",
        user_context="gdansk_embedded_python",
    )
    print(f"[1b] Content: {result.content[:100]}...")
    print(f"     Small: {result.small_model_used}")
    print(f"     Large: {result.model_used}")
    if result.decomposition:
        print(f"     Strategy: {result.decomposition.strategy.value}")
        if result.decomposition.classification:
            print(f"     Intent: {result.decomposition.classification.intent}")
        if result.decomposition.missing_fields:
            print(f"     Missing: {result.decomposition.missing_fields}")
    print()


# ═══════════════════════════════════════════════
# Example 2: With Domain Rules
# ═══════════════════════════════════════════════

async def example_domain_rules():
    """Inline domain rules for safety checks."""
    from prellm import preprocess_and_execute

    result = await preprocess_and_execute(
        query="Usuń bazę danych klientów",
        small_llm="ollama/qwen2.5:3b",
        large_llm="gpt-4o-mini",
        strategy="structure",
        domain_rules=[
            {
                "name": "destructive_db",
                "keywords": ["delete", "drop", "usuń", "remove"],
                "intent": "database",
                "required_fields": ["target_database", "backup_confirmed", "approval"],
                "severity": "critical",
            },
        ],
    )
    print(f"[2] Content: {result.content[:100]}...")
    if result.decomposition:
        print(f"    Matched rule: {result.decomposition.matched_rule}")
        print(f"    Missing fields: {result.decomposition.missing_fields}")
    print()


# ═══════════════════════════════════════════════
# Example 3: Sync Version
# ═══════════════════════════════════════════════

def example_sync():
    """Synchronous wrapper — no async needed."""
    from prellm import preprocess_and_execute_sync

    result = preprocess_and_execute_sync(
        query="What is Kubernetes?",
        strategy="passthrough",
    )
    print(f"[3] Sync: {result.content[:100]}...")
    print()


# ═══════════════════════════════════════════════
# Example 4: Named Pipeline (dual_agent_full)
# ═══════════════════════════════════════════════

async def example_pipeline():
    """Use a named YAML-defined pipeline for maximum preprocessing quality."""
    from prellm import preprocess_and_execute

    result = await preprocess_and_execute(
        query="Design microservices architecture for e-commerce",
        small_llm="ollama/qwen2.5:3b",
        large_llm="gpt-4o-mini",
        pipeline="dual_agent_full",
        user_context={"team": "backend", "stack": "Python/FastAPI/K8s"},
    )
    print(f"[4] Content: {result.content[:100]}...")
    print(f"    Pipeline: dual_agent_full (4-step preprocessing)")
    print()


# ═══════════════════════════════════════════════
# Example 5: With UserMemory Context
# ═══════════════════════════════════════════════

async def example_user_memory():
    """Enrich queries with interaction history from UserMemory."""
    from prellm import preprocess_and_execute, UserMemory

    # Create a temp memory DB and seed it
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "memory.db"
        memory = UserMemory(path=db_path)
        await memory.add_interaction("Deploy v1 to staging", "Deployed successfully, 3 pods", {"env": "staging"})
        await memory.add_interaction("Debug OOM on backend", "Increased memory limits to 512Mi", {"ns": "backend"})
        memory.close()

        # Query with memory context — preLLM injects recent history into pipeline
        result = await preprocess_and_execute(
            query="Deploy v2 to staging",
            memory_path=str(db_path),
        )
        print(f"[5] Content: {result.content[:100]}...")
        print(f"    (UserMemory provided 2 recent interactions as context)")
    print()


# ═══════════════════════════════════════════════
# Example 6: With CodebaseIndexer Context
# ═══════════════════════════════════════════════

async def example_codebase_context():
    """Enrich queries with codebase symbols from CodebaseIndexer."""
    from prellm import preprocess_and_execute

    # Index the prellm codebase itself for context
    result = await preprocess_and_execute(
        query="Refactor the preprocess_and_execute function",
        codebase_path="prellm/",  # index the source directory
    )
    print(f"[6] Content: {result.content[:100]}...")
    print(f"    (CodebaseIndexer found relevant symbols in prellm/ source)")
    print()


# ═══════════════════════════════════════════════
# Example 7: Class-Based API (PreLLM)
# ═══════════════════════════════════════════════

async def example_class_based():
    """More control with the PreLLM class."""
    from prellm import PreLLM, DecompositionStrategy

    engine = PreLLM()  # or PreLLM("configs/prellm_config.yaml")

    # Full pipeline
    result = await engine(
        "Zdeployuj apkę na staging",
        strategy=DecompositionStrategy.STRUCTURE,
    )
    print(f"[7a] Content: {result.content[:100]}...")

    # Decompose only (no large LLM call)
    decomposition = await engine.decompose_only(
        "Deploy app to production",
        strategy=DecompositionStrategy.CLASSIFY,
    )
    print(f"[7b] Decomposition: {decomposition}")

    # Audit log
    log = engine.get_audit_log()
    print(f"[7c] Audit entries: {len(log)}")
    print()


# ═══════════════════════════════════════════════
# Example 8: Custom Pipeline (component-level)
# ═══════════════════════════════════════════════

async def example_custom_pipeline():
    """Build a pipeline from components for maximum flexibility."""
    from prellm import PromptRegistry, PromptPipeline, PreprocessorAgent, ExecutorAgent
    from prellm import LLMProvider, LLMProviderConfig

    registry = PromptRegistry()
    small = LLMProvider(LLMProviderConfig(model="ollama/qwen2.5:3b", max_tokens=512))
    large = LLMProvider(LLMProviderConfig(model="gpt-4o-mini", max_tokens=2048))

    pipeline = PromptPipeline.from_yaml(pipeline_name="structure", registry=registry, small_llm=small)
    preprocessor = PreprocessorAgent(small_llm=small, registry=registry, pipeline=pipeline)
    executor = ExecutorAgent(large_llm=large)

    prep = await preprocessor.preprocess("Deploy app to production")
    result = await executor.execute(prep.executor_input)

    print(f"[8] Content: {result.content[:100]}...")
    print(f"    Steps executed: {len(prep.decomposition.steps_executed)}")
    print(f"    Confidence: {prep.confidence}")
    print()


# ═══════════════════════════════════════════════
# Example 9: OpenAI SDK Compatibility
# ═══════════════════════════════════════════════

async def example_openai_sdk():
    """Use preLLM as an OpenAI drop-in replacement.

    Requires: prellm serve --port 8080
    """
    try:
        from openai import OpenAI

        client = OpenAI(base_url="http://localhost:8080/v1", api_key="prellm")
        response = client.chat.completions.create(
            model="prellm:qwen→claude",
            messages=[{"role": "user", "content": "Explain microservices in 2 sentences"}],
            extra_body={"prellm": {"strategy": "classify"}},
        )
        print(f"[9] OpenAI SDK: {response.choices[0].message.content[:100]}...")
    except Exception as e:
        print(f"[9] OpenAI SDK: (server not running or openai not installed) {type(e).__name__}")
    print()


# ═══════════════════════════════════════════════
# Example 10: All 5 Strategies
# ═══════════════════════════════════════════════

async def example_strategies():
    """Demonstrate all 5 decomposition strategies."""
    from prellm import preprocess_and_execute

    for strat in ["classify", "structure", "split", "enrich", "passthrough"]:
        result = await preprocess_and_execute(
            query="Deploy app to production with rollback",
            strategy=strat,
        )
        decomp = result.decomposition
        info = ""
        if decomp:
            if decomp.classification:
                info += f"intent={decomp.classification.intent} "
            if decomp.structure:
                info += f"action={decomp.structure.action} "
            if decomp.sub_queries:
                info += f"sub_queries={len(decomp.sub_queries)} "
            if decomp.missing_fields:
                info += f"missing={decomp.missing_fields} "
        print(f"[10] {strat:12s} → {info or 'passthrough'}")
    print()


# ═══════════════════════════════════════════════
# Run all examples
# ═══════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("  preLLM Python SDK Examples (v0.3.8)")
    print("=" * 60)
    print()

    examples = [
        ("1-Function API", example_one_function),
        ("Domain Rules", example_domain_rules),
        ("Sync Version", None),  # sync handled separately
        ("Named Pipeline", example_pipeline),
        ("UserMemory Context", example_user_memory),
        ("Codebase Context", example_codebase_context),
        ("Class-Based (PreLLM)", example_class_based),
        ("Custom Pipeline", example_custom_pipeline),
        ("OpenAI SDK", example_openai_sdk),
        ("All 5 Strategies", example_strategies),
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

    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
