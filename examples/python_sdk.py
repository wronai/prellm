"""preLLM Python SDK Examples — all 3 interfaces: 1-function API, class-based, OpenAI SDK.

Usage:
    python examples/python_sdk.py
"""

from __future__ import annotations

import asyncio


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
# Example 4: Class-Based API (v0.2)
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
    print(f"[4a] Content: {result.content[:100]}...")
    print()

    # Decompose only (no large LLM call)
    decomposition = await engine.decompose_only(
        "Deploy app to production",
        strategy=DecompositionStrategy.CLASSIFY,
    )
    print(f"[4b] Decomposition: {decomposition}")
    print()

    # Audit log
    log = engine.get_audit_log()
    print(f"[4c] Audit entries: {len(log)}")
    print()


# ═══════════════════════════════════════════════
# Example 5: OpenAI SDK Compatibility
# ═══════════════════════════════════════════════

async def example_openai_sdk():
    """Use preLLM as an OpenAI drop-in replacement.

    Requires: prellm serve --port 8080
    """
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="http://localhost:8080",
            api_key="prellm",  # any string works
        )

        response = client.chat.completions.create(
            model="prellm:qwen→claude",
            messages=[
                {"role": "user", "content": "Explain microservices in 2 sentences"},
            ],
            extra_body={
                "prellm": {
                    "user_context": "gdansk embedded python",
                    "strategy": "classify",
                },
            },
        )
        print(f"[5] OpenAI SDK: {response.choices[0].message.content[:100]}...")
    except Exception as e:
        print(f"[5] OpenAI SDK: (server not running) {e}")
    print()


# ═══════════════════════════════════════════════
# Example 6: Batch Processing via API
# ═══════════════════════════════════════════════

async def example_batch():
    """Batch multiple queries via the API."""
    try:
        import httpx

        async with httpx.AsyncClient(base_url="http://localhost:8080") as client:
            resp = await client.post("/v1/batch", json=[
                {"query": "Refaktoryzuj hardcode", "context": "python", "strategy": "structure"},
                {"query": "K8s pod diagnostics", "context": "rpi cluster", "strategy": "enrich"},
                {"query": "Leasing calculation", "context": "PL automotive", "strategy": "classify"},
            ])
            data = resp.json()
            for r in data.get("results", []):
                print(f"[6] {r['query']}: {r['content'][:60]}...")
    except Exception as e:
        print(f"[6] Batch: (server not running) {e}")
    print()


# ═══════════════════════════════════════════════
# Example 7: All 5 Strategies
# ═══════════════════════════════════════════════

async def example_strategies():
    """Demonstrate all 5 decomposition strategies."""
    from prellm import preprocess_and_execute

    strategies = ["classify", "structure", "split", "enrich", "passthrough"]

    for strat in strategies:
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
        print(f"[7] {strat:12s} → {info or 'passthrough'}")
    print()


# ═══════════════════════════════════════════════
# Run all examples
# ═══════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("🧠 preLLM Python SDK Examples")
    print("=" * 60)
    print()

    await example_one_function()
    await example_domain_rules()
    example_sync()
    await example_class_based()
    await example_openai_sdk()
    await example_batch()
    await example_strategies()

    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
