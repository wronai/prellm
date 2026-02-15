#!/usr/bin/env python3
"""preLLM Example — Kubernetes Debugging

Demonstrates K8s-specific preprocessing with domain rules.
Requires: Ollama running locally or cloud API keys.

Usage:
    python examples/k8s/main.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


async def main():
    from prellm import preprocess_and_execute

    print("=" * 60)
    print("  preLLM — K8s Debugging Example")
    print("=" * 60)

    # Example 1: CrashLoopBackOff diagnosis
    print("\n--- 1. CrashLoopBackOff ---")
    result = await preprocess_and_execute(
        query="Pod backend-api w namespace production restartuje sie z CrashLoopBackOff",
        config_path="configs/domains/devops_k8s.yaml",
        strategy="structure",
    )
    print(f"Model:   {result.model_used}")
    print(f"Content: {result.content[:200]}...")
    if result.decomposition:
        print(f"Strategy: {result.decomposition.strategy.value}")
        if result.decomposition.matched_rule:
            print(f"Rule:     {result.decomposition.matched_rule}")
        if result.decomposition.missing_fields:
            print(f"Missing:  {result.decomposition.missing_fields}")

    # Example 2: OOM with extra context
    print("\n--- 2. OOM with context ---")
    result = await preprocess_and_execute(
        query="Kubernetes pods killed by OOM on RPi cluster",
        config_path="configs/domains/devops_k8s.yaml",
        strategy="enrich",
        user_context={
            "cluster": "rpi-k3s-prod",
            "namespace": "backend",
            "node_ram": "4GB",
            "k8s_version": "1.28",
        },
    )
    print(f"Content: {result.content[:200]}...")

    # Example 3: HPA scaling
    print("\n--- 3. HPA Scaling ---")
    result = await preprocess_and_execute(
        query="Skonfiguruj autoscaling dla deployment frontend z min 2 max 10 replik",
        config_path="configs/domains/devops_k8s.yaml",
    )
    print(f"Content: {result.content[:200]}...")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
