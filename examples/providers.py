#!/usr/bin/env python3
"""preLLM Provider Examples (v0.3.8) — every supported LLM provider with working code.

Shows how to use preLLM with different LLM providers via LiteLLM.
Each example is self-contained and can be run independently.

Usage:
    python examples/providers.py
"""

from __future__ import annotations

import asyncio
import os


# ============================================================
# Helper
# ============================================================

async def run_example(name: str, small_llm: str, large_llm: str, **kwargs):
    """Run a single provider example."""
    from prellm import preprocess_and_execute

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  small: {small_llm}")
    print(f"  large: {large_llm}")
    print(f"{'='*60}")

    try:
        result = await preprocess_and_execute(
            query="Explain how to deploy a Python app to Kubernetes",
            small_llm=small_llm,
            large_llm=large_llm,
            strategy="classify",
            **kwargs,
        )
        print(f"  Status: OK")
        print(f"  Model used: {result.model_used}")
        print(f"  Response: {result.content[:150]}...")
    except Exception as e:
        print(f"  Status: SKIPPED ({type(e).__name__}: {e})")


# ============================================================
# Provider examples
# ============================================================

async def ollama_local():
    """Both models local via Ollama. Cost: $0.00"""
    await run_example(
        "Ollama (both local)",
        small_llm="ollama/qwen2.5:3b",
        large_llm="ollama/llama3:8b",
    )


async def ollama_plus_openai():
    """Local preprocessing, OpenAI execution. Cost: ~$0.15"""
    await run_example(
        "Ollama + OpenAI (hybrid)",
        small_llm="ollama/qwen2.5:3b",
        large_llm="gpt-4o-mini",
    )


async def ollama_plus_anthropic():
    """Local preprocessing, Anthropic execution."""
    await run_example(
        "Ollama + Anthropic (hybrid)",
        small_llm="ollama/qwen2.5:3b",
        large_llm="anthropic/claude-sonnet-4-20250514",
    )


async def openai_only():
    """Both models on OpenAI. Cost: ~$0.20"""
    await run_example(
        "OpenAI only",
        small_llm="gpt-4o-mini",
        large_llm="gpt-4o",
    )


async def anthropic_only():
    """Both models on Anthropic."""
    await run_example(
        "Anthropic only",
        small_llm="anthropic/claude-haiku",
        large_llm="anthropic/claude-sonnet-4-20250514",
    )


async def groq_fast():
    """Groq for ultra-fast inference. Cost: very low."""
    await run_example(
        "Groq (fast inference)",
        small_llm="groq/llama-3.1-8b-instant",
        large_llm="groq/llama-3.3-70b-versatile",
    )


async def mistral_cloud():
    """Mistral AI cloud models."""
    await run_example(
        "Mistral AI",
        small_llm="mistral/mistral-small-latest",
        large_llm="mistral/mistral-large-latest",
    )


async def azure_openai():
    """Azure OpenAI deployments."""
    # Requires: AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION
    await run_example(
        "Azure OpenAI",
        small_llm="azure/gpt-4o-mini-deployment",
        large_llm="azure/gpt-4o-deployment",
    )


async def aws_bedrock():
    """AWS Bedrock models."""
    # Requires: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME
    await run_example(
        "AWS Bedrock",
        small_llm="bedrock/anthropic.claude-3-haiku-20240307-v1:0",
        large_llm="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    )


async def google_gemini():
    """Google Gemini models."""
    # Requires: GEMINI_API_KEY
    await run_example(
        "Google Gemini",
        small_llm="gemini/gemini-2.0-flash",
        large_llm="gemini/gemini-2.5-pro-preview-06-05",
    )


async def together_ai():
    """Together AI hosted models."""
    # Requires: TOGETHER_API_KEY
    await run_example(
        "Together AI",
        small_llm="together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
        large_llm="together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    )


async def deepseek():
    """DeepSeek models."""
    # Requires: DEEPSEEK_API_KEY
    await run_example(
        "DeepSeek",
        small_llm="deepseek/deepseek-chat",
        large_llm="deepseek/deepseek-reasoner",
    )


async def openrouter_kimi():
    """OpenRouter — access many providers through one API. Kimi K2.5 for strong reasoning."""
    # Requires: OPENROUTER_API_KEY
    await run_example(
        "OpenRouter (Kimi K2.5)",
        small_llm="ollama/qwen2.5:3b",
        large_llm="openrouter/moonshotai/kimi-k2.5",
    )


async def mixed_providers_pipeline():
    """Pipeline with mixed providers."""
    from prellm import preprocess_and_execute

    print(f"\n{'='*60}")
    print("  Mixed Providers + Pipeline")
    print(f"{'='*60}")

    try:
        result = await preprocess_and_execute(
            query="Design a microservices architecture for an e-commerce platform",
            small_llm="ollama/qwen2.5:3b",         # free local preprocessing
            large_llm="anthropic/claude-sonnet-4-20250514",    # powerful execution
            pipeline="structure",                    # YAML-defined pipeline
            user_context={"team": "backend", "stack": "Python/FastAPI/K8s"},
        )
        print(f"  Status: OK")
        print(f"  Response: {result.content[:150]}...")
    except Exception as e:
        print(f"  Status: SKIPPED ({type(e).__name__}: {e})")


# ============================================================
# Environment setup helper
# ============================================================

def print_env_setup():
    """Print required environment variables for each provider."""
    print("""
ENVIRONMENT SETUP
=================

Each provider needs specific env vars. Set only the ones you'll use:

# Ollama (local, no key needed)
# Just run: ollama serve && ollama pull qwen2.5:3b

# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Groq
export GROQ_API_KEY=gsk_...

# Mistral
export MISTRAL_API_KEY=...

# Azure OpenAI
export AZURE_API_KEY=...
export AZURE_API_BASE=https://your-resource.openai.azure.com
export AZURE_API_VERSION=2024-02-01

# AWS Bedrock
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION_NAME=us-east-1

# Google Gemini
export GEMINI_API_KEY=...

# Together AI
export TOGETHER_API_KEY=...

# DeepSeek
export DEEPSEEK_API_KEY=...

# OpenRouter (access many providers)
export OPENROUTER_API_KEY=sk-or-v1-...

# LiteLLM Proxy (if running your own)
export OPENAI_API_BASE=http://localhost:4000
""")


async def main():
    print_env_setup()

    providers = [
        ollama_local,
        ollama_plus_openai,
        ollama_plus_anthropic,
        openai_only,
        anthropic_only,
        groq_fast,
        mistral_cloud,
        azure_openai,
        aws_bedrock,
        google_gemini,
        together_ai,
        deepseek,
        openrouter_kimi,
        mixed_providers_pipeline,
    ]

    print("\nRunning provider examples (skips providers without API keys)...\n")

    for fn in providers:
        await fn()

    print("\n\nDone! Providers without API keys were skipped.")


if __name__ == "__main__":
    asyncio.run(main())
