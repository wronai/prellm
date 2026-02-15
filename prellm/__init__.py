"""preLLM — Small LLM preprocessing before large LLM execution. One function, like litellm.completion().

Usage:
    from prellm import preprocess_and_execute

    result = await preprocess_and_execute(
        query="Deploy app to production",
        small_llm="ollama/qwen2.5:3b",
        large_llm="gpt-4o-mini",
    )
    print(result.content)
"""

__version__ = "0.3.0"

# 1-function API — the primary interface
from prellm.core import preprocess_and_execute, preprocess_and_execute_sync

# v0.2 — class-based architecture
from prellm.core import PreLLM
from prellm.llm_provider import LLMProvider
from prellm.query_decomposer import QueryDecomposer
from prellm.models import (
    DecompositionStrategy,
    DecompositionResult,
    DomainRule,
    LLMProviderConfig,
    PreLLMConfig,
    PreLLMResponse,
)

# v0.1 — backward compatibility
from prellm.core import prellm
from prellm.models import GuardResponse, GuardConfig, AnalysisResult
from prellm.chains.process_chain import ProcessChain
from prellm.analyzers.bias_detector import BiasDetector
from prellm.analyzers.context_engine import ContextEngine

__all__ = [
    # 1-function API (primary)
    "preprocess_and_execute",
    "preprocess_and_execute_sync",
    # v0.2 class-based
    "PreLLM",
    "LLMProvider",
    "QueryDecomposer",
    "DecompositionStrategy",
    "DecompositionResult",
    "DomainRule",
    "LLMProviderConfig",
    "PreLLMConfig",
    "PreLLMResponse",
    # v0.1 compat
    "prellm",
    "ProcessChain",
    "GuardResponse",
    "GuardConfig",
    "AnalysisResult",
    "BiasDetector",
    "ContextEngine",
]
