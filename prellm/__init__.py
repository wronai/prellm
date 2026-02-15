"""preLLM — Small LLM decomposition middleware for prompt preprocessing and DevOps process chains."""

__version__ = "0.2.0"

# v0.2 — new architecture
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
    # v0.2
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
