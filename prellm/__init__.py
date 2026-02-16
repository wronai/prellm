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

__version__ = "0.4.0"

# 1-function API — the primary interface (always uses v0.3 pipeline internally)
from prellm.core import preprocess_and_execute, preprocess_and_execute_sync

# Backward-compatible alias
from prellm.core import preprocess_and_execute_v3
from prellm.agents.preprocessor import PreprocessorAgent, PreprocessResult
from prellm.agents.executor import ExecutorAgent, ExecutorResult
from prellm.pipeline import PromptPipeline, PipelineConfig, PipelineStep, PipelineResult
from prellm.prompt_registry import PromptRegistry
from prellm.validators import ResponseValidator

# Class-based architecture
from prellm.core import PreLLM
from prellm.llm_provider import LLMProvider
from prellm.query_decomposer import QueryDecomposer
from prellm.models import (
    CompressedFolder,
    ContextSchema,
    DecompositionStrategy,
    DecompositionResult,
    DomainRule,
    FilterReport,
    LLMProviderConfig,
    PreLLMConfig,
    PreLLMResponse,
    RuntimeContext,
    SessionSnapshot,
    SensitivityLevel,
    ShellContext,
)

# Components
from prellm.chains.process_chain import ProcessChain
from prellm.analyzers.context_engine import ContextEngine
from prellm.context.user_memory import UserMemory
from prellm.context.sensitive_filter import SensitiveDataFilter
from prellm.context.shell_collector import ShellContextCollector
from prellm.context.folder_compressor import FolderCompressor
from prellm.context.schema_generator import ContextSchemaGenerator

# Logging
from prellm.logging_setup import setup_logging, get_logger

# Trace
from prellm.trace import TraceRecorder, get_current_trace

# Budget
from prellm.budget import BudgetTracker, BudgetExceededError, get_budget_tracker

__all__ = [
    # 1-function API (primary)
    "preprocess_and_execute",
    "preprocess_and_execute_sync",
    "preprocess_and_execute_v3",
    # Agents
    "PreprocessorAgent",
    "PreprocessResult",
    "ExecutorAgent",
    "ExecutorResult",
    # Pipeline
    "PromptPipeline",
    "PipelineConfig",
    "PipelineStep",
    "PipelineResult",
    "PromptRegistry",
    "ResponseValidator",
    # Class-based
    "PreLLM",
    "LLMProvider",
    "QueryDecomposer",
    # Models
    "CompressedFolder",
    "ContextSchema",
    "DecompositionStrategy",
    "DecompositionResult",
    "DomainRule",
    "FilterReport",
    "LLMProviderConfig",
    "PreLLMConfig",
    "PreLLMResponse",
    "RuntimeContext",
    "SessionSnapshot",
    "SensitivityLevel",
    "ShellContext",
    # Components
    "ProcessChain",
    "ContextEngine",
    "UserMemory",
    "SensitiveDataFilter",
    "ShellContextCollector",
    "FolderCompressor",
    "ContextSchemaGenerator",
    # Logging
    "setup_logging",
    "get_logger",
    # Trace
    "TraceRecorder",
    "get_current_trace",
    # Budget
    "BudgetTracker",
    "BudgetExceededError",
    "get_budget_tracker",
]
