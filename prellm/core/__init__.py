"""prellm/core package — Internal implementation of the core preprocessing pipeline.

This package contains the internal implementation details for prellm's
core functionality. Public API remains in prellm.core module.
"""

# Re-export main public API from prellm.core file (avoiding circular import)
# These are imported lazily to avoid issues with the package/file shadowing
def __getattr__(name):
    """Lazy import to handle package/file shadowing."""
    if name in ("preprocess_and_execute", "preprocess_and_execute_sync", "preprocess_and_execute_v3", "PreLLM"):
        import importlib
        # Import from the core.py file directly
        core_module = importlib.import_module("prellm.core_file")
        return getattr(core_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Context collection helpers
from prellm.core.context import (
    _collect_user_context,
    _collect_environment_context,
    _compress_codebase_folder,
    _generate_context_schema,
    _build_sensitive_filter,
    _initialize_context_components,
    _prepare_context,
    _build_pipeline_context,
)

# Pipeline execution
from prellm.core.pipeline import (
    _execute_v3_pipeline,
    _run_preprocessing,
    _run_execution,
)

# System prompt building
from prellm.core.prompts import (
    _build_executor_system_prompt,
    _format_classification_context,
    _format_context_schema,
    _format_runtime_context,
    _format_user_context,
)

# Trace recording
from prellm.core.tracing import (
    _record_trace,
    _persist_session,
)

# State extraction and result building
from prellm.core.results import (
    _extract_classification_from_state,
    _extract_structure_from_state,
    _extract_sub_queries_from_state,
    _extract_missing_fields_from_state,
    _extract_matched_rule_from_state,
    _build_decomposition_result,
)

# Main entry point helpers
from prellm.core.main import (
    _resolve_pipeline_name,
    _load_config_overrides,
    _record_config_trace,
)

__all__ = [
    # Main
    "_resolve_pipeline_name",
    "_load_config_overrides",
    "_record_config_trace",
    # Context
    "_collect_user_context",
    "_collect_environment_context",
    "_compress_codebase_folder",
    "_generate_context_schema",
    "_build_sensitive_filter",
    "_initialize_context_components",
    "_prepare_context",
    "_build_pipeline_context",
    # Pipeline
    "_execute_v3_pipeline",
    "_run_preprocessing",
    "_run_execution",
    # Prompts
    "_build_executor_system_prompt",
    "_format_classification_context",
    "_format_context_schema",
    "_format_runtime_context",
    "_format_user_context",
    # Tracing
    "_record_trace",
    "_persist_session",
    # Results
    "_extract_classification_from_state",
    "_extract_structure_from_state",
    "_extract_sub_queries_from_state",
    "_extract_missing_fields_from_state",
    "_extract_matched_rule_from_state",
    "_build_decomposition_result",
]
