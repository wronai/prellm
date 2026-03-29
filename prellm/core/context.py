"""Context collection helpers for preLLM preprocessing pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("prellm")


def _collect_user_context(user_context: str | dict[str, str] | None) -> dict[str, Any]:
    """Process and normalize user context."""
    extra_context: dict[str, Any] = {}
    if isinstance(user_context, str) and user_context:
        extra_context["user_context"] = user_context
    elif isinstance(user_context, dict):
        extra_context.update(user_context)
    return extra_context


def _collect_environment_context(collect_env: bool) -> tuple[Any, dict[str, Any]]:
    """Collect shell and runtime context if requested."""
    extra_context: dict[str, Any] = {}
    shell_ctx = None
    
    if collect_env:
        try:
            from prellm.context.shell_collector import ShellContextCollector
            shell_ctx = ShellContextCollector().collect_all()
            extra_context["shell_context"] = shell_ctx.model_dump_json()
        except Exception as e:
            logger.warning(f"Shell context collection failed: {e}")
        
        # Also collect RuntimeContext
        try:
            from prellm.analyzers.context_engine import ContextEngine
            runtime_ctx = ContextEngine().gather_runtime()
            extra_context["runtime_context"] = runtime_ctx.model_dump()
        except Exception as e:
            logger.warning(f"RuntimeContext collection failed: {e}")
    
    return shell_ctx, extra_context


def _compress_codebase_folder(compress_folder: bool, codebase_path: str | Path | None) -> tuple[Any, dict[str, Any]]:
    """Compress codebase folder if requested."""
    extra_context: dict[str, Any] = {}
    compressed = None
    
    if compress_folder and codebase_path:
        try:
            from prellm.context.folder_compressor import FolderCompressor
            compressed = FolderCompressor().compress(codebase_path)
            extra_context["folder_compressed"] = compressed.toon_output
        except Exception as e:
            logger.warning(f"Folder compression failed: {e}")
    
    return compressed, extra_context


def _generate_context_schema(collect_env: bool, compress_folder: bool, shell_ctx: Any, compressed: Any) -> dict[str, Any]:
    """Generate context schema if environment collection or compression is enabled."""
    extra_context: dict[str, Any] = {}
    
    if collect_env or compress_folder:
        try:
            from prellm.context.schema_generator import ContextSchemaGenerator
            env_schema = ContextSchemaGenerator().generate(
                shell_context=shell_ctx,
                folder_compressed=compressed,
            )
            extra_context["context_schema"] = env_schema.model_dump_json()
        except Exception as e:
            logger.warning(f"Context schema generation failed: {e}")
    
    return extra_context


def _build_sensitive_filter(sanitize: bool, sensitive_rules: str | Path | None, extra_context: dict[str, Any]) -> Any:
    """Build and apply sensitive data filter if sanitization is enabled."""
    sensitive_filter = None
    
    if sanitize:
        try:
            from prellm.context.sensitive_filter import SensitiveDataFilter
            sensitive_filter = SensitiveDataFilter(
                rules_path=sensitive_rules if sensitive_rules else None,
            )
            filtered = sensitive_filter.filter_context_for_large_llm(extra_context)
            extra_context.clear()
            extra_context.update(filtered)
        except Exception as e:
            logger.warning(f"Sensitive filtering failed: {e}")
    
    return sensitive_filter


def _initialize_context_components(memory_path: str | Path | None, codebase_path: str | Path | None) -> tuple[Any, Any]:
    """Initialize optional context enrichment components."""
    user_memory = None
    if memory_path:
        try:
            from prellm.context.user_memory import UserMemory
            user_memory = UserMemory(path=memory_path)
        except Exception as e:
            logger.warning(f"Failed to initialize UserMemory: {e}")
    
    codebase_indexer = None
    if codebase_path:
        try:
            from prellm.context.codebase_indexer import CodebaseIndexer
            codebase_indexer = CodebaseIndexer()
        except Exception as e:
            logger.warning(f"Failed to initialize CodebaseIndexer: {e}")
    
    return user_memory, codebase_indexer


def _prepare_context(
    user_context: str | dict[str, str] | None,
    domain_rules: list[dict[str, Any]] | None,
    collect_env: bool,
    compress_folder: bool,
    codebase_path: str | Path | None,
    sanitize: bool,
    sensitive_rules: str | Path | None,
    memory_path: str | Path | None,
) -> tuple[dict[str, Any], Any, Any, Any]:
    """Gather all context: env, codebase, schema, sensitive filter, memory, indexer.

    Returns (extra_context, sensitive_filter, user_memory, codebase_indexer).
    """
    # Collect and normalize user context
    extra_context = _collect_user_context(user_context)
    
    # Add domain rules if provided
    if domain_rules:
        extra_context["domain_rules"] = domain_rules
    
    # Collect environment context
    shell_ctx, env_context = _collect_environment_context(collect_env)
    extra_context.update(env_context)
    
    # Compress codebase folder
    compressed, compress_context = _compress_codebase_folder(compress_folder, codebase_path)
    extra_context.update(compress_context)
    
    # Generate context schema
    schema_context = _generate_context_schema(collect_env, compress_folder, shell_ctx, compressed)
    extra_context.update(schema_context)
    
    # Build and apply sensitive filter
    sensitive_filter = _build_sensitive_filter(sanitize, sensitive_rules, extra_context)
    
    # Initialize optional components
    user_memory, codebase_indexer = _initialize_context_components(memory_path, codebase_path)
    
    return extra_context, sensitive_filter, user_memory, codebase_indexer


def _build_pipeline_context(extra_context: dict[str, Any]) -> dict[str, Any]:
    """Build compact context for small LLM pipeline — strips raw blobs.

    The small LLM only needs:
    - context_schema (compact summary of environment)
    - domain_rules (if any)
    - user_context (if any)
    - folder_compressed (toon format, if any)

    Raw shell_context and runtime_context are kept only in extra_context
    for _build_executor_system_prompt (large LLM).
    """
    # Keys that are too large / not useful for the small LLM pipeline
    _LARGE_BLOB_KEYS = {"shell_context", "runtime_context"}

    return {k: v for k, v in extra_context.items() if k not in _LARGE_BLOB_KEYS}
