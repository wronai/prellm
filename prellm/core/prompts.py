"""System prompt building helpers for preLLM preprocessing."""

from __future__ import annotations

from typing import Any


def _format_classification_context(prep_result: Any) -> list[str]:
    """Extract and format classification context from preprocessing result."""
    parts: list[str] = []
    
    if not prep_result.decomposition:
        return parts
    
    state = prep_result.decomposition.state
    classification = state.get("classification")
    
    if isinstance(classification, dict):
        intent = classification.get("intent", "unknown")
        confidence = classification.get("confidence", 0)
        domain = classification.get("domain", "general")
        parts.append(
            f"User intent: {intent} (confidence: {confidence}, domain: {domain})"
        )
    
    matched_rule = state.get("matched_rule")
    if isinstance(matched_rule, dict) and matched_rule.get("name"):
        parts.append(f"Matched domain rule: {matched_rule['name']}")
        if matched_rule.get("required_fields"):
            parts.append(f"Required fields: {', '.join(matched_rule['required_fields'])}")
    
    return parts


def _format_context_schema(extra_context: dict[str, Any]) -> list[str]:
    """Extract and format context schema information."""
    parts: list[str] = []
    
    ctx_schema = extra_context.get("context_schema")
    if not ctx_schema:
        return parts
    
    try:
        import json
        schema_data = json.loads(ctx_schema) if isinstance(ctx_schema, str) else ctx_schema
        
        tools = schema_data.get("available_tools", [])
        if tools:
            parts.append(f"Available tools on user's system: {', '.join(tools[:15])}")
        
        platform = schema_data.get("platform")
        if platform:
            parts.append(f"Platform: {platform}")
        
        locale = schema_data.get("locale")
        if locale:
            parts.append(f"Locale: {locale}")
    except Exception:
        pass
    
    return parts


def _format_runtime_context(extra_context: dict[str, Any]) -> list[str]:
    """Extract and format runtime context information."""
    parts: list[str] = []
    
    runtime = extra_context.get("runtime_context")
    if not isinstance(runtime, dict):
        return parts
    
    sys_info = runtime.get("system", {})
    proc_info = runtime.get("process", {})
    
    if sys_info.get("os"):
        parts.append(f"OS: {sys_info['os']} {sys_info.get('arch', '')}")
    
    if sys_info.get("python"):
        parts.append(f"Python: {sys_info['python']}")
    
    if proc_info.get("cwd"):
        parts.append(f"Working directory: {proc_info['cwd']}")
    
    return parts


def _format_user_context(extra_context: dict[str, Any]) -> list[str]:
    """Extract and format user context information."""
    parts: list[str] = []
    
    user_ctx = extra_context.get("user_context")
    if user_ctx:
        parts.append(f"User context: {user_ctx}")
    
    return parts


def _build_executor_system_prompt(
    prep_result: Any,
    extra_context: dict[str, Any],
) -> str:
    """Build a system prompt for the large LLM from preprocessing results and context.

    Injects classification, context schema, and runtime info so the large LLM
    understands the user's intent and environment.
    """
    # Collect all context sections
    sections = [
        _format_classification_context(prep_result),
        _format_context_schema(extra_context),
        _format_runtime_context(extra_context),
        _format_user_context(extra_context),
    ]
    
    # Flatten all parts
    parts: list[str] = []
    for section in sections:
        parts.extend(section)
    
    if not parts:
        return ""
    
    return "Context from preprocessing:\n" + "\n".join(f"- {p}" for p in parts)
