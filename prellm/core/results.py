"""Result extraction and building helpers for preLLM preprocessing."""

from __future__ import annotations

from typing import Any

from prellm.models import (
    ClassificationResult,
    DecompositionResult,
    DecompositionStrategy,
    StructureResult,
)


def _extract_classification_from_state(state: dict) -> ClassificationResult | None:
    """Extract classification result from pipeline state."""
    classification = state.get("classification")
    if isinstance(classification, dict):
        return ClassificationResult(
            intent=classification.get("intent", "unknown"),
            confidence=float(classification.get("confidence", 0.0)),
            domain=classification.get("domain", "general"),
        )
    return None


def _extract_structure_from_state(state: dict) -> StructureResult | None:
    """Extract structure result from pipeline state."""
    fields = state.get("fields")
    if isinstance(fields, dict):
        return StructureResult(
            action=fields.get("action", ""),
            target=fields.get("target", ""),
            parameters=fields.get("parameters", {}),
        )
    return None


def _extract_sub_queries_from_state(state: dict) -> list[str]:
    """Extract sub-queries from pipeline state."""
    sub_queries = state.get("sub_queries")
    
    if isinstance(sub_queries, dict) and "sub_queries" in sub_queries:
        return [str(q) for q in sub_queries["sub_queries"]]
    elif isinstance(sub_queries, list):
        return [str(q) for q in sub_queries]
    
    return []


def _extract_missing_fields_from_state(state: dict) -> list[str]:
    """Extract missing fields from pipeline state."""
    missing_fields = state.get("missing_fields")
    if isinstance(missing_fields, list):
        return missing_fields
    return []


def _extract_matched_rule_from_state(state: dict, current_missing_fields: list[str]) -> tuple[str | None, list[str]]:
    """Extract matched rule and missing fields from pipeline state."""
    matched_rule = state.get("matched_rule")
    
    if isinstance(matched_rule, dict) and "name" in matched_rule:
        rule_name = matched_rule["name"]
        
        # Also extract missing fields from rule matching if not already present
        if not current_missing_fields and matched_rule.get("required_fields"):
            missing_fields = matched_rule["required_fields"]
        else:
            missing_fields = current_missing_fields
        
        return rule_name, missing_fields
    
    return None, current_missing_fields


def _build_decomposition_result(
    query: str,
    pipeline_name: str,
    prep_result: Any,
) -> DecompositionResult | None:
    """Build a backward-compatible DecompositionResult from pipeline state."""
    if not prep_result.decomposition:
        return None

    state = prep_result.decomposition.state
    strategy_values = [s.value for s in DecompositionStrategy]
    strategy = DecompositionStrategy(pipeline_name) if pipeline_name in strategy_values else DecompositionStrategy.CLASSIFY

    result = DecompositionResult(
        strategy=strategy,
        original_query=query,
        composed_prompt=prep_result.executor_input,
    )

    # Extract all components from state
    result.classification = _extract_classification_from_state(state)
    result.structure = _extract_structure_from_state(state)
    result.sub_queries = _extract_sub_queries_from_state(state)
    result.missing_fields = _extract_missing_fields_from_state(state)
    
    # Extract matched rule and update missing fields
    matched_rule, missing_fields = _extract_matched_rule_from_state(state, result.missing_fields)
    result.matched_rule = matched_rule
    result.missing_fields = missing_fields

    return result
