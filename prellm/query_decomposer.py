"""QueryDecomposer — small LLM preprocessor for query decomposition.

Uses a small model (≤3B params) to classify, structure, split, enrich,
or passthrough user queries before routing to a large LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from prellm.llm_provider import LLMProvider
from prellm.models import (
    ClassificationResult,
    DecompositionResult,
    DecompositionStrategy,
    DomainRule,
    LLMProviderConfig,
    DecompositionPrompts,
    StructureResult,
)

logger = logging.getLogger("prellm.decomposer")


class QueryDecomposer:
    """Decomposes user queries using a small LLM before routing to a large model.

    Supports 5 strategies:
        - CLASSIFY: Classify intent and domain
        - STRUCTURE: Extract structured fields (action, target, params)
        - SPLIT: Break complex query into sub-queries
        - ENRICH: Add missing context and compose complete prompt
        - PASSTHROUGH: No decomposition, forward as-is

    Usage:
        decomposer = QueryDecomposer(
            small_llm=LLMProvider(LLMProviderConfig(model="phi3:mini")),
            prompts=DecompositionPrompts(),
            domain_rules=[...],
        )
        result = await decomposer.decompose("Zdeployuj apkę na prod", strategy=DecompositionStrategy.CLASSIFY)
    """

    def __init__(
        self,
        small_llm: LLMProvider,
        prompts: DecompositionPrompts | None = None,
        domain_rules: list[DomainRule] | None = None,
    ):
        self.small_llm = small_llm
        self.prompts = prompts or DecompositionPrompts()
        self.domain_rules = domain_rules or []

    async def decompose(
        self,
        query: str,
        strategy: DecompositionStrategy = DecompositionStrategy.CLASSIFY,
        context: dict[str, str] | None = None,
    ) -> DecompositionResult:
        """Run the decomposition pipeline for the given strategy.

        Args:
            query: The raw user query.
            strategy: Which decomposition strategy to use.
            context: Optional runtime context (env, git, system).

        Returns:
            DecompositionResult with all extracted information.
        """
        result = DecompositionResult(
            strategy=strategy,
            original_query=query,
        )

        if strategy == DecompositionStrategy.PASSTHROUGH:
            result.composed_prompt = query
            return result

        # Step 1: Always classify first (except passthrough)
        classification = await self._classify(query)
        result.classification = classification
        result.raw_small_llm_outputs["classify"] = str(classification.model_dump())

        # Step 2: Match domain rules
        matched_rule = self._match_domain_rule(query, classification)
        if matched_rule:
            result.matched_rule = matched_rule.name
            result.missing_fields = self._find_missing_fields(query, matched_rule, context)

        # Step 3: Strategy-specific processing
        if strategy == DecompositionStrategy.CLASSIFY:
            result.composed_prompt = await self._compose(query, result, context)

        elif strategy == DecompositionStrategy.STRUCTURE:
            structure = await self._structure(query)
            result.structure = structure
            result.raw_small_llm_outputs["structure"] = str(structure.model_dump())
            result.composed_prompt = await self._compose(query, result, context)

        elif strategy == DecompositionStrategy.SPLIT:
            sub_queries = await self._split(query)
            result.sub_queries = sub_queries
            result.raw_small_llm_outputs["split"] = str(sub_queries)
            result.composed_prompt = await self._compose(query, result, context)

        elif strategy == DecompositionStrategy.ENRICH:
            result.composed_prompt = await self._enrich(query, result, context)
            result.raw_small_llm_outputs["enrich"] = result.composed_prompt

        return result

    async def _classify(self, query: str) -> ClassificationResult:
        """Classify the query intent and domain using the small LLM."""
        try:
            data = await self.small_llm.complete_json(
                user_message=query,
                system_prompt=self.prompts.classify_prompt,
            )
            return ClassificationResult(
                intent=data.get("intent", "unknown"),
                confidence=float(data.get("confidence", 0.0)),
                domain=data.get("domain", "general"),
            )
        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return ClassificationResult(intent="unknown", confidence=0.0, domain="general")

    async def _structure(self, query: str) -> StructureResult:
        """Extract structured fields from the query."""
        try:
            data = await self.small_llm.complete_json(
                user_message=query,
                system_prompt=self.prompts.structure_prompt,
            )
            return StructureResult(
                action=data.get("action", ""),
                target=data.get("target", ""),
                parameters=data.get("parameters", {}),
            )
        except Exception as e:
            logger.warning(f"Structure extraction failed: {e}")
            return StructureResult()

    async def _split(self, query: str) -> list[str]:
        """Split a complex query into sub-queries."""
        try:
            data = await self.small_llm.complete_json(
                user_message=query,
                system_prompt=self.prompts.split_prompt,
            )
            sub_queries = data.get("sub_queries", [])
            if isinstance(sub_queries, list):
                return [str(q) for q in sub_queries]
            return [query]
        except Exception as e:
            logger.warning(f"Query split failed: {e}")
            return [query]

    async def _enrich(
        self,
        query: str,
        result: DecompositionResult,
        context: dict[str, str] | None,
    ) -> str:
        """Enrich the query with missing context using the small LLM."""
        context_str = ""
        if context:
            context_str = "\n".join(f"  {k}: {v}" for k, v in context.items())

        missing_str = ", ".join(result.missing_fields) if result.missing_fields else "none identified"

        user_msg = (
            f"Original query: {query}\n"
            f"Missing fields: {missing_str}\n"
        )
        if context_str:
            user_msg += f"Available context:\n{context_str}\n"
        if result.classification:
            user_msg += f"Intent: {result.classification.intent} (confidence: {result.classification.confidence})\n"

        try:
            return await self.small_llm.complete(
                user_message=user_msg,
                system_prompt=self.prompts.enrich_prompt,
            )
        except Exception as e:
            logger.warning(f"Enrichment failed: {e}")
            return query

    async def _compose(
        self,
        query: str,
        result: DecompositionResult,
        context: dict[str, str] | None,
    ) -> str:
        """Compose the final prompt for the large LLM."""
        parts = [f"Original query: {query}"]

        if result.classification:
            parts.append(
                f"Classification: intent={result.classification.intent}, "
                f"confidence={result.classification.confidence}, "
                f"domain={result.classification.domain}"
            )

        if result.structure:
            parts.append(
                f"Structure: action={result.structure.action}, "
                f"target={result.structure.target}, "
                f"params={result.structure.parameters}"
            )

        if result.sub_queries:
            parts.append(f"Sub-queries: {result.sub_queries}")

        if result.missing_fields:
            parts.append(f"Missing fields: {result.missing_fields}")

        if result.matched_rule:
            parts.append(f"Matched domain rule: {result.matched_rule}")

        if context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in context.items())
            parts.append(f"Context: {ctx_str}")

        user_msg = "\n".join(parts)

        try:
            return await self.small_llm.complete(
                user_message=user_msg,
                system_prompt=self.prompts.compose_prompt,
            )
        except Exception as e:
            logger.warning(f"Composition failed, using enriched query: {e}")
            return query

    def _match_domain_rule(
        self,
        query: str,
        classification: ClassificationResult,
    ) -> DomainRule | None:
        """Find the best matching domain rule for the query."""
        query_lower = query.lower()

        best_match: DomainRule | None = None
        best_score = 0

        for rule in self.domain_rules:
            score = 0

            # Keyword matching
            for keyword in rule.keywords:
                if keyword.lower() in query_lower:
                    score += 1

            # Intent matching
            if rule.intent and rule.intent == classification.intent:
                score += 2

            if score > best_score:
                best_score = score
                best_match = rule

        return best_match if best_score > 0 else None

    @staticmethod
    def _find_missing_fields(
        query: str,
        rule: DomainRule,
        context: dict[str, str] | None,
    ) -> list[str]:
        """Determine which required fields are missing from the query and context."""
        query_lower = query.lower()
        context_keys = set((context or {}).keys())

        missing = []
        for field in rule.required_fields:
            field_lower = field.lower().replace("_", " ")
            if field_lower not in query_lower and field not in context_keys:
                missing.append(field)

        return missing
