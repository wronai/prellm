"""Core Prellm — the main entry point for prompt analysis, enrichment, and LLM calls."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from prellm.analyzers.bias_detector import BiasDetector
from prellm.analyzers.context_engine import ContextEngine
from prellm.models import (
    AuditEntry,
    BiasPattern,
    GuardConfig,
    GuardResponse,
    ModelConfig,
    Policy,
)

logger = logging.getLogger("prellm")


class prellm:
    """Main Prellm middleware — analyze, enrich, and proxy LLM calls.

    Usage:
        guard = prellm("rules.yaml")
        result = await guard("Zdeployuj na produkcję", model="gpt-4o-mini")

    Or with inline config:
        guard = prellm(config=GuardConfig(policy=Policy.DEVOPS))
        result = await guard("Deploy the app")
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: GuardConfig | None = None,
    ):
        if config:
            self.config = config
        elif config_path:
            self.config = self._load_config(Path(config_path))
        else:
            self.config = GuardConfig()

        self.detector = BiasDetector(self.config.bias_patterns or None)
        self.context_engine = ContextEngine(self.config.context_sources)
        self.audit_log: list[AuditEntry] = []

    async def __call__(
        self,
        query: str,
        model: str | None = None,
        extra_context: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> GuardResponse:
        """Analyze query, enrich if needed, call LLM, validate response."""
        import litellm

        target_model = model or self.config.models.fallback[0]

        # Step 1: Analyze
        analysis = self.detector.analyze(query)
        analysis.original_query = query

        # Step 2: Enrich if needed
        if analysis.needs_clarify:
            enriched = self.config.clarify_template.format(query=query)
            enriched = self.context_engine.enrich_prompt(enriched, extra_context)
            analysis.enriched_query = enriched
        else:
            enriched = self.context_engine.enrich_prompt(query, extra_context)
            analysis.enriched_query = enriched

        # Step 3: Call LLM with retry/fallback
        response_content = ""
        model_used = target_model
        retries = 0

        fallback_models = [target_model] + [
            m for m in self.config.models.fallback if m != target_model
        ]

        for attempt_model in fallback_models:
            for attempt in range(self.config.max_retries):
                try:
                    resp = await litellm.acompletion(
                        model=attempt_model,
                        messages=[{"role": "user", "content": enriched}],
                        max_tokens=self.config.models.max_tokens,
                        timeout=self.config.models.timeout,
                        **kwargs,
                    )
                    response_content = resp.choices[0].message.content
                    model_used = attempt_model
                    break
                except Exception as e:
                    retries += 1
                    logger.warning(f"Attempt {attempt + 1} with {attempt_model} failed: {e}")
            if response_content:
                break

        # Step 4: Build response
        result = GuardResponse(
            content=response_content or "No response from any model.",
            clarified=analysis.needs_clarify,
            needs_more_context=analysis.needs_clarify and not response_content,
            model_used=model_used,
            analysis=analysis,
            retries=retries,
        )

        # Audit
        self._audit("query", query, result, model_used)

        return result

    def analyze_only(self, query: str) -> dict[str, Any]:
        """Run analysis without calling LLM — useful for dry-run / testing."""
        analysis = self.detector.analyze(query)
        return {
            "needs_clarify": analysis.needs_clarify,
            "patterns": analysis.detected_patterns,
            "ambiguity_flags": analysis.ambiguity_flags,
            "readability": analysis.readability_score,
            "enriched": self.config.clarify_template.format(query=query)
            if analysis.needs_clarify
            else query,
        }

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Return audit log as list of dicts."""
        return [entry.model_dump() for entry in self.audit_log]

    def _audit(self, action: str, query: str, response: GuardResponse, model: str) -> None:
        entry = AuditEntry(
            action=action,
            query=query,
            response_summary=response.content[:200] if response.content else "",
            model=model,
            policy=self.config.policy,
        )
        self.audit_log.append(entry)

    @staticmethod
    def _load_config(path: Path) -> GuardConfig:
        """Load config from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        # Normalize bias_patterns
        patterns = []
        for p in raw.get("bias_patterns", []):
            if isinstance(p, dict):
                patterns.append(BiasPattern(**p))

        models_raw = raw.get("models", {})
        models = ModelConfig(**models_raw) if isinstance(models_raw, dict) else ModelConfig()

        return GuardConfig(
            bias_patterns=patterns,
            clarify_template=raw.get("clarify_template", GuardConfig.model_fields["clarify_template"].default),
            max_retries=raw.get("max_retries", 3),
            policy=Policy(raw.get("policy", "strict")),
            models=models,
            context_sources=raw.get("context_sources", []),
        )
