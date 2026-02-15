"""BiasDetector — scans queries for bias patterns, ambiguity, and readability issues."""

from __future__ import annotations

import re
from typing import Any

from promptguard.models import AnalysisResult, BiasPattern


# Default bias/ambiguity patterns for Polish & English DevOps context
DEFAULT_PATTERNS: list[dict[str, str]] = [
    {"regex": r"(zawsze|always)\s+\w+", "action": "clarify", "severity": "medium", "description": "Absolute quantifier"},
    {"regex": r"(tylko|only|just)\s+\w+", "action": "clarify", "severity": "low", "description": "Exclusive quantifier"},
    {"regex": r"(głupi|stupid|dumb)\s+\w+", "action": "flag", "severity": "high", "description": "Derogatory language"},
    {"regex": r"(zdeployuj|deploy|push)\s+(na|to)\s+(prod|production)", "action": "clarify", "severity": "critical",
     "description": "Production deployment without context"},
    {"regex": r"(usuń|delete|drop|remove)\s+(baz[ęa]|database|db|table)", "action": "clarify", "severity": "critical",
     "description": "Destructive DB operation"},
    {"regex": r"(restart|reboot|kill)\s+(server|service|pod|container)", "action": "clarify", "severity": "high",
     "description": "Service disruption command"},
    {"regex": r"(migrate|migruj)\s+.*(?:prod|production)", "action": "clarify", "severity": "critical",
     "description": "Production migration"},
    {"regex": r"(skaluj|scale)\s+(down|up|to\s+\d+)", "action": "clarify", "severity": "high",
     "description": "Scaling operation"},
    {"regex": r"(zmień|change|update)\s+(config|konfigurację|env|secret)", "action": "clarify", "severity": "high",
     "description": "Configuration change"},
]


class BiasDetector:
    """Detects bias, ambiguity, and dangerous patterns in queries.

    Uses regex patterns (configurable via YAML) + optional NLTK readability scoring.
    Designed for both general prompt safety and DevOps-specific guardrails.
    """

    def __init__(self, patterns: list[BiasPattern] | None = None):
        if patterns:
            self.patterns = patterns
        else:
            self.patterns = [BiasPattern(**p) for p in DEFAULT_PATTERNS]

        self._readability_available = False
        try:
            import textstat  # noqa: F401
            self._readability_available = True
        except ImportError:
            pass

    def analyze(self, query: str) -> AnalysisResult:
        """Analyze a query for bias patterns and ambiguity."""
        detected: list[str] = []
        ambiguity_flags: list[str] = []
        needs_clarify = False

        for pattern in self.patterns:
            if re.search(pattern.regex, query, re.IGNORECASE):
                detected.append(f"[{pattern.severity}] {pattern.description}")
                if pattern.action == "clarify":
                    needs_clarify = True
                    ambiguity_flags.append(pattern.description)

        # Check for very short / context-free queries
        word_count = len(query.split())
        if word_count < 4:
            needs_clarify = True
            ambiguity_flags.append("Query too short — likely missing context")

        # Check for missing subject/object in DevOps context
        devops_verbs = ["deploy", "zdeployuj", "migrate", "migruj", "scale", "skaluj",
                        "restart", "delete", "usuń", "push", "rollback"]
        has_devops_verb = any(v in query.lower() for v in devops_verbs)
        has_target = any(t in query.lower() for t in [
            "staging", "production", "prod", "dev", "test", "cluster", "namespace"])
        if has_devops_verb and not has_target:
            needs_clarify = True
            ambiguity_flags.append("DevOps command without target environment")

        # Readability score
        readability = None
        if self._readability_available:
            try:
                import textstat
                readability = textstat.flesch_reading_ease(query)
            except Exception:
                pass

        return AnalysisResult(
            needs_clarify=needs_clarify,
            detected_patterns=detected,
            enriched_query="",
            original_query=query,
            readability_score=readability,
            ambiguity_flags=ambiguity_flags,
        )
