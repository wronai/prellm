"""SensitiveDataFilter — classifies and filters sensitive data before LLM calls.

Ensures API keys, tokens, passwords never reach the large LLM.
Supports key-based and value-based pattern matching.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from prellm.models import FilterReport, SensitivityLevel

logger = logging.getLogger("prellm.context.sensitive_filter")

# Built-in sensitive key patterns
_DEFAULT_SENSITIVE_KEY_PATTERNS = [
    re.compile(r"API_KEY", re.IGNORECASE),
    re.compile(r"SECRET", re.IGNORECASE),
    re.compile(r"TOKEN", re.IGNORECASE),
    re.compile(r"PASSWORD", re.IGNORECASE),
    re.compile(r"PRIVATE_KEY", re.IGNORECASE),
    re.compile(r"CREDENTIAL", re.IGNORECASE),
    re.compile(r"AUTH_KEY", re.IGNORECASE),
]

# Built-in sensitive value patterns (detect tokens by format)
_DEFAULT_SENSITIVE_VALUE_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),          # OpenAI
    re.compile(r"sk-ant-[a-zA-Z0-9]{20,}"),       # Anthropic
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),            # GitHub PAT
    re.compile(r"gsk_[a-zA-Z0-9]{20,}"),           # Groq
    re.compile(r"sk-or-v1-[a-zA-Z0-9]{20,}"),     # OpenRouter
    re.compile(r"xox[bpsa]-[a-zA-Z0-9\-]{20,}"),  # Slack
]

# Keys that are masked (partially shown) rather than blocked
_DEFAULT_MASKED_PATTERNS = [
    re.compile(r"DATABASE_URL", re.IGNORECASE),
    re.compile(r"REDIS_URL", re.IGNORECASE),
    re.compile(r"SMTP_", re.IGNORECASE),
    re.compile(r"MONGO_URI", re.IGNORECASE),
]

# Keys that are always safe
_DEFAULT_SAFE_KEYS = {
    "LANG", "TERM", "SHELL", "HOME", "USER", "PWD", "PATH",
    "EDITOR", "VISUAL", "HOSTNAME", "COLUMNS", "LINES", "SHLVL",
    "TZ", "LC_ALL", "LC_CTYPE", "VIRTUAL_ENV", "PYTHONPATH",
}

_DEFAULT_RULES_PATH = Path(__file__).parent.parent.parent / "configs" / "sensitive_rules.yaml"


class SensitiveDataFilter:
    """Classifies and filters sensitive data from context before LLM calls."""

    def __init__(
        self,
        rules_path: str | Path | None = None,
        extra_blocked: list[str] | None = None,
        extra_safe: set[str] | None = None,
    ):
        self._sensitive_key_patterns = list(_DEFAULT_SENSITIVE_KEY_PATTERNS)
        self._sensitive_value_patterns = list(_DEFAULT_SENSITIVE_VALUE_PATTERNS)
        self._masked_patterns = list(_DEFAULT_MASKED_PATTERNS)
        self._safe_keys = set(_DEFAULT_SAFE_KEYS)
        self._report = FilterReport()

        if extra_safe:
            self._safe_keys |= extra_safe
        if extra_blocked:
            for pat in extra_blocked:
                self._sensitive_key_patterns.append(re.compile(pat, re.IGNORECASE))

        # Load custom rules from YAML if available
        rp = Path(rules_path) if rules_path else _DEFAULT_RULES_PATH
        if rp.is_file():
            self._load_rules(rp)

    def _load_rules(self, path: Path) -> None:
        """Load custom sensitive rules from YAML."""
        try:
            with open(path) as f:
                raw = yaml.safe_load(f) or {}

            # Blocked key patterns
            for pat in raw.get("sensitive_keys", {}).get("blocked", []):
                self._sensitive_key_patterns.append(re.compile(pat, re.IGNORECASE))

            # Masked key patterns
            for pat in raw.get("sensitive_keys", {}).get("masked", []):
                self._masked_patterns.append(re.compile(pat, re.IGNORECASE))

            # Safe keys
            for key in raw.get("sensitive_keys", {}).get("safe", []):
                self._safe_keys.add(key)

            # Value patterns
            for pat_str in raw.get("sensitive_value_patterns", []):
                self._sensitive_value_patterns.append(re.compile(pat_str))

        except Exception as e:
            logger.warning(f"Failed to load sensitive rules from {path}: {e}")

    def classify_key(self, key: str) -> SensitivityLevel:
        """Classify a key name as SAFE, MASKED, or BLOCKED."""
        if key in self._safe_keys:
            return SensitivityLevel.SAFE

        for pattern in self._sensitive_key_patterns:
            if pattern.search(key):
                return SensitivityLevel.BLOCKED

        for pattern in self._masked_patterns:
            if pattern.search(key):
                return SensitivityLevel.MASKED

        return SensitivityLevel.SAFE

    def classify_value(self, value: str) -> SensitivityLevel:
        """Classify a value by checking against known token/key patterns."""
        for pattern in self._sensitive_value_patterns:
            if pattern.search(value):
                return SensitivityLevel.BLOCKED
        return SensitivityLevel.SAFE

    def filter_dict(
        self, data: dict[str, str], level: SensitivityLevel = SensitivityLevel.MASKED
    ) -> dict[str, str]:
        """Filter a dict — mask or remove sensitive entries.

        Args:
            data: Input dict to filter.
            level: Minimum sensitivity to act on. MASKED masks values, BLOCKED removes them.
        """
        self._report = FilterReport(total_processed=len(data))
        result: dict[str, str] = {}

        for key, value in data.items():
            key_level = self.classify_key(key)
            val_level = self.classify_value(str(value))

            # Use the more restrictive classification
            effective = max(
                [key_level, val_level],
                key=lambda x: [SensitivityLevel.SAFE, SensitivityLevel.MASKED, SensitivityLevel.BLOCKED].index(x),
            )

            if effective == SensitivityLevel.BLOCKED:
                self._report.blocked_keys.append(key)
                if level == SensitivityLevel.MASKED:
                    # Even at MASKED level, blocked keys are removed
                    continue
                continue
            elif effective == SensitivityLevel.MASKED:
                self._report.masked_keys.append(key)
                if level in (SensitivityLevel.MASKED, SensitivityLevel.BLOCKED):
                    result[key] = self._mask_value(value)
                    continue
            else:
                self._report.safe_keys.append(key)

            result[key] = value

        return result

    def filter_context_for_large_llm(self, context: dict[str, Any]) -> dict[str, Any]:
        """Strict filter: removes EVERYTHING that could be sensitive before large LLM.

        Recursively filters nested dicts and sanitizes string values.
        """
        return self._filter_recursive(context)

    def sanitize_text(self, text: str) -> str:
        """Find and redact tokens/keys embedded in free text."""
        result = text
        for pattern in self._sensitive_value_patterns:
            result = pattern.sub("[REDACTED]", result)
        return result

    def get_filter_report(self) -> FilterReport:
        """Return report of what was blocked/masked in the last filter_dict call."""
        return self._report

    def _filter_recursive(self, data: Any) -> Any:
        """Recursively filter a data structure."""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                str_key = str(key)
                # Only apply key-based classification to env-var-style keys (ALL_CAPS)
                if self._looks_like_env_var(str_key):
                    key_level = self.classify_key(str_key)
                    if key_level == SensitivityLevel.BLOCKED:
                        continue
                    val_str = str(value) if not isinstance(value, (dict, list)) else ""
                    if val_str and self.classify_value(val_str) == SensitivityLevel.BLOCKED:
                        continue
                    if key_level == SensitivityLevel.MASKED:
                        if isinstance(value, str):
                            result[key] = self._mask_value(value)
                        else:
                            result[key] = self._filter_recursive(value)
                        continue
                else:
                    # For non-env-var keys, sanitize value rather than dropping key
                    val_str = str(value) if not isinstance(value, (dict, list)) else ""
                    if val_str and self.classify_value(val_str) == SensitivityLevel.BLOCKED:
                        if isinstance(value, str):
                            result[key] = self.sanitize_text(value)
                            continue
                        continue
                result[key] = self._filter_recursive(value)
            return result
        elif isinstance(data, list):
            return [self._filter_recursive(item) for item in data]
        elif isinstance(data, str):
            return self.sanitize_text(data)
        return data

    @staticmethod
    def _looks_like_env_var(key: str) -> bool:
        """Check if a key looks like an environment variable name (ALL_CAPS with underscores)."""
        if not key:
            return False
        return key == key.upper() and key.replace("_", "").isalnum()

    @staticmethod
    def _mask_value(value: str) -> str:
        """Mask a sensitive value, keeping first 2 and last 2 chars."""
        if len(value) <= 6:
            return "***"
        return value[:2] + "***" + value[-2:]
