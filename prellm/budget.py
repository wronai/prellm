"""Budget tracking — monitors and enforces spend limits for LLM API calls.

Tracks cost per request using LiteLLM's cost tracking, persists totals to
a JSON file in .prellm/, and raises BudgetExceededError when the monthly
limit is reached.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

logger = logging.getLogger("prellm.budget")


class BudgetExceededError(Exception):
    """Raised when the monthly budget limit has been reached."""

    def __init__(self, spent: float, limit: float, model: str = ""):
        self.spent = spent
        self.limit = limit
        self.model = model
        msg = f"Monthly budget exceeded: ${spent:.4f} / ${limit:.2f}"
        if model:
            msg += f" (blocked: {model})"
        super().__init__(msg)


@dataclass
class UsageEntry:
    """Single API call cost record."""
    timestamp: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


@dataclass
class BudgetTracker:
    """Tracks LLM API spend against a monthly budget.

    Usage:
        tracker = BudgetTracker(monthly_limit=50.0)
        tracker.check(model="gpt-4o")          # raises if over budget
        tracker.record(model="gpt-4o", cost=0.05, prompt_tokens=100, completion_tokens=50)
    """
    monthly_limit: float | None = None
    persist_path: Path = field(default_factory=lambda: Path(".prellm") / "budget.json")
    _entries: list[UsageEntry] = field(default_factory=list)
    _total_cost: float = 0.0
    _lock: RLock = field(default_factory=RLock)
    _loaded: bool = False

    def _ensure_loaded(self) -> None:
        """Lazy-load persisted budget data."""
        if self._loaded:
            return
        self._loaded = True
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text())
                month_key = _current_month_key()
                month_data = data.get(month_key, {})
                self._total_cost = month_data.get("total_cost", 0.0)
                for entry in month_data.get("entries", []):
                    self._entries.append(UsageEntry(**entry))
                logger.debug(f"Budget loaded: ${self._total_cost:.4f} for {month_key}")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to load budget data: {e}")

    def check(self, model: str = "") -> None:
        """Check if budget allows another request. Raises BudgetExceededError if not."""
        if self.monthly_limit is None:
            return
        with self._lock:
            self._ensure_loaded()
            if self._total_cost >= self.monthly_limit:
                raise BudgetExceededError(self._total_cost, self.monthly_limit, model)

    def record(
        self,
        model: str,
        cost: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record a completed API call cost."""
        entry = UsageEntry(
            timestamp=datetime.now().isoformat(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
        )
        with self._lock:
            self._ensure_loaded()
            self._entries.append(entry)
            self._total_cost += cost
            self._persist()
            logger.debug(f"Budget record: {model} ${cost:.4f} (total: ${self._total_cost:.4f})")

    def record_from_response(self, response: Any, model: str = "") -> None:
        """Extract cost from a LiteLLM response and record it.

        LiteLLM responses include usage info in response.usage and
        cost can be computed via litellm.completion_cost().
        """
        try:
            import litellm
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        model_used = model or getattr(response, "model", "unknown")

        self.record(
            model=model_used,
            cost=cost,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    @property
    def total_cost(self) -> float:
        with self._lock:
            self._ensure_loaded()
            return self._total_cost

    @property
    def remaining(self) -> float | None:
        if self.monthly_limit is None:
            return None
        return max(0.0, self.monthly_limit - self.total_cost)

    @property
    def entries(self) -> list[UsageEntry]:
        with self._lock:
            self._ensure_loaded()
            return list(self._entries)

    def summary(self) -> dict[str, Any]:
        """Return budget summary as dict."""
        with self._lock:
            self._ensure_loaded()
            # Group by model
            by_model: dict[str, float] = {}
            for e in self._entries:
                by_model[e.model] = by_model.get(e.model, 0.0) + e.cost
            return {
                "month": _current_month_key(),
                "total_cost": self._total_cost,
                "monthly_limit": self.monthly_limit,
                "remaining": self.remaining,
                "requests": len(self._entries),
                "by_model": by_model,
            }

    def _persist(self) -> None:
        """Save current month's data to JSON file."""
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)

            # Load existing data (other months)
            data: dict[str, Any] = {}
            if self.persist_path.exists():
                try:
                    data = json.loads(self.persist_path.read_text())
                except (json.JSONDecodeError, TypeError):
                    data = {}

            month_key = _current_month_key()
            # Only keep last 10 entries in file to avoid bloat
            recent_entries = self._entries[-100:] if len(self._entries) > 100 else self._entries
            data[month_key] = {
                "total_cost": self._total_cost,
                "entries": [
                    {
                        "timestamp": e.timestamp,
                        "model": e.model,
                        "prompt_tokens": e.prompt_tokens,
                        "completion_tokens": e.completion_tokens,
                        "cost": e.cost,
                    }
                    for e in recent_entries
                ],
            }
            self.persist_path.write_text(json.dumps(data, indent=2))
        except OSError as e:
            logger.warning(f"Failed to persist budget data: {e}")

    def reset(self) -> None:
        """Reset current month's budget (for testing)."""
        with self._lock:
            self._entries.clear()
            self._total_cost = 0.0
            self._persist()


def _current_month_key() -> str:
    return datetime.now().strftime("%Y-%m")


# ─── Singleton ───────────────────────────────────────────────────────────────

_global_tracker: BudgetTracker | None = None


def get_budget_tracker(monthly_limit: float | None = None, persist_path: Path | None = None) -> BudgetTracker:
    """Get or create the global budget tracker singleton."""
    global _global_tracker
    if _global_tracker is None:
        kwargs: dict[str, Any] = {}
        if monthly_limit is not None:
            kwargs["monthly_limit"] = monthly_limit
        if persist_path is not None:
            kwargs["persist_path"] = persist_path
        _global_tracker = BudgetTracker(**kwargs)
    elif monthly_limit is not None and _global_tracker.monthly_limit != monthly_limit:
        _global_tracker.monthly_limit = monthly_limit
    return _global_tracker


def reset_budget_tracker() -> None:
    """Reset the global tracker (for testing)."""
    global _global_tracker
    _global_tracker = None
