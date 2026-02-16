"""Tests for prellm.budget module — spend tracking and enforcement."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from prellm.budget import (
    BudgetExceededError,
    BudgetTracker,
    UsageEntry,
    get_budget_tracker,
    reset_budget_tracker,
)


class TestBudgetTracker:
    def test_no_limit(self):
        tracker = BudgetTracker()
        tracker.check(model="gpt-4o")  # Should not raise
        assert tracker.remaining is None

    def test_record_cost(self, tmp_path: Path):
        tracker = BudgetTracker(
            monthly_limit=50.0,
            persist_path=tmp_path / "budget.json",
        )
        tracker.record(model="gpt-4o", cost=0.05, prompt_tokens=100, completion_tokens=50)
        assert tracker.total_cost == pytest.approx(0.05)
        assert tracker.remaining == pytest.approx(49.95)
        assert len(tracker.entries) == 1

    def test_multiple_records(self, tmp_path: Path):
        tracker = BudgetTracker(
            monthly_limit=10.0,
            persist_path=tmp_path / "budget.json",
        )
        tracker.record(model="gpt-4o", cost=1.0)
        tracker.record(model="gpt-4o-mini", cost=0.5)
        tracker.record(model="gpt-4o", cost=2.0)

        assert tracker.total_cost == pytest.approx(3.5)
        assert tracker.remaining == pytest.approx(6.5)
        assert len(tracker.entries) == 3

    def test_budget_exceeded(self, tmp_path: Path):
        tracker = BudgetTracker(
            monthly_limit=1.0,
            persist_path=tmp_path / "budget.json",
        )
        tracker.record(model="gpt-4o", cost=1.0)
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check(model="gpt-4o")
        assert exc_info.value.spent == pytest.approx(1.0)
        assert exc_info.value.limit == 1.0
        assert "gpt-4o" in str(exc_info.value)

    def test_persist_and_reload(self, tmp_path: Path):
        persist_path = tmp_path / "budget.json"

        # First tracker records some costs
        t1 = BudgetTracker(monthly_limit=50.0, persist_path=persist_path)
        t1.record(model="gpt-4o", cost=5.0)
        t1.record(model="claude", cost=3.0)

        # Second tracker loads from file
        t2 = BudgetTracker(monthly_limit=50.0, persist_path=persist_path)
        assert t2.total_cost == pytest.approx(8.0)
        assert len(t2.entries) == 2

    def test_summary(self, tmp_path: Path):
        tracker = BudgetTracker(
            monthly_limit=100.0,
            persist_path=tmp_path / "budget.json",
        )
        tracker.record(model="gpt-4o", cost=2.0)
        tracker.record(model="gpt-4o-mini", cost=0.5)
        tracker.record(model="gpt-4o", cost=1.5)

        summary = tracker.summary()
        assert summary["total_cost"] == pytest.approx(4.0)
        assert summary["monthly_limit"] == 100.0
        assert summary["remaining"] == pytest.approx(96.0)
        assert summary["requests"] == 3
        assert summary["by_model"]["gpt-4o"] == pytest.approx(3.5)
        assert summary["by_model"]["gpt-4o-mini"] == pytest.approx(0.5)

    def test_reset(self, tmp_path: Path):
        tracker = BudgetTracker(
            monthly_limit=10.0,
            persist_path=tmp_path / "budget.json",
        )
        tracker.record(model="gpt-4o", cost=5.0)
        assert tracker.total_cost == pytest.approx(5.0)

        tracker.reset()
        assert tracker.total_cost == pytest.approx(0.0)
        assert len(tracker.entries) == 0

    def test_record_from_response(self, tmp_path: Path):
        tracker = BudgetTracker(
            monthly_limit=50.0,
            persist_path=tmp_path / "budget.json",
        )
        # Mock a LiteLLM response
        mock_resp = MagicMock()
        mock_resp.model = "gpt-4o"
        mock_resp.usage.prompt_tokens = 100
        mock_resp.usage.completion_tokens = 50

        # record_from_response will try litellm.completion_cost which may fail in test
        # so cost falls back to 0.0
        tracker.record_from_response(mock_resp, model="gpt-4o")
        assert len(tracker.entries) == 1
        assert tracker.entries[0].model == "gpt-4o"
        assert tracker.entries[0].prompt_tokens == 100

    def test_no_persist_path_error(self, tmp_path: Path):
        """Tracker should handle persist errors gracefully."""
        tracker = BudgetTracker(
            monthly_limit=10.0,
            persist_path=tmp_path / "nonexistent" / "deep" / "budget.json",
        )
        # Should not raise even if path creation works
        tracker.record(model="test", cost=1.0)
        assert tracker.total_cost == pytest.approx(1.0)


class TestBudgetExceededError:
    def test_message(self):
        err = BudgetExceededError(spent=45.50, limit=50.0, model="gpt-4o")
        assert "$45.50" in str(err)
        assert "$50.00" in str(err)
        assert "gpt-4o" in str(err)

    def test_no_model(self):
        err = BudgetExceededError(spent=10.0, limit=10.0)
        assert "$10.00" in str(err)


class TestSingleton:
    def setup_method(self):
        reset_budget_tracker()

    def teardown_method(self):
        reset_budget_tracker()

    def test_get_creates_singleton(self):
        t1 = get_budget_tracker(monthly_limit=50.0)
        t2 = get_budget_tracker()
        assert t1 is t2

    def test_update_limit(self):
        t1 = get_budget_tracker(monthly_limit=50.0)
        assert t1.monthly_limit == 50.0
        t2 = get_budget_tracker(monthly_limit=100.0)
        assert t2.monthly_limit == 100.0
        assert t1 is t2
