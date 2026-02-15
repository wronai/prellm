"""Edge case tests — Polish input, mixed languages, boundary conditions."""

from __future__ import annotations

import pytest

from promptguard.analyzers.bias_detector import BiasDetector, DEFAULT_PATTERNS
from promptguard.models import BiasPattern


class TestPolishPatterns:
    def test_polish_deploy_production(self):
        d = BiasDetector()
        r = d.analyze("Zdeployuj na produkcję natychmiast")
        # Should trigger: short + DevOps verb without clear target
        assert r.needs_clarify is True

    def test_polish_delete_database(self):
        d = BiasDetector()
        r = d.analyze("Usuń bazę danych klientów z serwera głównego")
        assert r.needs_clarify is True
        assert any("Destructive" in p for p in r.detected_patterns)

    def test_polish_always_bias(self):
        d = BiasDetector()
        r = d.analyze("Zawsze robimy to w ten sposób")
        assert r.needs_clarify is True

    def test_polish_only_bias(self):
        d = BiasDetector()
        r = d.analyze("Tylko ten sposób działa poprawnie")
        assert r.needs_clarify is True

    def test_polish_safe_long_query(self):
        d = BiasDetector()
        r = d.analyze("Proszę wyświetl logi z klastra staging za ostatnie 24 godziny dla namespace backend")
        assert r.needs_clarify is False


class TestMixedLanguage:
    def test_polish_english_mix(self):
        d = BiasDetector()
        r = d.analyze("Deploy to production na klastrze głównym")
        assert r.needs_clarify is True

    def test_english_with_polish_verbs(self):
        d = BiasDetector()
        # Has "staging" target so no missing-target flag, but long enough
        r = d.analyze("Zdeployuj to na staging cluster right now")
        assert r.needs_clarify is False  # has target + enough words

    def test_polish_verb_no_target(self):
        d = BiasDetector()
        r = d.analyze("Zdeployuj nową wersję aplikacji teraz")
        assert r.needs_clarify is True


class TestBoundaryConditions:
    def test_empty_query(self):
        d = BiasDetector()
        r = d.analyze("")
        assert r.needs_clarify is True  # empty = too short

    def test_single_word(self):
        d = BiasDetector()
        r = d.analyze("deploy")
        assert r.needs_clarify is True

    def test_very_long_safe_query(self):
        d = BiasDetector()
        long_query = "Proszę wyświetl szczegółowy raport " * 20 + "z klastra staging"
        r = d.analyze(long_query)
        assert r.needs_clarify is False

    def test_case_insensitive_patterns(self):
        d = BiasDetector()
        r = d.analyze("DEPLOY TO PRODUCTION")
        assert r.needs_clarify is True

    def test_no_false_positive_on_partial_match(self):
        d = BiasDetector()
        # "deployment" contains "deploy" but "deploy to prod" pattern needs specific structure
        r = d.analyze("Sprawdź status ostatniego deployment na klastrze staging z health-checkami")
        assert r.needs_clarify is False

    def test_multiple_patterns_detected(self):
        d = BiasDetector()
        r = d.analyze("Zawsze deploy to production immediately")
        assert r.needs_clarify is True
        assert len(r.detected_patterns) >= 2

    def test_custom_empty_patterns(self):
        d = BiasDetector(patterns=[])
        r = d.analyze("This is a detailed enough query about deploying things to staging cluster right now")
        assert r.needs_clarify is False

    def test_default_patterns_count(self):
        assert len(DEFAULT_PATTERNS) >= 9

    def test_severity_levels(self):
        d = BiasDetector()
        r = d.analyze("Delete database tables from production server now")
        critical = [p for p in r.detected_patterns if "critical" in p]
        assert len(critical) >= 1
