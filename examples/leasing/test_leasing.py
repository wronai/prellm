"""Tests for Polish leasing example — fully mocked, no LLM needed."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from prellm import preprocess_and_execute
from prellm.models import PreLLMResponse


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


CONFIG_PATH = str(Path(__file__).parent.parent.parent / "configs" / "domains" / "polish_finance.yaml")


class TestLeasingCalculation:
    """Test leasing calculation scenario."""

    @pytest.mark.asyncio
    async def test_leasing_returns_response(self):
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_response(
                "Kalkulacja leasingu operacyjnego:\n"
                "Kwota netto: 250 000 PLN\n"
                "Okres: 48 miesiecy\n"
                "Rata netto: ~6 200 PLN/mies.\n"
                "VAT: 23%\n"
                "Rata brutto: ~7 626 PLN/mies."
            )
        )):
            result = await preprocess_and_execute(
                query="Oblicz rate leasingu operacyjnego camper van za 250000 PLN netto, 48 miesiecy",
                config_path=CONFIG_PATH,
                strategy="structure",
            )

        assert isinstance(result, PreLLMResponse)
        assert "250" in result.content or "leasing" in result.content.lower()

    @pytest.mark.asyncio
    async def test_leasing_with_wibor(self):
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_response(
                "Leasing z WIBOR 3M (5.75%) + marza 2.5% = 8.25% rocznie"
            )
        )):
            result = await preprocess_and_execute(
                query="Oblicz rate leasingu z WIBOR 3M + 2.5%, kwota 200000 PLN, 36 miesiecy",
                config_path=CONFIG_PATH,
                strategy="structure",
            )

        assert isinstance(result, PreLLMResponse)
        assert result.content  # non-empty


class TestFakturaVAT:
    """Test invoice generation scenario."""

    @pytest.mark.asyncio
    async def test_faktura_generation(self):
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_response(
                "FAKTURA VAT nr FV/2026/02/001\n"
                "Sprzedawca: NIP 5213000001\n"
                "Nabywca: NIP 1234567890\n"
                "Netto: 10 000 PLN | VAT 23%: 2 300 PLN | Brutto: 12 300 PLN"
            )
        )):
            result = await preprocess_and_execute(
                query="Wygeneruj fakture VAT dla NIP 5213000001, nabywca NIP 1234567890",
                config_path=CONFIG_PATH,
            )

        assert isinstance(result, PreLLMResponse)
        assert "VAT" in result.content or "NIP" in result.content


class TestPITCalculation:
    """Test tax calculation scenario."""

    @pytest.mark.asyncio
    async def test_pit_calculation(self):
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_response(
                "PIT liniowy 19%: 34 200 PLN\n"
                "Skala podatkowa: 32% powyzej 120 000 PLN = ~39 000 PLN\n"
                "Rekomendacja: liniowy 19% korzystniejszy o ~4 800 PLN"
            )
        )):
            result = await preprocess_and_execute(
                query="Oblicz podatek PIT dla przychodu 180000 PLN rocznie",
                config_path=CONFIG_PATH,
                strategy="enrich",
                user_context="samozatrudnienie_IT_B2B",
            )

        assert isinstance(result, PreLLMResponse)
        assert "PIT" in result.content or "podatkow" in result.content.lower()


class TestPolishFinanceConfigLoading:
    """Test that the Polish finance config loads correctly."""

    def test_config_file_exists(self):
        assert Path(CONFIG_PATH).is_file()

    def test_config_is_valid_yaml(self):
        import yaml
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        assert "domain_rules" in data
        assert len(data["domain_rules"]) >= 3

    def test_config_has_finance_rules(self):
        import yaml
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        rule_names = [r["name"] for r in data["domain_rules"]]
        assert "leasing_pl" in rule_names
        assert "faktura_pl" in rule_names

    def test_config_has_polish_prompts(self):
        import yaml
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        assert "prompts" in data
        # Polish prompts should contain Polish text
        structure_prompt = data["prompts"].get("structure", "")
        assert "PLN" in structure_prompt or "VAT" in structure_prompt
