"""Tests for embedded systems example — fully mocked, no LLM needed."""

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


CONFIG_PATH = str(Path(__file__).parent.parent.parent / "configs" / "domains" / "embedded.yaml")


class TestESP32Refactoring:
    """Test ESP32 refactoring scenario."""

    @pytest.mark.asyncio
    async def test_esp32_refactor_returns_response(self):
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_response(
                "ESP32 Refactoring Plan:\n"
                "1. Extract config to NVS (non-volatile storage)\n"
                "2. Add OTA via esp_https_ota\n"
                "3. Implement deep sleep between measurements\n"
                "4. Target: <10mA idle current"
            )
        )):
            result = await preprocess_and_execute(
                query="Zrefaktoruj moj ESP32 monitoring system - za duzo hardcode'ow, brak OTA",
                config_path=CONFIG_PATH,
                strategy="structure",
                user_context={
                    "mcu": "ESP32-S3",
                    "flash": "8MB",
                    "ram": "512KB",
                },
            )

        assert isinstance(result, PreLLMResponse)
        assert "ESP32" in result.content or "OTA" in result.content or "Refactor" in result.content

    @pytest.mark.asyncio
    async def test_esp32_uses_correct_config_models(self):
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_response("Refactored code")
        )):
            result = await preprocess_and_execute(
                query="Refactor ESP32 firmware",
                config_path=CONFIG_PATH,
                strategy="structure",
            )

        assert result.small_model_used == "ollama/qwen2.5:3b"


class TestPowerOptimization:
    """Test power optimization scenario."""

    @pytest.mark.asyncio
    async def test_power_optimization(self):
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_response(
                "Power budget analysis:\n"
                "- Active: 160mA x 2s = 0.089mAh per measurement\n"
                "- Deep sleep: 10uA x 298s = 0.828mAh per cycle\n"
                "- Daily: 288 cycles x 0.917mAh = 264mAh\n"
                "- Battery life: 3000mAh / 264mAh = 11.4 days\n"
                "Recommendation: Reduce WiFi time, use light sleep"
            )
        )):
            result = await preprocess_and_execute(
                query="Zoptymalizuj zuzycie pradu - bateria 3000mAh musi starczyc na 30 dni",
                config_path=CONFIG_PATH,
                strategy="enrich",
            )

        assert isinstance(result, PreLLMResponse)
        assert result.content  # non-empty


class TestFreeRTOSTasks:
    """Test FreeRTOS task design scenario."""

    @pytest.mark.asyncio
    async def test_freertos_task_design(self):
        with patch("litellm.acompletion", new=AsyncMock(
            return_value=_mock_response(
                "FreeRTOS Architecture:\n"
                "Task 1: SensorTask (priority 3, 4KB stack, 100ms period)\n"
                "Task 2: WiFiTask (priority 2, 8KB stack, event-driven)\n"
                "Task 3: DisplayTask (priority 1, 4KB stack, 500ms period)\n"
                "Task 4: WatchdogTask (priority 4, 2KB stack, 1s period)"
            )
        )):
            result = await preprocess_and_execute(
                query="Zaprojektuj architekture taskow FreeRTOS: sensor, WiFi, display, watchdog",
                config_path=CONFIG_PATH,
                user_context={"mcu": "STM32F407", "rtos": "FreeRTOS"},
            )

        assert isinstance(result, PreLLMResponse)
        assert "Task" in result.content or "FreeRTOS" in result.content


class TestEmbeddedConfigLoading:
    """Test that the embedded config loads correctly."""

    def test_config_file_exists(self):
        assert Path(CONFIG_PATH).is_file()

    def test_config_is_valid_yaml(self):
        import yaml
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        assert "domain_rules" in data
        assert "small_model" in data
        assert len(data["domain_rules"]) >= 3

    def test_config_has_embedded_rules(self):
        import yaml
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        rule_names = [r["name"] for r in data["domain_rules"]]
        assert "embedded_refactor" in rule_names
        assert "power_optimization" in rule_names

    def test_config_has_hardware_keywords(self):
        import yaml
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f)
        all_keywords = []
        for rule in data["domain_rules"]:
            all_keywords.extend(rule.get("keywords", []))
        assert "esp32" in all_keywords
        assert "freertos" in all_keywords
