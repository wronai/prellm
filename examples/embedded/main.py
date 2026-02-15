#!/usr/bin/env python3
"""preLLM Example — Embedded Systems Refactoring

Demonstrates hardware-aware preprocessing for IoT/embedded.
Requires: Ollama running locally or cloud API keys.

Usage:
    python examples/embedded/main.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


async def main():
    from prellm import preprocess_and_execute

    print("=" * 60)
    print("  preLLM — Embedded Refactoring Example")
    print("=" * 60)

    # Example 1: ESP32 refactoring with constraints
    print("\n--- 1. ESP32 Monitoring Refactor ---")
    result = await preprocess_and_execute(
        query="Zrefaktoruj moj ESP32 monitoring system - za duzo hardcode'ow, brak OTA, zuzywa 200mA w idle",
        config_path="configs/domains/embedded.yaml",
        strategy="structure",
        user_context={
            "mcu": "ESP32-S3",
            "flash": "8MB",
            "ram": "512KB",
            "framework": "ESP-IDF 5.1",
            "sensors": "BME280, MPU6050, GPS NEO-6M",
        },
    )
    print(f"Model:   {result.model_used}")
    print(f"Content: {result.content[:200]}...")
    if result.decomposition and result.decomposition.matched_rule:
        print(f"Rule:    {result.decomposition.matched_rule}")

    # Example 2: Power optimization
    print("\n--- 2. Power Optimization ---")
    result = await preprocess_and_execute(
        query="Zoptymalizuj zuzycie pradu - bateria 3000mAh musi starczyc na 30 dni, pomiar co 5 minut",
        config_path="configs/domains/embedded.yaml",
        strategy="enrich",
    )
    print(f"Content: {result.content[:200]}...")

    # Example 3: FreeRTOS task design
    print("\n--- 3. FreeRTOS Tasks ---")
    result = await preprocess_and_execute(
        query="Zaprojektuj architekture taskow FreeRTOS: sensor reading, WiFi upload, display update, watchdog",
        config_path="configs/domains/embedded.yaml",
        user_context={"mcu": "STM32F407", "rtos": "FreeRTOS", "stack": "4KB per task"},
    )
    print(f"Content: {result.content[:200]}...")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
