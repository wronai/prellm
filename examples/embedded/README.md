# preLLM Example — Embedded Systems Refactoring

Refactor embedded/IoT code with hardware-aware preprocessing.

## Quick Start

```python
from prellm import preprocess_and_execute

result = await preprocess_and_execute(
    query="Zrefaktoruj moj ESP32 monitoring system - za duzo hardcode'ow, brak OTA",
    config_path="configs/domains/embedded.yaml",
    strategy="structure",
    user_context={"mcu": "ESP32-S3", "flash": "8MB", "ram": "512KB"},
)
print(result.content)
```

## Run Example

```bash
# With real LLMs
python examples/embedded/main.py

# Run tests (no LLM needed)
pytest examples/embedded/test_embedded.py -v
```

## Domain Config

Uses `configs/domains/embedded.yaml` with rules for:
- **Refactoring** — ESP32, RPi, STM32, Arduino
- **Firmware build** — PlatformIO, ESP-IDF, CMake
- **Sensors** — ADC, I2C, SPI integration
- **Power optimization** — deep sleep, battery life
- **FreeRTOS** — tasks, queues, semaphores
