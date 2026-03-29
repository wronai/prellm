"""Environment configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnvConfig:
    """Resolved environment configuration."""
    # Auth
    master_key: str | None = None

    # Models
    small_model: str = "ollama/qwen2.5:3b"
    large_model: str = "gpt-4o-mini"
    strategy: str = "auto"
    fallbacks: list[str] = field(default_factory=list)

    # Server
    host: str = "0.0.0.0"
    port: int = 8080

    # Config
    config_path: str | None = None
    log_level: str = "info"

    # Budget
    monthly_budget: float | None = None
    max_tokens: int = 4096
    timeout: int = 30

    # Providers (resolved)
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)
