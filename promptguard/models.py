"""Data models for PromptGuard — all inputs/outputs are Pydantic v2 validated."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# --- Enums ---

class Policy(str, enum.Enum):
    STRICT = "strict"
    LENIENT = "lenient"
    DEVOPS = "devops"


class ApprovalMode(str, enum.Enum):
    AUTO = "auto"
    MANUAL = "manual"


class StepStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


# --- Config models ---

class BiasPattern(BaseModel):
    regex: str
    action: str = "clarify"
    severity: str = "medium"
    description: str = ""


class ModelConfig(BaseModel):
    fallback: list[str] = Field(default_factory=lambda: ["gpt-4o-mini"])
    timeout: int = 30
    max_tokens: int = 2048


class GuardConfig(BaseModel):
    """Top-level YAML config model."""
    bias_patterns: list[BiasPattern] = Field(default_factory=list)
    clarify_template: str = "[KONTEKST]: Podaj szczegóły lub alternatywy dla: {query}"
    max_retries: int = 3
    policy: Policy = Policy.STRICT
    models: ModelConfig = Field(default_factory=ModelConfig)
    context_sources: list[dict[str, Any]] = Field(default_factory=list)


# --- Process chain models ---

class ProcessStep(BaseModel):
    name: str
    prompt: str
    policy: Policy = Policy.STRICT
    approval: ApprovalMode = ApprovalMode.AUTO
    rollback: bool = False
    timeout: int = 300
    depends_on: list[str] = Field(default_factory=list)


class ProcessConfig(BaseModel):
    process: str
    description: str = ""
    context_sources: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[ProcessStep]


# --- Runtime models ---

class AnalysisResult(BaseModel):
    """Result of query analysis — what was detected and what was done."""
    needs_clarify: bool = False
    detected_patterns: list[str] = Field(default_factory=list)
    enriched_query: str = ""
    original_query: str = ""
    readability_score: float | None = None
    ambiguity_flags: list[str] = Field(default_factory=list)


class GuardResponse(BaseModel):
    """Response from PromptGuard — the final output after analysis and LLM call."""
    content: str
    clarified: bool = False
    needs_more_context: bool = False
    model_used: str = ""
    analysis: AnalysisResult | None = None
    retries: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StepResult(BaseModel):
    """Result of a single process chain step."""
    step_name: str
    status: StepStatus = StepStatus.PENDING
    response: GuardResponse | None = None
    approved_by: str | None = None
    approved_at: datetime | None = None
    error: str | None = None
    duration_seconds: float = 0.0


class ProcessResult(BaseModel):
    """Result of a full process chain execution."""
    process_name: str
    steps: list[StepResult] = Field(default_factory=list)
    completed: bool = False
    total_duration_seconds: float = 0.0
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None


# --- Audit models ---

class AuditEntry(BaseModel):
    """Single audit log entry for traceability."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    action: str
    query: str = ""
    response_summary: str = ""
    model: str = ""
    policy: Policy = Policy.STRICT
    step_name: str | None = None
    process_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
