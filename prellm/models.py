"""Data models for preLLM v0.2 — all inputs/outputs are Pydantic v2 validated.

v0.2 replaces hardcoded regex pipeline with small LLM decomposition.
Old v0.1 models (BiasPattern, GuardConfig, GuardResponse) kept for backward compat.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ============================================================
# Sensitivity levels (Task 2)
# ============================================================

class SensitivityLevel(str, enum.Enum):
    SAFE = "safe"
    MASKED = "masked"
    BLOCKED = "blocked"


# ============================================================
# Shell Context models (Task 1)
# ============================================================

class ProcessInfo(BaseModel):
    pid: int = 0
    cwd: str = ""
    user: str = ""
    parent_pid: int | None = None
    tty: str = ""


class LocaleInfo(BaseModel):
    lang: str = ""
    lc_all: str = ""
    timezone: str = ""
    encoding: str = ""


class ShellInfo(BaseModel):
    shell: str = ""
    term: str = ""
    columns: int = 0
    lines: int = 0


class NetworkContext(BaseModel):
    hostname: str = ""
    local_ip: str = ""
    dns_suffix: str = ""


class ShellContext(BaseModel):
    env_vars: dict[str, str] = Field(default_factory=dict)
    process: ProcessInfo = Field(default_factory=ProcessInfo)
    locale: LocaleInfo = Field(default_factory=LocaleInfo)
    shell: ShellInfo = Field(default_factory=ShellInfo)
    network: NetworkContext = Field(default_factory=NetworkContext)
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    collection_duration_ms: float = 0.0


# ============================================================
# Context Schema model (Task 4)
# ============================================================

class ContextSchema(BaseModel):
    execution_env: str = "cli"
    platform: str = ""
    project_type: str | None = None
    project_summary: str | None = None
    available_tools: list[str] = Field(default_factory=list)
    locale: str = ""
    timezone: str = ""
    user_history_summary: str | None = None
    sensitive_fields_blocked: int = 0
    schema_token_cost: int = 0


# ============================================================
# Filter report model (Task 2)
# ============================================================

class FilterReport(BaseModel):
    blocked_keys: list[str] = Field(default_factory=list)
    masked_keys: list[str] = Field(default_factory=list)
    safe_keys: list[str] = Field(default_factory=list)
    total_processed: int = 0


# ============================================================
# Compressed folder model (Task 3)
# ============================================================

class CompressedFolder(BaseModel):
    root: str = ""
    toon_output: str = ""
    dependency_graph: dict[str, list[str]] = Field(default_factory=dict)
    module_summaries: dict[str, str] = Field(default_factory=dict)
    total_modules: int = 0
    total_functions: int = 0
    estimated_tokens: int = 0


# ============================================================
# Enums
# ============================================================

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


class DecompositionStrategy(str, enum.Enum):
    """Strategy for how the small LLM preprocesses a query."""
    AUTO = "auto"
    CLASSIFY = "classify"
    STRUCTURE = "structure"
    SPLIT = "split"
    ENRICH = "enrich"
    PASSTHROUGH = "passthrough"


# ============================================================
# v0.1 compat models (kept for backward compatibility)
# ============================================================

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
    """Top-level YAML config model (v0.1 compat)."""
    bias_patterns: list[BiasPattern] = Field(default_factory=list)
    clarify_template: str = "[KONTEKST]: Podaj szczegóły lub alternatywy dla: {query}"
    max_retries: int = 3
    policy: Policy = Policy.STRICT
    models: ModelConfig = Field(default_factory=ModelConfig)
    context_sources: list[dict[str, Any]] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Result of query analysis (v0.1 compat)."""
    needs_clarify: bool = False
    detected_patterns: list[str] = Field(default_factory=list)
    enriched_query: str = ""
    original_query: str = ""
    readability_score: float | None = None
    ambiguity_flags: list[str] = Field(default_factory=list)


class GuardResponse(BaseModel):
    """Response from Prellm (v0.1 compat)."""
    content: str
    clarified: bool = False
    needs_more_context: bool = False
    model_used: str = ""
    analysis: AnalysisResult | None = None
    retries: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# v0.2 — Domain Rules (YAML-driven, replaces hardcoded regex)
# ============================================================

class DomainRule(BaseModel):
    """Configurable domain rule — keywords, intent, required fields, enrich template.

    Replaces hardcoded BiasPattern regex with dynamic LLM-based matching.
    """
    name: str
    keywords: list[str] = Field(default_factory=list)
    intent: str = ""
    required_fields: list[str] = Field(default_factory=list)
    enrich_template: str = ""
    severity: str = "medium"
    strategy: DecompositionStrategy = DecompositionStrategy.CLASSIFY


# ============================================================
# v0.2 — LLM Provider Config
# ============================================================

class LLMProviderConfig(BaseModel):
    """Configuration for a single LLM provider (small or large)."""
    model: str = "gpt-4o-mini"
    fallback: list[str] = Field(default_factory=list)
    max_retries: int = 3
    timeout: int = 30
    max_tokens: int = 2048
    temperature: float = 0.0


# ============================================================
# v0.2 — Decomposition Prompts
# ============================================================

class DecompositionPrompts(BaseModel):
    """System prompts for each decomposition step — all configurable via YAML."""
    classify_prompt: str = (
        "You are a query classifier. Analyze the user query and return JSON: "
        '{"intent": "<intent>", "confidence": <0.0-1.0>, "domain": "<domain>"}. '
        "Be concise. Only return valid JSON."
    )
    structure_prompt: str = (
        "You are a field extractor. Given the user query, extract structured fields as JSON: "
        '{"action": "<action>", "target": "<target>", "parameters": {}}. '
        "Only return valid JSON."
    )
    enrich_prompt: str = (
        "You are a prompt enricher. The user query is missing context. "
        "Given the query and the list of missing fields, compose a complete, "
        "unambiguous prompt for a large LLM. Return only the enriched prompt text."
    )
    compose_prompt: str = (
        "You are a prompt composer. Given the classification, structured fields, "
        "and domain context, compose a final optimized prompt for a large LLM. "
        "Return only the composed prompt text."
    )
    split_prompt: str = (
        "You are a query splitter. Break the complex user query into independent "
        "sub-questions. Return JSON: {\"sub_queries\": [\"q1\", \"q2\", ...]}. "
        "Only return valid JSON."
    )


# ============================================================
# v0.2 — PreLLM Config (replaces GuardConfig for new pipeline)
# ============================================================

class PreLLMConfig(BaseModel):
    """Top-level config for preLLM v0.2 — fully YAML-driven."""
    small_model: LLMProviderConfig = Field(
        default_factory=lambda: LLMProviderConfig(model="phi3:mini", max_tokens=512, temperature=0.0)
    )
    large_model: LLMProviderConfig = Field(
        default_factory=lambda: LLMProviderConfig(model="gpt-4o-mini", max_tokens=2048)
    )
    domain_rules: list[DomainRule] = Field(default_factory=list)
    prompts: DecompositionPrompts = Field(default_factory=DecompositionPrompts)
    default_strategy: DecompositionStrategy = DecompositionStrategy.CLASSIFY
    context_sources: list[dict[str, Any]] = Field(default_factory=list)
    max_retries: int = 3
    policy: Policy = Policy.STRICT


# ============================================================
# v0.2 — Decomposition Result
# ============================================================

class ClassificationResult(BaseModel):
    """Output of the CLASSIFY step."""
    intent: str = ""
    confidence: float = 0.0
    domain: str = ""


class StructureResult(BaseModel):
    """Output of the STRUCTURE step."""
    action: str = ""
    target: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class DecompositionResult(BaseModel):
    """Full result of the small LLM decomposition pipeline."""
    strategy: DecompositionStrategy = DecompositionStrategy.PASSTHROUGH
    original_query: str = ""
    classification: ClassificationResult | None = None
    structure: StructureResult | None = None
    sub_queries: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    matched_rule: str | None = None
    composed_prompt: str = ""
    raw_small_llm_outputs: dict[str, str] = Field(default_factory=dict)


# ============================================================
# v0.2 — PreLLM Response
# ============================================================

class PreLLMResponse(BaseModel):
    """Response from preLLM v0.2 — includes decomposition + large LLM output."""
    content: str
    decomposition: DecompositionResult | None = None
    model_used: str = ""
    small_model_used: str = ""
    retries: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # Backward compat fields
    clarified: bool = False
    needs_more_context: bool = False


# ============================================================
# Process chain models (updated for v0.2 per-step strategy)
# ============================================================

class ProcessStep(BaseModel):
    name: str
    prompt: str
    policy: Policy = Policy.STRICT
    approval: ApprovalMode = ApprovalMode.AUTO
    rollback: bool = False
    timeout: int = 300
    depends_on: list[str] = Field(default_factory=list)
    strategy: DecompositionStrategy | None = None


class ProcessConfig(BaseModel):
    process: str
    description: str = ""
    context_sources: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[ProcessStep]


class StepResult(BaseModel):
    """Result of a single process chain step."""
    step_name: str
    status: StepStatus = StepStatus.PENDING
    response: PreLLMResponse | GuardResponse | None = None
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


# ============================================================
# Audit models
# ============================================================

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
