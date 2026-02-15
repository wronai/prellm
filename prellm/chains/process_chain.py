"""ProcessChain — Multi-step DevOps workflow engine with approval gates and audit trail.

Defines workflows as YAML, validates each step through preLLM, supports
manual/auto approval, rollback, per-step decomposition strategy, and full audit logging.

v0.3: Supports two-agent architecture via preprocess_and_execute when pipeline is specified.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Union

import yaml

from prellm.analyzers.context_engine import ContextEngine
from prellm.models import (
    ApprovalMode,
    AuditEntry,
    GuardConfig,
    ProcessConfig,
    ProcessResult,
    ProcessStep,
    StepResult,
    StepStatus,
)

logger = logging.getLogger("prellm.chains")

# Type for approval callback: receives step info, returns (approved: bool, approved_by: str)
ApprovalCallback = Callable[[str, str], Awaitable[tuple[bool, str]]]


class ProcessChain:
    """Execute multi-step DevOps workflows with preLLM validation at each step.

    Usage:
        from prellm.core import PreLLM
        engine = PreLLM("prellm_config.yaml")
        chain = ProcessChain("deploy.yaml", engine=engine)
        result = await chain.execute(env="production")
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: ProcessConfig | None = None,
        guard_config_path: str | Path | None = None,
        guard: Any | None = None,
        engine: Any | None = None,
    ):
        if config:
            self.process_config = config
        elif config_path:
            self.process_config = self._load_process_config(Path(config_path))
        else:
            raise ValueError("Either config_path or config must be provided")

        # v0.2+ engine takes priority over guard
        self._engine = engine
        self._guard = guard
        if not engine and not guard:
            from prellm.core import PreLLM
            self._engine = PreLLM(config_path=guard_config_path)

        self.context_engine = ContextEngine(self.process_config.context_sources)
        self.audit_log: list[AuditEntry] = []
        self._step_results: dict[str, StepResult] = {}

    async def execute(
        self,
        extra_context: dict[str, str] | None = None,
        approval_callback: ApprovalCallback | None = None,
        dry_run: bool = False,
        **env_overrides: str,
    ) -> ProcessResult:
        """Execute the full process chain.

        Args:
            extra_context: Additional key-value context to inject into prompts.
            approval_callback: Async function for manual approval steps.
                Receives (step_name, enriched_prompt) → (approved, approved_by).
            dry_run: If True, analyze prompts but don't call LLM.
            **env_overrides: Override environment-level context (e.g., env="production").

        Returns:
            ProcessResult with status of all steps.
        """
        start_time = time.time()
        ctx = self.context_engine.gather()
        if extra_context:
            ctx.update(extra_context)
        ctx.update(env_overrides)

        result = ProcessResult(
            process_name=self.process_config.process,
            started_at=datetime.utcnow(),
        )

        for step in self.process_config.steps:
            step_result = await self._execute_step(
                step=step,
                ctx=ctx,
                approval_callback=approval_callback,
                dry_run=dry_run,
            )
            result.steps.append(step_result)
            self._step_results[step.name] = step_result

            if step_result.status == StepStatus.FAILED:
                logger.error(f"Step '{step.name}' failed — halting chain.")
                break

            if step_result.status == StepStatus.AWAITING_APPROVAL:
                logger.info(f"Step '{step.name}' awaiting approval — chain paused.")
                break

        # Final status
        all_completed = all(s.status == StepStatus.COMPLETED for s in result.steps)
        result.completed = all_completed
        result.total_duration_seconds = time.time() - start_time
        result.finished_at = datetime.utcnow()

        return result

    async def _execute_step(
        self,
        step: ProcessStep,
        ctx: dict[str, str],
        approval_callback: ApprovalCallback | None,
        dry_run: bool,
    ) -> StepResult:
        """Execute a single step in the chain."""
        step_start = time.time()
        step_result = StepResult(step_name=step.name, status=StepStatus.RUNNING)

        # 1. Check dependencies
        dep_error = self._check_dependencies(step)
        if dep_error:
            step_result.status = StepStatus.FAILED
            step_result.error = dep_error
            return step_result

        # 2. Enrich prompt
        enriched_prompt = self.context_engine.enrich_prompt(step.prompt, ctx)
        logger.info(f"Step '{step.name}': enriched prompt = {enriched_prompt[:100]}...")

        # 3. Approval gate
        if step.approval == ApprovalMode.MANUAL:
            blocked = await self._handle_approval(step_result, step, enriched_prompt, approval_callback)
            if blocked:
                return step_result

        # 4. Dry run
        if dry_run:
            await self._run_dry_run(step_result, step, enriched_prompt, step_start)
            return step_result

        # 5. Execute
        await self._run_engine(step_result, step, enriched_prompt, ctx)

        step_result.duration_seconds = time.time() - step_start
        return step_result

    def _check_dependencies(self, step: ProcessStep) -> str | None:
        """Check that all step dependencies are completed. Returns error string or None."""
        for dep in step.depends_on:
            dep_result = self._step_results.get(dep)
            if not dep_result or dep_result.status != StepStatus.COMPLETED:
                return f"Dependency '{dep}' not completed"
        return None

    async def _handle_approval(
        self,
        step_result: StepResult,
        step: ProcessStep,
        enriched_prompt: str,
        approval_callback: ApprovalCallback | None,
    ) -> bool:
        """Handle manual approval gate. Returns True if step is blocked (awaiting/failed)."""
        if approval_callback:
            try:
                approved, approved_by = await approval_callback(step.name, enriched_prompt)
            except Exception as e:
                step_result.status = StepStatus.FAILED
                step_result.error = f"Approval callback error: {e}"
                return True

            if not approved:
                step_result.status = StepStatus.AWAITING_APPROVAL
                return True

            step_result.approved_by = approved_by
            step_result.approved_at = datetime.utcnow()
            step_result.status = StepStatus.APPROVED
            return False
        else:
            step_result.status = StepStatus.AWAITING_APPROVAL
            return True

    async def _run_dry_run(
        self, step_result: StepResult, step: ProcessStep, enriched_prompt: str, step_start: float,
    ) -> None:
        """Dry-run: analyze without calling LLM."""
        if self._engine:
            analysis = await self._engine.decompose_only(enriched_prompt, strategy=step.strategy)
        elif self._guard:
            analysis = self._guard.analyze_only(enriched_prompt)
        else:
            analysis = {}
        step_result.status = StepStatus.COMPLETED
        step_result.duration_seconds = time.time() - step_start
        logger.info(f"Step '{step.name}' (dry-run): {analysis}")

    async def _run_engine(
        self, step_result: StepResult, step: ProcessStep, enriched_prompt: str, ctx: dict[str, str],
    ) -> None:
        """Execute step through v0.3 pipeline, v0.2 engine, or v0.1 guard."""
        try:
            if hasattr(step, 'pipeline') and step.pipeline:
                from prellm.core import preprocess_and_execute
                response = await preprocess_and_execute(
                    query=enriched_prompt,
                    pipeline=step.pipeline,
                    user_context=ctx,
                )
            elif self._engine:
                response = await self._engine(enriched_prompt, strategy=step.strategy, extra_context=ctx)
            elif self._guard:
                response = await self._guard(enriched_prompt, extra_context=ctx)
            else:
                raise RuntimeError("No engine or guard configured")
            step_result.response = response
            step_result.status = StepStatus.COMPLETED
            self._audit_step(step.name, enriched_prompt, response.content)

        except Exception as e:
            step_result.status = StepStatus.FAILED
            step_result.error = str(e)
            logger.error(f"Step '{step.name}' failed: {e}")
            if step.rollback:
                logger.warning(f"Step '{step.name}' has rollback=true — rollback should be triggered")
                step_result.status = StepStatus.ROLLED_BACK

    def get_audit_log(self) -> list[dict[str, Any]]:
        return [e.model_dump() for e in self.audit_log]

    def _audit_step(self, step_name: str, prompt: str, response: str) -> None:
        entry = AuditEntry(
            action="process_step",
            query=prompt,
            response_summary=response[:200] if response else "",
            step_name=step_name,
            process_name=self.process_config.process,
        )
        self.audit_log.append(entry)

    @staticmethod
    def _load_process_config(path: Path) -> ProcessConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        steps = []
        for s in raw.get("steps", []):
            steps.append(ProcessStep(**s))

        return ProcessConfig(
            process=raw.get("process", "unnamed"),
            description=raw.get("description", ""),
            context_sources=raw.get("context_sources", []),
            steps=steps,
        )
