"""Pipeline — 6-stage cognitive orchestrator.

Manages the flow: Classify → Distill → Plan → Execute → Evaluate → Reflect
with System 1 bypass and retry/backtrack loops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

logger = logging.getLogger("metascaffold.pipeline")


@dataclass
class PipelineState:
    """State that flows through the 6-stage pipeline."""
    task: str
    context: str
    classification: object | None = None
    template: object | None = None
    plan: object | None = None
    execution: object | None = None
    evaluation: object | None = None
    reflection: object | None = None
    attempt: int = 1
    max_attempts: int = 3
    compute_level: float | None = None

    @property
    def should_bypass(self) -> bool:
        """True if both distill and plan should be bypassed (legacy compat)."""
        return self.should_bypass_distill and self.should_bypass_plan

    @property
    def should_bypass_distill(self) -> bool:
        """True if compute_level is 1 (System 1 — skip distill+plan)."""
        if self.compute_level is not None:
            return self.compute_level <= 1
        # Legacy fallback: no compute_level → use routing
        if self.classification is None:
            return False
        return getattr(self.classification, "routing", "") == "system1"

    @property
    def should_bypass_plan(self) -> bool:
        """True if compute_level < 2 (System 1 or 1.5 — skip plan)."""
        if self.compute_level is not None:
            return self.compute_level < 2
        # Legacy fallback
        if self.classification is None:
            return False
        return getattr(self.classification, "routing", "") == "system1"

    @property
    def should_escalate(self) -> bool:
        """True if attempts exhausted."""
        return self.attempt >= self.max_attempts

    def with_retry(self) -> PipelineState:
        """Return new state with incremented attempt, cleared execution/evaluation."""
        return replace(
            self,
            attempt=self.attempt + 1,
            execution=None,
            evaluation=None,
        )

    def to_dict(self) -> dict:
        def _safe_dict(obj: object | None) -> dict | None:
            if obj is None:
                return None
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if hasattr(obj, "__dict__"):
                return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
            return None

        return {
            "task": self.task,
            "context": self.context[:200],
            "attempt": self.attempt,
            "compute_level": self.compute_level,
            "classification": _safe_dict(self.classification),
            "template": _safe_dict(self.template),
            "plan": _safe_dict(self.plan),
            "evaluation": _safe_dict(self.evaluation),
            "reflection": _safe_dict(self.reflection),
        }


class CognitivePipeline:
    """Orchestrates the 6-stage cognitive pipeline."""

    def __init__(
        self,
        classifier: object | None = None,
        distiller: object | None = None,
        planner: object | None = None,
        evaluator: object | None = None,
        reflector: object | None = None,
    ):
        self.classifier = classifier
        self.distiller = distiller
        self.planner = planner
        self.evaluator = evaluator
        self.reflector = reflector

    async def classify_stage(self, state: PipelineState) -> PipelineState:
        """Stage 1: Classify the task as System 1, 1.5, or 2."""
        if self.classifier is None:
            return state
        result = await self.classifier.classify_async(
            tool_name="pipeline",
            tool_input={},
            context=state.task + " " + state.context,
        )
        compute_level = None
        if hasattr(result, "signals") and isinstance(result.signals, dict):
            compute_level = result.signals.get("compute_level")
        return replace(state, classification=result, compute_level=compute_level)

    async def distill_stage(self, state: PipelineState) -> PipelineState:
        """Stage 2: Distill the task into a structured template."""
        if state.should_bypass_distill or self.distiller is None:
            return state
        template = await self.distiller.distill(state.task, state.context)
        return replace(state, template=template)

    async def plan_stage(self, state: PipelineState) -> PipelineState:
        """Stage 3: Generate execution strategies."""
        if state.should_bypass_plan or self.planner is None:
            return state
        task_text = state.task
        if state.template and hasattr(state.template, "objective"):
            task_text = state.template.objective
        plan = await self.planner.create_plan_async(
            task=task_text,
            context=state.context,
        )
        return replace(state, plan=plan)

    async def evaluate_stage(self, state: PipelineState, sandbox_result: object) -> PipelineState:
        """Stage 5: Evaluate execution results."""
        if self.evaluator is None:
            return replace(state, execution=sandbox_result)
        evaluation = await self.evaluator.evaluate_async(
            sandbox_result=sandbox_result,
            attempt=state.attempt,
        )
        return replace(state, execution=sandbox_result, evaluation=evaluation)

    async def reflect_stage(self, state: PipelineState, events: list[dict]) -> PipelineState:
        """Stage 6: Extract learning from telemetry."""
        if self.reflector is None:
            return state
        reflection = await self.reflector.reflect(events)
        return replace(state, reflection=reflection)
