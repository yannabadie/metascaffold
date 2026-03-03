"""Planner — decomposes System 2 tasks into strategies with steps, risks, and rollback plans.

The Planner produces structured plans that the Evaluator can later assess.
It uses heuristic decomposition (not LLM calls) to suggest strategies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Strategy:
    """A single execution strategy with steps, risks, and rollback."""
    id: str
    description: str
    steps: list[str]
    confidence: float
    risks: list[str]
    rollback_plan: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "steps": self.steps,
            "confidence": self.confidence,
            "risks": self.risks,
            "rollback_plan": self.rollback_plan,
        }


@dataclass
class Plan:
    """A structured execution plan with multiple strategies."""
    task: str
    strategies: list[Strategy]
    recommended: str
    notebooklm_insights: str = ""

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "strategies": [s.to_dict() for s in self.strategies],
            "recommended": self.recommended,
            "notebooklm_insights": self.notebooklm_insights,
        }


_REFACTOR_PATTERN = re.compile(r"refactor|restructure|reorganize|rewrite", re.IGNORECASE)
_BUG_PATTERN = re.compile(r"fix|bug|error|issue|broken|crash", re.IGNORECASE)
_FEATURE_PATTERN = re.compile(r"add|create|implement|build|new", re.IGNORECASE)


class Planner:
    """Heuristic planner that decomposes tasks into strategies."""

    def create_plan(
        self,
        task: str,
        context: str,
        notebooklm_insights: str = "",
    ) -> Plan:
        strategies: list[Strategy] = []

        if _REFACTOR_PATTERN.search(task):
            strategies = self._plan_refactor(task, context)
        elif _BUG_PATTERN.search(task):
            strategies = self._plan_bugfix(task, context)
        elif _FEATURE_PATTERN.search(task):
            strategies = self._plan_feature(task, context)
        else:
            strategies = self._plan_generic(task, context)

        recommended = strategies[0].id if strategies else "A"

        return Plan(
            task=task,
            strategies=strategies,
            recommended=recommended,
            notebooklm_insights=notebooklm_insights,
        )

    def _plan_refactor(self, task: str, context: str) -> list[Strategy]:
        return [
            Strategy(
                id="A",
                description="Incremental refactor with adapter pattern",
                steps=[
                    "Identify all call sites for the target code",
                    "Create adapter interface that wraps the current implementation",
                    "Implement new version behind the adapter",
                    "Migrate call sites one at a time to the new interface",
                    "Remove adapter and old implementation",
                    "Run full test suite",
                ],
                confidence=0.8,
                risks=["Adapter adds temporary complexity", "Partial migration state"],
                rollback_plan="Revert adapter changes, restore original implementation",
            ),
            Strategy(
                id="B",
                description="Direct replacement in a feature branch",
                steps=[
                    "Create feature branch",
                    "Rewrite the target module from scratch",
                    "Update all call sites",
                    "Run full test suite",
                    "Merge or discard branch",
                ],
                confidence=0.6,
                risks=["All-or-nothing: partial completion is not usable", "High blast radius"],
                rollback_plan="Delete feature branch entirely",
            ),
        ]

    def _plan_bugfix(self, task: str, context: str) -> list[Strategy]:
        return [
            Strategy(
                id="A",
                description="Reproduce, isolate, fix with regression test",
                steps=[
                    "Write a failing test that reproduces the bug",
                    "Run test to confirm it fails",
                    "Identify root cause via debugger or logging",
                    "Apply minimal fix",
                    "Run test to confirm it passes",
                    "Run full test suite for regressions",
                ],
                confidence=0.85,
                risks=["Root cause may be deeper than symptoms"],
                rollback_plan="Revert the fix commit",
            ),
            Strategy(
                id="B",
                description="Defensive fix with extra validation",
                steps=[
                    "Add input validation at the entry point",
                    "Add error handling around the failing code path",
                    "Write tests for both valid and invalid inputs",
                    "Run full test suite",
                ],
                confidence=0.7,
                risks=["May mask the real bug instead of fixing it"],
                rollback_plan="Revert validation changes",
            ),
        ]

    def _plan_feature(self, task: str, context: str) -> list[Strategy]:
        return [
            Strategy(
                id="A",
                description="TDD: test-first implementation",
                steps=[
                    "Define the public API / interface for the feature",
                    "Write failing tests for the happy path",
                    "Implement minimal code to pass tests",
                    "Write edge case tests",
                    "Implement edge case handling",
                    "Integration test",
                ],
                confidence=0.8,
                risks=["API design may need revision after implementation"],
                rollback_plan="Revert feature branch",
            ),
        ]

    def _plan_generic(self, task: str, context: str) -> list[Strategy]:
        return [
            Strategy(
                id="A",
                description="Standard step-by-step approach",
                steps=[
                    "Analyze the current state and requirements",
                    "Identify files to modify",
                    "Make changes incrementally with tests",
                    "Verify all tests pass",
                ],
                confidence=0.75,
                risks=["Requirements may be unclear"],
                rollback_plan="Revert all changes via git",
            ),
        ]
