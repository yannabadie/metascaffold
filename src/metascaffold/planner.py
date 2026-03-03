"""Planner — decomposes System 2 tasks into strategies with steps, risks, and rollback plans.

The Planner produces structured plans that the Evaluator can later assess.
It uses heuristic decomposition by default and can optionally delegate to an
LLM for context-aware strategy generation.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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


_PLANNER_SYSTEM_PROMPT = """\
You are a software engineering planner. Given a task and its context, generate \
1 to 3 execution strategies. Return ONLY valid JSON with no extra text.

The JSON schema is:
{
  "strategies": [
    {
      "id": "A",
      "description": "Short description of the strategy",
      "steps": ["Step 1", "Step 2", "..."],
      "confidence": 0.85,
      "risks": ["Risk 1", "Risk 2"],
      "rollback_plan": "How to undo this strategy"
    }
  ],
  "recommended": "A"
}

Rules:
- Generate between 1 and 3 strategies, labelled A, B, C.
- Each strategy must have at least 2 steps, a confidence between 0 and 1, \
at least 1 risk, and a rollback plan.
- The "recommended" field must be the id of the best strategy.
- Tailor strategies to the specific task and context provided.
- Return ONLY the JSON object, no markdown fences or commentary.
"""

_PLANNER_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "strategies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                    "risks": {"type": "array", "items": {"type": "string"}},
                    "rollback_plan": {"type": "string"},
                },
                "required": ["id", "description", "steps", "confidence", "risks", "rollback_plan"],
                "additionalProperties": False,
            },
        },
        "recommended": {"type": "string"},
    },
    "required": ["strategies", "recommended"],
    "additionalProperties": False,
}


class Planner:
    """Planner that decomposes tasks into strategies.

    Supports both synchronous heuristic planning and async LLM-powered planning.
    """

    def __init__(self, llm_client: object | None = None) -> None:
        self._llm_client = llm_client

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

    # ------------------------------------------------------------------
    # Async LLM-powered planning
    # ------------------------------------------------------------------

    async def create_plan_async(
        self,
        task: str,
        context: str,
        notebooklm_insights: str = "",
    ) -> Plan:
        """Create a plan using LLM if available, falling back to heuristics."""
        if self._llm_client and getattr(self._llm_client, "enabled", False):
            plan = await self._llm_plan(task, context, notebooklm_insights)
            if plan is not None:
                return plan
            logger.warning("LLM planning failed; falling back to heuristics")

        return self.create_plan(task, context, notebooklm_insights)

    async def _llm_plan(
        self,
        task: str,
        context: str,
        notebooklm_insights: str,
    ) -> Plan | None:
        """Call the LLM to generate a context-aware plan.

        Returns *None* on any failure so the caller can fall back gracefully.
        """
        user_prompt_parts = [f"Task: {task}", f"Context: {context}"]
        if notebooklm_insights:
            user_prompt_parts.append(f"Domain insights: {notebooklm_insights}")
        user_prompt = "\n".join(user_prompt_parts)

        try:
            response = await self._llm_client.complete(
                model="",  # let the client pick the default model
                system_prompt=_PLANNER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.2,
                max_tokens=2048,
                response_format=_PLANNER_RESPONSE_SCHEMA,
            )

            if response.error:
                logger.error("LLM returned error: %s", response.error)
                return None

            data = json.loads(response.content)
            strategies = [
                Strategy(
                    id=s["id"],
                    description=s["description"],
                    steps=s["steps"],
                    confidence=float(s["confidence"]),
                    risks=s["risks"],
                    rollback_plan=s["rollback_plan"],
                )
                for s in data["strategies"]
            ]
            recommended = data.get("recommended", strategies[0].id if strategies else "A")

            return Plan(
                task=task,
                strategies=strategies,
                recommended=recommended,
                notebooklm_insights=notebooklm_insights,
            )

        except (json.JSONDecodeError, KeyError, TypeError, IndexError) as exc:
            logger.error("Failed to parse LLM plan response: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error during LLM planning: %s", exc)
            return None
