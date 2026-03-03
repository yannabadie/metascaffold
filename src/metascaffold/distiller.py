"""Distiller — LLM-powered task structuring (Self-Thought Task Distillation).

Transforms raw task descriptions into structured TaskTemplates before planning.
Falls back to passthrough when LLM is unavailable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("metascaffold.distiller")

_DISTILLER_SYSTEM_PROMPT = """You are a task analyst for a coding AI agent.
Given a raw task description and context, extract a structured task template.

Extract:
- objective: clear one-sentence goal
- constraints: technical or business rules that must be respected
- target_files: files likely affected (infer from context)
- variables: key parameters, values, or configuration mentioned

Return ONLY valid JSON:
{
  "objective": "...",
  "constraints": ["..."],
  "target_files": ["..."],
  "variables": {"key": "value"}
}"""


@dataclass
class TaskTemplate:
    """Structured task representation produced by the Distiller."""
    objective: str
    constraints: list[str] = field(default_factory=list)
    target_files: list[str] = field(default_factory=list)
    variables: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "objective": self.objective,
            "constraints": self.constraints,
            "target_files": self.target_files,
            "variables": self.variables,
        }


class Distiller:
    """Transforms raw task text into structured TaskTemplate."""

    def __init__(self, llm_client: object | None = None):
        self._llm = llm_client

    async def distill(self, task: str, context: str) -> TaskTemplate:
        """Distill a raw task into a structured template."""
        if self._llm and getattr(self._llm, "enabled", False):
            result = await self._llm_distill(task, context)
            if result is not None:
                return result

        # Fallback: passthrough with no enrichment
        return TaskTemplate(objective=task)

    async def _llm_distill(self, task: str, context: str) -> TaskTemplate | None:
        """Use LLM for task distillation. Returns None on failure."""
        user_prompt = f"Task: {task}\nContext: {context[:1000]}"
        try:
            resp = await self._llm.complete(
                model="gpt-4.1-nano",
                system_prompt=_DISTILLER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=512,
            )
            if resp.error:
                return None
            data = json.loads(resp.content)
            return TaskTemplate(
                objective=data.get("objective", task),
                constraints=data.get("constraints", []),
                target_files=data.get("target_files", []),
                variables=data.get("variables", {}),
            )
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("LLM distillation failed: %s", e)
            return None
