"""Reflector — MARS reflection loop for learning from telemetry.

Analyzes recent cognitive events (evaluations, backtracks, escalations)
and extracts reusable rules and procedures via LLM analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("metascaffold.reflector")

_REFLECTOR_SYSTEM_PROMPT = """You are a metacognitive reflector for a coding AI agent.
Analyze the batch of cognitive telemetry events below and extract learning patterns.

Extract two types of artifacts:
1. **Rules**: Normative constraints the agent should always follow (e.g., "Always run tests after modifying shared code")
2. **Procedures**: Step-by-step strategies that worked well or should replace failed approaches

Focus on:
- Patterns in failures (what keeps going wrong?)
- Patterns in successes (what keeps working?)
- Recurring backtracks or escalations (what should the agent avoid?)

Return a JSON object with rules and procedures arrays."""

_REFLECTOR_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "rules": {"type": "array", "items": {"type": "string"}},
        "procedures": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["rules", "procedures"],
    "additionalProperties": False,
}


@dataclass
class ReflectionResult:
    """Result of the MARS reflection analysis."""
    rules: list[str] = field(default_factory=list)
    procedures: list[str] = field(default_factory=list)
    source_event_count: int = 0

    def to_dict(self) -> dict:
        return {
            "rules": self.rules,
            "procedures": self.procedures,
            "source_event_count": self.source_event_count,
        }


class Reflector:
    """Analyzes telemetry to extract reusable rules and procedures."""

    def __init__(self, llm_client: object | None = None):
        self._llm = llm_client

    async def reflect(self, events: list[dict]) -> ReflectionResult:
        """Analyze a batch of telemetry events and extract patterns."""
        if not events:
            return ReflectionResult(source_event_count=0)

        if self._llm and getattr(self._llm, "enabled", False):
            result = await self._llm_reflect(events)
            if result is not None:
                return result

        # Fallback: no reflection without LLM
        return ReflectionResult(source_event_count=len(events))

    async def _llm_reflect(self, events: list[dict]) -> ReflectionResult | None:
        """Use LLM to analyze telemetry and extract patterns."""
        recent = events[-50:]
        events_text = json.dumps(recent, indent=2)[:4000]
        user_prompt = f"Telemetry events ({len(recent)} most recent):\n{events_text}"

        try:
            resp = await self._llm.complete(
                model="o3-mini",
                system_prompt=_REFLECTOR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=1024,
                response_format=_REFLECTOR_RESPONSE_SCHEMA,
            )
            if resp.error:
                return None
            data = json.loads(resp.content)
            return ReflectionResult(
                rules=data.get("rules", []),
                procedures=data.get("procedures", []),
                source_event_count=len(events),
            )
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("LLM reflection failed: %s", e)
            return None
