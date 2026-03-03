"""System 1/2 Classifier — routes tasks to fast or deliberate processing.

Uses heuristic signals (complexity, reversibility, uncertainty, history)
to determine whether a task needs deep reflection (System 2) or can
proceed directly (System 1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_COMPLEX_KEYWORDS = re.compile(
    r"refactor|architect|redesign|migrate|across\s+\d+|entire|all\s+modules|system-wide",
    re.IGNORECASE,
)

_DESTRUCTIVE_COMMANDS = re.compile(
    r"rm\s+-rf|drop\s+table|delete|git\s+reset\s+--hard|git\s+push\s+--force|truncate|format",
    re.IGNORECASE,
)

_SIMPLE_COMMANDS = re.compile(
    r"^(ls|pwd|echo|cat|head|tail|wc|date|whoami|which|type|git\s+status|git\s+log|git\s+diff)(\s|$)",
    re.IGNORECASE,
)


@dataclass
class ClassificationResult:
    """Result of the System 1/2 classification."""
    routing: str
    confidence: float
    reasoning: str
    signals: dict = field(default_factory=dict)


class Classifier:
    """Heuristic classifier for System 1/2 routing."""

    def __init__(
        self,
        system2_threshold: float = 0.8,
        always_system2_tools: list[str] | None = None,
    ):
        self.system2_threshold = system2_threshold
        self.always_system2_tools = always_system2_tools or []

    def classify(
        self,
        tool_name: str,
        tool_input: dict,
        context: str,
        historical_success_rate: float | None = None,
    ) -> ClassificationResult:
        signals = {
            "complexity": "low",
            "reversibility": "high",
            "uncertainty": "low",
            "historical_success_rate": historical_success_rate,
        }
        confidence = 0.9
        reasons: list[str] = []

        if tool_name in self.always_system2_tools:
            return ClassificationResult(
                routing="system2",
                confidence=0.5,
                reasoning=f"Tool '{tool_name}' is configured for mandatory System 2",
                signals=signals,
            )

        if tool_name in ("Read", "Grep", "Glob", "WebSearch", "WebFetch"):
            return ClassificationResult(
                routing="system1",
                confidence=0.95,
                reasoning=f"Read-only tool '{tool_name}'",
                signals=signals,
            )

        command = tool_input.get("command", "")
        if _DESTRUCTIVE_COMMANDS.search(command):
            confidence -= 0.35
            signals["reversibility"] = "low"
            reasons.append("Destructive command detected")

        if _SIMPLE_COMMANDS.match(command):
            confidence += 0.05
            reasons.append("Simple read-only command")

        if _COMPLEX_KEYWORDS.search(context):
            confidence -= 0.25
            signals["complexity"] = "high"
            reasons.append("Complex task keywords detected")

        if historical_success_rate is not None:
            if historical_success_rate < 0.5:
                confidence -= 0.2
                signals["uncertainty"] = "high"
                reasons.append(f"Low historical success rate ({historical_success_rate:.0%})")
            elif historical_success_rate < 0.7:
                confidence -= 0.1
                reasons.append(f"Moderate historical success rate ({historical_success_rate:.0%})")

        confidence = max(0.0, min(1.0, confidence))
        routing = "system1" if confidence >= self.system2_threshold else "system2"
        reasoning = "; ".join(reasons) if reasons else f"Standard {tool_name} operation"

        return ClassificationResult(
            routing=routing,
            confidence=confidence,
            reasoning=reasoning,
            signals=signals,
        )
