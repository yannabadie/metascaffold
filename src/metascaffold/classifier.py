"""System 1/2 Classifier — routes tasks to fast or deliberate processing.

Uses heuristic signals (complexity, reversibility, uncertainty, history)
to determine whether a task needs deep reflection (System 2) or can
proceed directly (System 1).

v0.2 adds optional LLM-powered classification with heuristic fallback.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("metascaffold.classifier")

# ---------------------------------------------------------------------------
# Regex patterns for heuristic classification
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Constants for fast-path routing
# ---------------------------------------------------------------------------

_READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "Read", "Grep", "Glob", "WebSearch", "WebFetch",
})

# ---------------------------------------------------------------------------
# LLM system prompt for semantic classification
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM_PROMPT = """\
You are a task-routing classifier for a developer-tools IDE agent.
Given a tool invocation and its context, decide whether the task should be
handled by System 1 (fast, automatic execution) or System 2 (slow,
deliberate planning before execution).

Criteria for System 2:
- The change is irreversible or hard to undo
- Multiple files or systems are affected
- The task involves complex reasoning, refactoring, or architecture decisions
- There is significant uncertainty about the correct approach
- The context suggests high stakes (production, security, data loss)

Criteria for System 1:
- The change is trivial and easily reversible
- Only one file or a small scope is affected
- The task is well-understood with a clear, single-step solution
- Read-only or informational operations

Respond with ONLY a JSON object (no markdown fences):
{"routing": "system1" or "system2", "confidence": 0.0 to 1.0, "reasoning": "brief explanation"}
"""


@dataclass
class ClassificationResult:
    """Result of the System 1/2 classification."""
    routing: str
    confidence: float
    reasoning: str
    signals: dict = field(default_factory=dict)


class Classifier:
    """Hybrid classifier for System 1/2 routing.

    Supports optional LLM-powered classification (v0.2) with automatic
    fallback to heuristic-only classification (v0.1) when the LLM client
    is unavailable or fails.
    """

    def __init__(
        self,
        system2_threshold: float = 0.8,
        always_system2_tools: list[str] | None = None,
        llm_client: object | None = None,
    ):
        self.system2_threshold = system2_threshold
        self.always_system2_tools = always_system2_tools or []
        self._llm = llm_client

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def classify(
        self,
        tool_name: str,
        tool_input: dict,
        context: str,
        historical_success_rate: float | None = None,
    ) -> ClassificationResult:
        """Synchronous classification — heuristic only (v0.1 compat)."""
        return self._heuristic_classify(
            tool_name, tool_input, context, historical_success_rate,
        )

    async def classify_async(
        self,
        tool_name: str,
        tool_input: dict,
        context: str,
        historical_success_rate: float | None = None,
    ) -> ClassificationResult:
        """Async classification — tries LLM first, falls back to heuristics.

        Fast-paths:
        - Read-only tools -> system1 immediately (no LLM call)
        - always_system2_tools -> system2 immediately (no LLM call)

        For everything else, attempts LLM classification if available,
        then falls back to heuristic classification on any failure.
        """
        # Fast-path: always_system2_tools
        if tool_name in self.always_system2_tools:
            return ClassificationResult(
                routing="system2",
                confidence=0.5,
                reasoning=f"Tool '{tool_name}' is configured for mandatory System 2",
                signals={"source": "fast-path"},
            )

        # Fast-path: read-only tools
        if tool_name in _READ_ONLY_TOOLS:
            return ClassificationResult(
                routing="system1",
                confidence=0.95,
                reasoning=f"Read-only tool '{tool_name}'",
                signals={"source": "fast-path"},
            )

        # Try LLM classification if available
        if getattr(self._llm, "enabled", False):
            llm_result = await self._llm_classify(
                tool_name, tool_input, context,
            )
            if llm_result is not None:
                return llm_result

        # Fallback to heuristic classification
        logger.debug("Falling back to heuristic classification for %s", tool_name)
        return self._heuristic_classify(
            tool_name, tool_input, context, historical_success_rate,
        )

    # -------------------------------------------------------------------
    # Private: LLM classification
    # -------------------------------------------------------------------

    async def _llm_classify(
        self,
        tool_name: str,
        tool_input: dict,
        context: str,
    ) -> ClassificationResult | None:
        """Call LLM for semantic classification. Returns None on failure."""
        user_prompt = (
            f"Tool: {tool_name}\n"
            f"Input: {json.dumps(tool_input, default=str)}\n"
            f"Context: {context}"
        )
        try:
            response = await self._llm.complete(
                model="gpt-5.3-codex",
                system_prompt=_CLASSIFIER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )

            if response.error:
                logger.warning("LLM classification error: %s", response.error)
                return None

            data = json.loads(response.content)
            routing = data.get("routing", "system2")
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "LLM classification")

            # Validate routing value
            if routing not in ("system1", "system2"):
                logger.warning("Invalid LLM routing value: %s", routing)
                return None

            return ClassificationResult(
                routing=routing,
                confidence=max(0.0, min(1.0, confidence)),
                reasoning=reasoning,
                signals={"source": "llm"},
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to parse LLM classification response: %s", exc)
            return None
        except Exception as exc:
            logger.warning("Unexpected LLM classification failure: %s", exc)
            return None

    # -------------------------------------------------------------------
    # Private: heuristic classification (v0.1 logic)
    # -------------------------------------------------------------------

    def _heuristic_classify(
        self,
        tool_name: str,
        tool_input: dict,
        context: str,
        historical_success_rate: float | None = None,
    ) -> ClassificationResult:
        """Pure heuristic classification using regex patterns and thresholds."""
        signals: dict = {
            "complexity": "low",
            "reversibility": "high",
            "uncertainty": "low",
            "historical_success_rate": historical_success_rate,
            "source": "heuristic",
        }
        confidence = 0.9
        reasons: list[str] = []

        if tool_name in self.always_system2_tools:
            signals["source"] = "heuristic"
            return ClassificationResult(
                routing="system2",
                confidence=0.5,
                reasoning=f"Tool '{tool_name}' is configured for mandatory System 2",
                signals=signals,
            )

        if tool_name in _READ_ONLY_TOOLS:
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
