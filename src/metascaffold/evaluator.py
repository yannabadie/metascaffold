"""Evaluator — auto-critique and correction engine.

Analyzes sandbox execution results and produces a verdict:
- pass: result is acceptable
- retry: try again with corrections (same strategy)
- backtrack: change strategy (return to Planner)
- escalate: request human intervention

v0.2 adds LLM-as-Judge with SOFAI-LM feedback, adversarial check,
and PAG (Plan-and-Gate) revision control.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from metascaffold.sandbox import SandboxResult

logger = logging.getLogger("metascaffold.evaluator")

_SEVERE_ERRORS = re.compile(
    r"ModuleNotFoundError|ImportError|SyntaxError|IndentationError|RecursionError|MemoryError|PermissionError",
    re.IGNORECASE,
)

_TEST_FAILURE = re.compile(
    r"FAIL|failed|error|assert|AssertionError|AssertionError",
    re.IGNORECASE,
)

_EVALUATOR_SYSTEM_PROMPT = """\
You are a code execution evaluator. Analyze the sandbox execution result and \
produce a structured JSON verdict.

Respond with ONLY valid JSON matching this schema:
{
  "verdict": "pass" | "retry" | "backtrack" | "escalate",
  "confidence": <float 0.0-1.0>,
  "feedback": {
    "failing_tests": [<list of test names that failed>],
    "error_lines": [<list of key error lines from output>],
    "root_cause": "<brief root-cause analysis>",
    "suggested_fix": "<actionable fix suggestion>"
  },
  "adversarial_findings": [
    {"issue": "<security or logic issue>", "severity": "low" | "medium" | "high"}
  ],
  "revision_allowed": <boolean - false if the output is too broken or dangerous to auto-fix>
}

Rules:
- "pass" only if exit_code==0 AND no security/logic issues
- "retry" if the error looks fixable with a code change
- "backtrack" if the approach is fundamentally wrong (import errors, architecture issues)
- "escalate" if human judgment is needed
- Always check for security issues (injection, path traversal, credential leaks)
- Set revision_allowed=false if the output is too dangerous or ambiguous to auto-fix
"""

_EVALUATOR_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "retry", "backtrack", "escalate"]},
        "confidence": {"type": "number"},
        "feedback": {
            "type": "object",
            "properties": {
                "failing_tests": {"type": "array", "items": {"type": "string"}},
                "error_lines": {"type": "array", "items": {"type": "string"}},
                "root_cause": {"type": "string"},
                "suggested_fix": {"type": "string"},
            },
            "required": ["failing_tests", "error_lines", "root_cause", "suggested_fix"],
            "additionalProperties": False,
        },
        "adversarial_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["issue", "severity"],
                "additionalProperties": False,
            },
        },
        "revision_allowed": {"type": "boolean"},
    },
    "required": ["verdict", "confidence", "feedback", "adversarial_findings", "revision_allowed"],
    "additionalProperties": False,
}


@dataclass
class Issue:
    """A single issue found during evaluation."""
    type: str
    detail: str
    severity: str

    def to_dict(self) -> dict:
        return {"type": self.type, "detail": self.detail, "severity": self.severity}


@dataclass
class EvaluationResult:
    """Result of the auto-evaluation."""
    verdict: str
    confidence: float
    issues: list[Issue] = field(default_factory=list)
    corrections: list[dict] = field(default_factory=list)
    attempt: int = 1
    max_attempts: int = 3
    feedback: dict = field(default_factory=dict)
    adversarial_findings: list[dict] = field(default_factory=list)
    revision_allowed: bool = True

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "issues": [i.to_dict() for i in self.issues],
            "corrections": self.corrections,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "feedback": self.feedback,
            "adversarial_findings": self.adversarial_findings,
            "revision_allowed": self.revision_allowed,
        }


class Evaluator:
    """Evaluates sandbox results and produces verdicts.

    Supports two modes:
    - Sync heuristic evaluation via evaluate() (v0.1 behaviour)
    - Async LLM-as-Judge evaluation via evaluate_async() with heuristic fallback (v0.2)
    """

    def __init__(self, max_retry_attempts: int = 3, llm_client=None):
        self.max_retry_attempts = max_retry_attempts
        self._llm_client = llm_client

    def evaluate(
        self,
        sandbox_result: SandboxResult,
        attempt: int = 1,
    ) -> EvaluationResult:
        """Synchronous evaluation using heuristics only (v0.1 API)."""
        return self._heuristic_evaluate(sandbox_result, attempt)

    async def evaluate_async(
        self,
        sandbox_result: SandboxResult,
        attempt: int = 1,
    ) -> EvaluationResult:
        """Async evaluation: tries LLM-as-Judge first, falls back to heuristics.

        The LLM path provides structured SOFAI feedback, adversarial analysis,
        and PAG revision gating. On any LLM failure, falls back silently to
        heuristic evaluation.
        """
        if self._llm_client and self._llm_client.enabled:
            llm_result = await self._llm_evaluate(sandbox_result, attempt)
            if llm_result is not None:
                return llm_result
            logger.warning("LLM evaluation failed, falling back to heuristics")

        return self._heuristic_evaluate(sandbox_result, attempt)

    def _heuristic_evaluate(
        self,
        sandbox_result: SandboxResult,
        attempt: int = 1,
    ) -> EvaluationResult:
        """Rule-based heuristic evaluation (original v0.1 logic)."""
        issues: list[Issue] = []
        combined_output = sandbox_result.stdout + "\n" + sandbox_result.stderr

        if sandbox_result.timed_out:
            issues.append(Issue(
                type="timeout",
                detail=f"Command timed out after {sandbox_result.duration_ms}ms",
                severity="medium",
            ))

        if _SEVERE_ERRORS.search(sandbox_result.stderr):
            issues.append(Issue(
                type="severe_error",
                detail=sandbox_result.stderr.strip()[:200],
                severity="critical",
            ))

        if sandbox_result.exit_code != 0 and _TEST_FAILURE.search(combined_output):
            issues.append(Issue(
                type="test_failure",
                detail=sandbox_result.stderr.strip()[:200],
                severity="medium",
            ))

        if sandbox_result.exit_code == 0 and not issues:
            return EvaluationResult(
                verdict="pass",
                confidence=0.9,
                issues=issues,
                attempt=attempt,
                max_attempts=self.max_retry_attempts,
            )

        if any(i.severity == "critical" for i in issues):
            return EvaluationResult(
                verdict="backtrack",
                confidence=0.3,
                issues=issues,
                attempt=attempt,
                max_attempts=self.max_retry_attempts,
            )

        if attempt >= self.max_retry_attempts:
            return EvaluationResult(
                verdict="escalate",
                confidence=0.2,
                issues=issues,
                attempt=attempt,
                max_attempts=self.max_retry_attempts,
            )

        return EvaluationResult(
            verdict="retry",
            confidence=0.5,
            issues=issues,
            attempt=attempt,
            max_attempts=self.max_retry_attempts,
        )

    async def _llm_evaluate(
        self,
        sandbox_result: SandboxResult,
        attempt: int = 1,
    ) -> EvaluationResult | None:
        """LLM-as-Judge evaluation with SOFAI feedback and adversarial check.

        Returns None on any failure so the caller can fall back to heuristics.
        """
        user_prompt = (
            f"exit_code: {sandbox_result.exit_code}\n"
            f"timed_out: {sandbox_result.timed_out}\n"
            f"duration_ms: {sandbox_result.duration_ms}\n"
            f"attempt: {attempt} / {self.max_retry_attempts}\n"
            f"\n--- stdout (truncated) ---\n{sandbox_result.stdout[:2000]}\n"
            f"\n--- stderr (truncated) ---\n{sandbox_result.stderr[:2000]}"
        )

        try:
            response = await self._llm_client.complete(
                model="default",
                system_prompt=_EVALUATOR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=2048,
                response_format=_EVALUATOR_RESPONSE_SCHEMA,
            )

            if response.error:
                logger.warning("LLM evaluate returned error: %s", response.error)
                return None

            data = json.loads(response.content)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("LLM evaluate parse/call failed: %s", exc)
            return None

        verdict = data.get("verdict", "retry")
        confidence = float(data.get("confidence", 0.5))
        feedback = data.get("feedback", {})
        adversarial_findings = data.get("adversarial_findings", [])
        revision_allowed = data.get("revision_allowed", True)

        # Adversarial override: if LLM said "pass" but found security/logic issues,
        # downgrade to "retry" so the issues get addressed.
        if verdict == "pass" and adversarial_findings:
            logger.info(
                "Adversarial override: downgrading 'pass' to 'retry' due to %d findings",
                len(adversarial_findings),
            )
            verdict = "retry"

        # PAG gate: respect the LLM's revision_allowed flag
        # (caller should check this before attempting auto-fix)

        # Max attempts: if we've exhausted retries, escalate
        if attempt >= self.max_retry_attempts and verdict == "retry":
            logger.info(
                "Max attempts reached (%d/%d), escalating from 'retry'",
                attempt, self.max_retry_attempts,
            )
            verdict = "escalate"

        return EvaluationResult(
            verdict=verdict,
            confidence=confidence,
            issues=[],  # LLM feedback replaces issue-based analysis
            attempt=attempt,
            max_attempts=self.max_retry_attempts,
            feedback=feedback,
            adversarial_findings=adversarial_findings,
            revision_allowed=revision_allowed,
        )
