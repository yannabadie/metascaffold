"""Evaluator — auto-critique and correction engine.

Analyzes sandbox execution results and produces a verdict:
- pass: result is acceptable
- retry: try again with corrections (same strategy)
- backtrack: change strategy (return to Planner)
- escalate: request human intervention
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from metascaffold.sandbox import SandboxResult


_SEVERE_ERRORS = re.compile(
    r"ModuleNotFoundError|ImportError|SyntaxError|IndentationError|RecursionError|MemoryError|PermissionError",
    re.IGNORECASE,
)

_TEST_FAILURE = re.compile(
    r"FAIL|failed|error|assert|AssertionError|AssertionError",
    re.IGNORECASE,
)


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

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "issues": [i.to_dict() for i in self.issues],
            "corrections": self.corrections,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
        }


class Evaluator:
    """Evaluates sandbox results and produces verdicts."""

    def __init__(self, max_retry_attempts: int = 3):
        self.max_retry_attempts = max_retry_attempts

    def evaluate(
        self,
        sandbox_result: SandboxResult,
        attempt: int = 1,
    ) -> EvaluationResult:
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
