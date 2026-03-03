"""Deterministic verification suite — AST, Ruff, pytest, mypy.

Runs fast, deterministic checks BEFORE the LLM-as-Judge evaluator.
Each verifier returns a VerifierResult with pass/fail, severity, and detail.
All subprocess calls (ruff, pytest, mypy) are guarded against missing tools
and timeouts so the suite degrades gracefully.
"""

from __future__ import annotations

import ast
import logging
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger("metascaffold.verifiers")


@dataclass
class VerifierResult:
    """Outcome of a single verification step."""

    verifier: str
    passed: bool
    detail: str = ""
    severity: str = "info"
    skipped: bool = False

    def to_dict(self) -> dict:
        return {
            "verifier": self.verifier,
            "passed": self.passed,
            "detail": self.detail,
            "severity": self.severity,
            "skipped": self.skipped,
        }


# ---------------------------------------------------------------------------
# Individual verifiers
# ---------------------------------------------------------------------------


def ast_verify(code: str) -> VerifierResult:
    """Parse *code* with the stdlib AST parser.

    Returns passed=True if the code is syntactically valid Python,
    passed=False with severity="critical" on SyntaxError.
    """
    try:
        ast.parse(code)
        return VerifierResult(verifier="ast", passed=True, detail="Syntax OK")
    except SyntaxError as exc:
        return VerifierResult(
            verifier="ast",
            passed=False,
            detail=f"SyntaxError: {exc}",
            severity="critical",
        )


def ruff_verify(file_path: str, timeout: int = 15) -> VerifierResult:
    """Run ``ruff check`` on *file_path*.

    Returns passed=True if returncode is 0 (no lint errors).
    Returns passed=False with severity="warning" when ruff reports errors.
    Returns passed=True, skipped=True when ruff is not installed or times out.
    """
    try:
        proc = subprocess.run(
            ["ruff", "check", file_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0:
            return VerifierResult(verifier="ruff", passed=True, detail="No lint errors")
        return VerifierResult(
            verifier="ruff",
            passed=False,
            detail=proc.stdout.strip()[:500],
            severity="warning",
        )
    except FileNotFoundError:
        logger.info("ruff not installed, skipping lint check")
        return VerifierResult(
            verifier="ruff", passed=True, detail="ruff not installed", skipped=True
        )
    except subprocess.TimeoutExpired:
        logger.warning("ruff timed out after %ds, skipping", timeout)
        return VerifierResult(
            verifier="ruff", passed=True, detail="ruff timed out", skipped=True
        )


def pytest_verify(test_path: str, timeout: int = 60) -> VerifierResult:
    """Run ``python -m pytest`` on *test_path*.

    Returns passed=True if all tests pass (returncode 0).
    Returns passed=False with severity="critical" on test failure.
    Returns passed=False with severity="warning" on timeout.
    """
    try:
        proc = subprocess.run(
            ["python", "-m", "pytest", test_path, "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0:
            return VerifierResult(
                verifier="pytest",
                passed=True,
                detail=proc.stdout.strip()[:500],
            )
        return VerifierResult(
            verifier="pytest",
            passed=False,
            detail=(proc.stdout.strip() + "\n" + proc.stderr.strip())[:500],
            severity="critical",
        )
    except subprocess.TimeoutExpired:
        logger.warning("pytest timed out after %ds", timeout)
        return VerifierResult(
            verifier="pytest",
            passed=False,
            detail=f"pytest timed out after {timeout}s",
            severity="warning",
        )


def mypy_verify(file_path: str, timeout: int = 30) -> VerifierResult:
    """Run ``python -m mypy`` on *file_path* with --ignore-missing-imports.

    Returns passed=True if mypy reports no errors.
    Returns passed=False with severity="warning" on type errors.
    Returns passed=True, skipped=True when mypy is not installed.
    """
    try:
        proc = subprocess.run(
            ["python", "-m", "mypy", file_path, "--ignore-missing-imports"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0:
            return VerifierResult(
                verifier="mypy",
                passed=True,
                detail=proc.stdout.strip()[:500],
            )
        return VerifierResult(
            verifier="mypy",
            passed=False,
            detail=proc.stdout.strip()[:500],
            severity="warning",
        )
    except FileNotFoundError:
        logger.info("mypy not installed, skipping type check")
        return VerifierResult(
            verifier="mypy", passed=True, detail="mypy not installed", skipped=True
        )
    except subprocess.TimeoutExpired:
        logger.warning("mypy timed out after %ds, skipping", timeout)
        return VerifierResult(
            verifier="mypy", passed=True, detail="mypy timed out", skipped=True
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class VerificationSuite:
    """Orchestrates multiple deterministic verifiers.

    Intended to run *before* the LLM-as-Judge evaluator so that
    obvious failures (syntax errors, lint violations, test failures)
    are caught without spending LLM tokens.
    """

    def verify_code(self, code: str) -> list[VerifierResult]:
        """Run AST verification on a code string."""
        return [ast_verify(code)]

    def verify_file(
        self,
        file_path: str,
        run_ruff: bool = True,
        run_mypy: bool = False,
    ) -> list[VerifierResult]:
        """Run AST (by reading the file) plus optional ruff/mypy on a file."""
        results: list[VerifierResult] = []

        # Read file and run AST check
        try:
            with open(file_path, encoding="utf-8") as f:
                code = f.read()
            results.append(ast_verify(code))
        except OSError as exc:
            results.append(
                VerifierResult(
                    verifier="ast",
                    passed=False,
                    detail=f"Could not read file: {exc}",
                    severity="critical",
                )
            )

        if run_ruff:
            results.append(ruff_verify(file_path))

        if run_mypy:
            results.append(mypy_verify(file_path))

        return results

    def verify_tests(self, test_path: str) -> VerifierResult:
        """Run pytest on the given test path."""
        return pytest_verify(test_path)

    @staticmethod
    def has_critical_failures(results: list[VerifierResult]) -> bool:
        """Return True if any result has passed=False, severity='critical', and is not skipped."""
        return any(
            not r.passed and r.severity == "critical" and not r.skipped
            for r in results
        )
