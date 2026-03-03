"""Tests for the evaluator (auto-critique & correction) module."""

from metascaffold.evaluator import Evaluator, EvaluationResult
from metascaffold.sandbox import SandboxResult


class TestEvaluator:
    def test_pass_on_zero_exit_code(self):
        """Exit code 0 with clean output should evaluate as 'pass'."""
        evaluator = Evaluator(max_retry_attempts=3)
        result = evaluator.evaluate(
            sandbox_result=SandboxResult(
                exit_code=0, stdout="All 12 tests passed", stderr="", duration_ms=500
            ),
            attempt=1,
        )
        assert isinstance(result, EvaluationResult)
        assert result.verdict == "pass"
        assert result.confidence >= 0.8

    def test_retry_on_test_failure(self):
        """Test failures should trigger 'retry' verdict."""
        evaluator = Evaluator(max_retry_attempts=3)
        result = evaluator.evaluate(
            sandbox_result=SandboxResult(
                exit_code=1,
                stdout="3 passed, 2 failed",
                stderr="FAILED test_auth - AssertionError",
                duration_ms=800,
            ),
            attempt=1,
        )
        assert result.verdict == "retry"
        assert len(result.issues) > 0

    def test_escalate_after_max_attempts(self):
        """Exceeding max attempts should trigger 'escalate'."""
        evaluator = Evaluator(max_retry_attempts=3)
        result = evaluator.evaluate(
            sandbox_result=SandboxResult(
                exit_code=1, stdout="", stderr="Error", duration_ms=100
            ),
            attempt=3,
        )
        assert result.verdict == "escalate"

    def test_backtrack_on_severe_error(self):
        """Severe errors (crash, import error) should trigger 'backtrack'."""
        evaluator = Evaluator(max_retry_attempts=3)
        result = evaluator.evaluate(
            sandbox_result=SandboxResult(
                exit_code=1,
                stdout="",
                stderr="ModuleNotFoundError: No module named 'nonexistent'",
                duration_ms=100,
            ),
            attempt=1,
        )
        assert result.verdict == "backtrack"

    def test_timeout_is_retry(self):
        """Timed out commands should trigger retry (may need more time)."""
        evaluator = Evaluator(max_retry_attempts=3)
        result = evaluator.evaluate(
            sandbox_result=SandboxResult(
                exit_code=-1, stdout="", stderr="Timed out", duration_ms=30000, timed_out=True
            ),
            attempt=1,
        )
        assert result.verdict == "retry"

    def test_result_serializes_to_dict(self):
        """EvaluationResult should serialize to dict for MCP transport."""
        evaluator = Evaluator(max_retry_attempts=3)
        result = evaluator.evaluate(
            sandbox_result=SandboxResult(exit_code=0, stdout="ok", stderr="", duration_ms=100),
            attempt=1,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "verdict" in d
        assert "confidence" in d
        assert "issues" in d
