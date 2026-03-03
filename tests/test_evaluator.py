"""Tests for the evaluator (auto-critique & correction) module."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

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

    def test_result_serializes_new_fields_to_dict(self):
        """EvaluationResult.to_dict() should include feedback, adversarial_findings, revision_allowed."""
        result = EvaluationResult(
            verdict="retry",
            confidence=0.7,
            feedback={"root_cause": "test"},
            adversarial_findings=[{"issue": "xss"}],
            revision_allowed=False,
        )
        d = result.to_dict()
        assert d["feedback"] == {"root_cause": "test"}
        assert d["adversarial_findings"] == [{"issue": "xss"}]
        assert d["revision_allowed"] is False


class TestEvaluatorLLM:
    """Tests for LLM-as-Judge evaluation with SOFAI feedback."""

    @pytest.mark.asyncio
    async def test_llm_evaluates_with_semantic_feedback(self):
        """LLM returns retry with full SOFAI feedback — verify fields populated."""
        llm_response = {
            "verdict": "retry",
            "confidence": 0.65,
            "feedback": {
                "failing_tests": ["test_login"],
                "error_lines": ["line 42: AssertionError"],
                "root_cause": "Missing auth token in header",
                "suggested_fix": "Add Authorization header to request",
            },
            "adversarial_findings": [],
            "revision_allowed": True,
        }
        mock_client = AsyncMock()
        mock_client.enabled = True
        mock_client.complete = AsyncMock(return_value=MagicMock(
            content=json.dumps(llm_response),
            error="",
        ))

        evaluator = Evaluator(max_retry_attempts=3, llm_client=mock_client)
        result = await evaluator.evaluate_async(
            sandbox_result=SandboxResult(
                exit_code=1,
                stdout="3 passed, 1 failed",
                stderr="FAILED test_login - AssertionError",
                duration_ms=800,
            ),
            attempt=1,
        )

        assert result.verdict == "retry"
        assert result.confidence == 0.65
        assert result.feedback["root_cause"] == "Missing auth token in header"
        assert result.feedback["suggested_fix"] == "Add Authorization header to request"
        assert result.revision_allowed is True
        mock_client.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_adversarial_check_downgrades_pass(self):
        """LLM returns 'pass' but with adversarial findings — verdict should downgrade to 'retry'."""
        llm_response = {
            "verdict": "pass",
            "confidence": 0.85,
            "feedback": {
                "failing_tests": [],
                "error_lines": [],
                "root_cause": "",
                "suggested_fix": "",
            },
            "adversarial_findings": [
                {"issue": "SQL injection in user input", "severity": "high"},
            ],
            "revision_allowed": True,
        }
        mock_client = AsyncMock()
        mock_client.enabled = True
        mock_client.complete = AsyncMock(return_value=MagicMock(
            content=json.dumps(llm_response),
            error="",
        ))

        evaluator = Evaluator(max_retry_attempts=3, llm_client=mock_client)
        result = await evaluator.evaluate_async(
            sandbox_result=SandboxResult(
                exit_code=0, stdout="All tests passed", stderr="", duration_ms=500,
            ),
            attempt=1,
        )

        assert result.verdict == "retry", "Adversarial findings must downgrade 'pass' to 'retry'"
        assert len(result.adversarial_findings) == 1
        assert result.adversarial_findings[0]["issue"] == "SQL injection in user input"

    @pytest.mark.asyncio
    async def test_pag_blocks_empty_retry(self):
        """LLM returns retry with revision_allowed=False — PAG gate should block revision."""
        llm_response = {
            "verdict": "retry",
            "confidence": 0.4,
            "feedback": {},
            "adversarial_findings": [],
            "revision_allowed": False,
        }
        mock_client = AsyncMock()
        mock_client.enabled = True
        mock_client.complete = AsyncMock(return_value=MagicMock(
            content=json.dumps(llm_response),
            error="",
        ))

        evaluator = Evaluator(max_retry_attempts=3, llm_client=mock_client)
        result = await evaluator.evaluate_async(
            sandbox_result=SandboxResult(
                exit_code=1, stdout="", stderr="Error", duration_ms=200,
            ),
            attempt=1,
        )

        assert result.verdict == "retry"
        assert result.revision_allowed is False

    @pytest.mark.asyncio
    async def test_fallback_to_heuristics_when_llm_fails(self):
        """When LLM is disabled, evaluate_async falls back to heuristic evaluation."""
        mock_client = AsyncMock()
        mock_client.enabled = False

        evaluator = Evaluator(max_retry_attempts=3, llm_client=mock_client)
        result = await evaluator.evaluate_async(
            sandbox_result=SandboxResult(
                exit_code=0, stdout="ok", stderr="", duration_ms=100,
            ),
            attempt=1,
        )

        assert result.verdict == "pass"
        assert result.feedback == {}
        assert result.adversarial_findings == []
        assert result.revision_allowed is True

    @pytest.mark.asyncio
    async def test_llm_max_attempts_escalate(self):
        """LLM returns retry but attempt >= max → escalate."""
        llm_response = {
            "verdict": "retry",
            "confidence": 0.5,
            "feedback": {"root_cause": "flaky"},
            "adversarial_findings": [],
            "revision_allowed": True,
        }
        mock_client = AsyncMock()
        mock_client.enabled = True
        mock_client.complete = AsyncMock(return_value=MagicMock(
            content=json.dumps(llm_response),
            error="",
        ))

        evaluator = Evaluator(max_retry_attempts=3, llm_client=mock_client)
        result = await evaluator.evaluate_async(
            sandbox_result=SandboxResult(
                exit_code=1, stdout="", stderr="Error", duration_ms=200,
            ),
            attempt=3,
        )

        assert result.verdict == "escalate"
