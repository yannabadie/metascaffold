"""Tests for the System 1/2 classifier."""

import json
import math
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import pytest

from metascaffold.classifier import Classifier, ClassificationResult


class TestClassifier:
    def test_simple_read_is_system1(self):
        """Simple read operations should route to System 1."""
        c = Classifier(system2_threshold=0.8, always_system2_tools=[])
        result = c.classify(
            tool_name="Read",
            tool_input={"file_path": "/tmp/test.py"},
            context="Read a file to understand the code",
        )
        assert result.routing == "system1"
        assert result.confidence >= 0.8

    def test_multi_file_edit_is_system2(self):
        """Complex multi-file edits should route to System 2."""
        c = Classifier(system2_threshold=0.8, always_system2_tools=[])
        result = c.classify(
            tool_name="Edit",
            tool_input={"file_path": "/src/auth.py"},
            context="Refactor the entire authentication system across 5 modules",
        )
        assert result.routing == "system2"
        assert result.confidence < 0.8

    def test_always_system2_tools(self):
        """Tools in always_system2_tools should always route to System 2."""
        c = Classifier(system2_threshold=0.8, always_system2_tools=["Write"])
        result = c.classify(
            tool_name="Write",
            tool_input={"file_path": "/tmp/new_file.py"},
            context="Create a small utility function",
        )
        assert result.routing == "system2"

    def test_destructive_bash_is_system2(self):
        """Destructive bash commands should route to System 2."""
        c = Classifier(system2_threshold=0.8, always_system2_tools=[])
        result = c.classify(
            tool_name="Bash",
            tool_input={"command": "rm -rf /tmp/project"},
            context="Clean up temporary files",
        )
        assert result.routing == "system2"
        assert result.confidence < 0.8

    def test_classification_returns_all_fields(self):
        """ClassificationResult should have all required fields."""
        c = Classifier(system2_threshold=0.8, always_system2_tools=[])
        result = c.classify(
            tool_name="Bash",
            tool_input={"command": "ls"},
            context="List files",
        )
        assert isinstance(result, ClassificationResult)
        assert result.routing in ("system1", "system2")
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.reasoning, str)
        assert isinstance(result.signals, dict)

    def test_historical_success_rate_lowers_confidence(self):
        """Low historical success rate should lower confidence."""
        c = Classifier(system2_threshold=0.8, always_system2_tools=[])
        result = c.classify(
            tool_name="Edit",
            tool_input={"file_path": "/src/complex.py"},
            context="Fix the flaky test",
            historical_success_rate=0.3,
        )
        assert result.routing == "system2"


# ---------------------------------------------------------------------------
# Helper: fake LLM client for async tests
# ---------------------------------------------------------------------------

@dataclass
class _FakeLLMResponse:
    content: str = ""
    model: str = "fake-model"
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_logprobs: list = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _make_llm_client(*, enabled: bool = True, response: _FakeLLMResponse | None = None):
    """Build a mock LLM client with an async ``complete`` method."""
    client = AsyncMock()
    client.enabled = enabled
    if response is not None:
        client.complete.return_value = response
    return client


# ---------------------------------------------------------------------------
# Async tests — LLM-powered classification (v0.2)
# ---------------------------------------------------------------------------

class TestClassifierLLM:
    @pytest.mark.asyncio
    async def test_llm_classifies_ambiguous_task(self):
        """When LLM is available, ambiguous tasks use LLM classification."""
        llm_response = _FakeLLMResponse(
            content=json.dumps({
                "routing": "system2",
                "confidence": 0.85,
                "reasoning": "Multi-step edit with side effects",
            }),
        )
        mock_llm = _make_llm_client(enabled=True, response=llm_response)
        c = Classifier(llm_client=mock_llm)

        result = await c.classify_async(
            tool_name="Edit",
            tool_input={"file_path": "/src/app.py"},
            context="Refactor the middleware stack",
        )

        assert result.routing == "system2"
        assert result.confidence == pytest.approx(0.85)
        assert "Multi-step edit" in result.reasoning
        assert result.signals.get("source") == "llm"
        mock_llm.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_readonly_tools_skip_llm(self):
        """Read-only tools should fast-path to system1 without calling LLM."""
        mock_llm = _make_llm_client(enabled=True, response=_FakeLLMResponse())
        c = Classifier(llm_client=mock_llm)

        result = await c.classify_async(
            tool_name="Read",
            tool_input={"file_path": "/tmp/test.py"},
            context="Read a file to understand the code",
        )

        assert result.routing == "system1"
        assert result.confidence >= 0.9
        mock_llm.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fallback_to_heuristics_when_llm_disabled(self):
        """When LLM is disabled, classify_async falls back to heuristic logic."""
        mock_llm = _make_llm_client(enabled=False)
        c = Classifier(llm_client=mock_llm)

        result = await c.classify_async(
            tool_name="Bash",
            tool_input={"command": "rm -rf /tmp/old_project"},
            context="Clean up old files",
        )

        assert result.routing == "system2"
        assert result.signals.get("source") == "heuristic"
        mock_llm.complete.assert_not_awaited()


# ---------------------------------------------------------------------------
# Async tests — Entropy-based classification (v0.3)
# ---------------------------------------------------------------------------

class TestClassifierEntropy:
    @pytest.mark.asyncio
    async def test_low_entropy_routes_to_system1(self):
        """Low entropy (high certainty) should route to system1 with compute_level=1."""
        # Codex fallback response (should NOT be used)
        codex_response = _FakeLLMResponse(
            content=json.dumps({
                "routing": "system2",
                "confidence": 0.9,
                "reasoning": "Codex says system2",
            }),
        )
        # Logprobs response: low entropy (p=0.98 vs 0.02)
        logprobs_response = _FakeLLMResponse(
            content=json.dumps({
                "routing": "system1",
                "confidence": 0.95,
                "reasoning": "Simple read operation",
            }),
            token_logprobs=[
                {
                    "token": "system",
                    "logprob": math.log(0.98),
                    "top_logprobs": [
                        {"token": "system", "logprob": math.log(0.98)},
                        {"token": "complex", "logprob": math.log(0.02)},
                    ],
                },
            ],
        )

        mock_llm = _make_llm_client(enabled=True, response=codex_response)
        mock_llm.complete_with_logprobs = AsyncMock(return_value=logprobs_response)
        c = Classifier(llm_client=mock_llm)

        result = await c.classify_async(
            tool_name="Edit",
            tool_input={"file_path": "/src/app.py"},
            context="Fix a typo in a comment",
        )

        assert result.routing == "system1"
        assert result.signals.get("source") == "entropy"
        assert result.signals.get("compute_level") == 1
        assert "entropy" in result.signals
        # Codex complete() should NOT have been called
        mock_llm.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_high_entropy_routes_to_system2(self):
        """High entropy (near-equal probs) should force system2 with compute_level=2."""
        codex_response = _FakeLLMResponse(
            content=json.dumps({
                "routing": "system1",
                "confidence": 0.7,
                "reasoning": "Codex says system1",
            }),
        )
        # Logprobs response: high entropy (p=0.55 vs 0.45)
        logprobs_response = _FakeLLMResponse(
            content=json.dumps({
                "routing": "system1",
                "confidence": 0.6,
                "reasoning": "Model says system1 but is uncertain",
            }),
            token_logprobs=[
                {
                    "token": "system",
                    "logprob": math.log(0.55),
                    "top_logprobs": [
                        {"token": "system", "logprob": math.log(0.55)},
                        {"token": "complex", "logprob": math.log(0.45)},
                    ],
                },
            ],
        )

        mock_llm = _make_llm_client(enabled=True, response=codex_response)
        mock_llm.complete_with_logprobs = AsyncMock(return_value=logprobs_response)
        c = Classifier(llm_client=mock_llm)

        result = await c.classify_async(
            tool_name="Edit",
            tool_input={"file_path": "/src/complex.py"},
            context="Restructure the module",
        )

        assert result.routing == "system2"
        assert result.signals.get("source") == "entropy"
        assert result.signals.get("compute_level") == 2
        assert result.signals.get("entropy") is not None
        # Codex complete() should NOT have been called
        mock_llm.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_medium_entropy_routes_to_system15(self):
        """Medium entropy should keep model routing with compute_level=1.5."""
        codex_response = _FakeLLMResponse(
            content=json.dumps({
                "routing": "system1",
                "confidence": 0.8,
                "reasoning": "Codex says system1",
            }),
        )
        # Logprobs response: medium entropy (p=0.90 vs 0.10 -> ~0.469 bits)
        logprobs_response = _FakeLLMResponse(
            content=json.dumps({
                "routing": "system1",
                "confidence": 0.8,
                "reasoning": "Moderate confidence routing",
            }),
            token_logprobs=[
                {
                    "token": "system",
                    "logprob": math.log(0.90),
                    "top_logprobs": [
                        {"token": "system", "logprob": math.log(0.90)},
                        {"token": "complex", "logprob": math.log(0.10)},
                    ],
                },
            ],
        )

        mock_llm = _make_llm_client(enabled=True, response=codex_response)
        mock_llm.complete_with_logprobs = AsyncMock(return_value=logprobs_response)
        c = Classifier(llm_client=mock_llm)

        result = await c.classify_async(
            tool_name="Bash",
            tool_input={"command": "npm test"},
            context="Run the test suite",
        )

        assert result.signals.get("source") == "entropy"
        assert result.signals.get("compute_level") == 1.5
        assert result.signals.get("entropy") is not None
        # Codex complete() should NOT have been called
        mock_llm.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fallback_to_codex_when_logprobs_fail(self):
        """When logprobs call fails, fall back to codex LLM classification."""
        codex_response = _FakeLLMResponse(
            content=json.dumps({
                "routing": "system2",
                "confidence": 0.85,
                "reasoning": "Codex classification result",
            }),
        )

        mock_llm = _make_llm_client(enabled=True, response=codex_response)
        # complete_with_logprobs raises an error
        mock_llm.complete_with_logprobs = AsyncMock(side_effect=RuntimeError("API error"))
        c = Classifier(llm_client=mock_llm)

        result = await c.classify_async(
            tool_name="Edit",
            tool_input={"file_path": "/src/app.py"},
            context="Complex refactoring task",
        )

        assert result.routing == "system2"
        assert result.signals.get("source") == "llm"
        assert result.confidence == pytest.approx(0.85)
        # Codex complete() should have been called as fallback
        mock_llm.complete.assert_awaited_once()
