"""Tests for the System 1/2 classifier."""

import json
from dataclasses import dataclass
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
