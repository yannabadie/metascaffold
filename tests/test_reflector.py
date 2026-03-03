"""Tests for the MARS Reflector component."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from metascaffold.reflector import Reflector, ReflectionResult


class TestReflector:
    def test_reflection_result_dataclass(self):
        """ReflectionResult should hold rules and procedures."""
        r = ReflectionResult(
            rules=["Always validate input before DB queries"],
            procedures=["1. Write test 2. Implement 3. Verify"],
            source_event_count=10,
        )
        assert len(r.rules) == 1
        assert r.source_event_count == 10

    def test_reflection_result_to_dict(self):
        """ReflectionResult should serialize to dict."""
        r = ReflectionResult(rules=["rule1"], procedures=["proc1"], source_event_count=5)
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["rules"] == ["rule1"]
        assert d["source_event_count"] == 5

    async def test_reflect_extracts_rules_from_telemetry(self):
        """Reflector should analyze telemetry events and extract patterns."""
        mock_client = AsyncMock()
        mock_client.enabled = True
        mock_client.complete = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "rules": [
                    "Always run the full test suite after modifying shared modules",
                    "SyntaxError usually indicates a missing import — check imports first",
                ],
                "procedures": [
                    "When a test fails: 1) Read the error 2) Check the assertion 3) Fix minimal code",
                ],
            }),
            error=""
        ))

        events = [
            {"event_type": "evaluation", "data": {"verdict": "retry", "num_issues": 2}},
            {"event_type": "evaluation", "data": {"verdict": "pass", "num_issues": 0}},
            {"event_type": "evaluation", "data": {"verdict": "backtrack", "num_issues": 1}},
        ]

        reflector = Reflector(llm_client=mock_client)
        result = await reflector.reflect(events)
        assert len(result.rules) > 0
        assert len(result.procedures) > 0
        assert result.source_event_count == 3
        mock_client.complete.assert_awaited_once()

    async def test_reflect_returns_empty_when_no_events(self):
        """Reflector should return empty result with no events."""
        mock_client = AsyncMock()
        mock_client.enabled = True

        reflector = Reflector(llm_client=mock_client)
        result = await reflector.reflect([])
        assert result.rules == []
        assert result.source_event_count == 0

    async def test_reflect_fallback_when_llm_disabled(self):
        """When LLM is disabled, return empty reflection."""
        mock_client = AsyncMock()
        mock_client.enabled = False

        events = [{"event_type": "evaluation", "data": {"verdict": "pass"}}]
        reflector = Reflector(llm_client=mock_client)
        result = await reflector.reflect(events)
        assert result.rules == []
        assert result.source_event_count == 1

    async def test_reflect_fallback_on_llm_error(self):
        """When LLM returns an error, fall back gracefully."""
        mock_client = AsyncMock()
        mock_client.enabled = True
        mock_client.complete = AsyncMock(return_value=MagicMock(
            content="",
            error="codex exec timed out"
        ))

        events = [{"event_type": "evaluation", "data": {"verdict": "retry"}}]
        reflector = Reflector(llm_client=mock_client)
        result = await reflector.reflect(events)
        assert result.rules == []
        assert result.source_event_count == 1
