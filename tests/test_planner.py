"""Tests for the planner (decomposition & strategy) module."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from metascaffold.planner import Planner, Plan, Strategy


class TestPlanner:
    def test_plan_has_at_least_one_strategy(self):
        """Every plan should propose at least one strategy."""
        planner = Planner()
        plan = planner.create_plan(
            task="Add input validation to the login form",
            context="Simple single-file change in auth.py",
        )
        assert isinstance(plan, Plan)
        assert len(plan.strategies) >= 1

    def test_plan_has_recommended_strategy(self):
        """Plan should indicate which strategy is recommended."""
        planner = Planner()
        plan = planner.create_plan(
            task="Refactor database module",
            context="Move from raw SQL to ORM pattern across 3 files",
        )
        assert plan.recommended in [s.id for s in plan.strategies]

    def test_each_strategy_has_steps(self):
        """Each strategy should have concrete steps."""
        planner = Planner()
        plan = planner.create_plan(
            task="Add caching layer",
            context="Implement in-memory caching for API responses",
        )
        for strategy in plan.strategies:
            assert isinstance(strategy, Strategy)
            assert len(strategy.steps) >= 1
            assert strategy.confidence > 0.0
            assert isinstance(strategy.risks, list)
            assert isinstance(strategy.rollback_plan, str)

    def test_plan_includes_task_description(self):
        """Plan should echo back the original task."""
        planner = Planner()
        plan = planner.create_plan(task="Fix bug in parser", context="Off-by-one error")
        assert plan.task == "Fix bug in parser"

    def test_plan_serializes_to_dict(self):
        """Plan should be serializable to a dict for MCP transport."""
        planner = Planner()
        plan = planner.create_plan(task="Test task", context="Test context")
        d = plan.to_dict()
        assert isinstance(d, dict)
        assert "task" in d
        assert "strategies" in d
        assert "recommended" in d


class TestPlannerLLM:
    """Tests for LLM-powered planning via create_plan_async."""

    @pytest.mark.asyncio
    async def test_llm_generates_contextual_strategies(self):
        """LLM should produce strategies with context-aware content."""
        llm_response_payload = {
            "strategies": [
                {
                    "id": "A",
                    "description": "Rotate JWT signing keys and add expiry validation",
                    "steps": [
                        "Audit current JWT token handling in auth middleware",
                        "Add token expiry validation to the verification flow",
                        "Implement signing key rotation with backward compatibility",
                        "Write integration tests for expired and rotated tokens",
                    ],
                    "confidence": 0.9,
                    "risks": ["Key rotation may invalidate active sessions"],
                    "rollback_plan": "Revert to previous signing key and remove expiry check",
                },
                {
                    "id": "B",
                    "description": "Replace JWT with opaque session tokens",
                    "steps": [
                        "Create session store backed by Redis",
                        "Issue opaque tokens instead of JWTs",
                        "Migrate existing sessions gradually",
                        "Remove JWT dependencies",
                    ],
                    "confidence": 0.65,
                    "risks": ["Requires Redis infrastructure", "Session migration complexity"],
                    "rollback_plan": "Fall back to JWT-based auth",
                },
            ],
            "recommended": "A",
        }

        mock_client = AsyncMock()
        mock_client.enabled = True
        mock_client.complete = AsyncMock(
            return_value=MagicMock(
                content=json.dumps(llm_response_payload),
                error="",
            )
        )

        planner = Planner(llm_client=mock_client)
        plan = await planner.create_plan_async(
            task="Fix JWT authentication vulnerability",
            context="The auth service uses HS256 JWTs with no expiry validation",
        )

        # Plan should have the LLM-generated strategies
        assert isinstance(plan, Plan)
        assert len(plan.strategies) == 2
        assert plan.recommended == "A"

        # Strategies should contain JWT-specific content (context-aware)
        descriptions = [s.description for s in plan.strategies]
        assert any("JWT" in d or "jwt" in d.lower() for d in descriptions)

        # Each strategy should be a proper Strategy dataclass
        for strategy in plan.strategies:
            assert isinstance(strategy, Strategy)
            assert len(strategy.steps) >= 1
            assert strategy.confidence > 0.0
            assert isinstance(strategy.risks, list)
            assert isinstance(strategy.rollback_plan, str)

        # LLM complete should have been called exactly once
        mock_client.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fallback_to_heuristics_when_llm_disabled(self):
        """When LLM is disabled, create_plan_async falls back to heuristic plan."""
        mock_client = AsyncMock()
        mock_client.enabled = False

        planner = Planner(llm_client=mock_client)
        plan = await planner.create_plan_async(
            task="Fix the login bug",
            context="Users cannot log in after password reset",
        )

        # Should still produce a valid plan via heuristic fallback
        assert isinstance(plan, Plan)
        assert plan.task == "Fix the login bug"
        assert len(plan.strategies) >= 1

        # The heuristic path for "Fix" tasks produces bugfix strategies
        assert plan.strategies[0].id == "A"
        assert "bug" in plan.strategies[0].description.lower() or "fix" in plan.strategies[0].description.lower() or "reproduce" in plan.strategies[0].description.lower()

        # LLM complete should NOT have been called
        mock_client.complete.assert_not_awaited()
