"""Tests for the planner (decomposition & strategy) module."""

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
