"""Tests for the cognitive pipeline orchestrator."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from metascaffold.pipeline import CognitivePipeline, PipelineState


class TestPipelineState:
    def test_pipeline_state_dataclass(self):
        """PipelineState should track all 6 stages."""
        state = PipelineState(task="Fix bug", context="src/main.py")
        assert state.task == "Fix bug"
        assert state.classification is None
        assert state.template is None
        assert state.plan is None
        assert state.execution is None
        assert state.evaluation is None
        assert state.reflection is None

    def test_should_bypass_false_when_no_classification(self):
        """should_bypass should be False when no classification yet."""
        state = PipelineState(task="test", context="")
        assert state.should_bypass is False

    def test_should_bypass_true_for_system1(self):
        """should_bypass should be True for system1 routing."""
        state = PipelineState(task="test", context="",
                             classification=MagicMock(routing="system1"))
        assert state.should_bypass is True

    def test_should_bypass_false_for_system2(self):
        """should_bypass should be False for system2 routing."""
        state = PipelineState(task="test", context="",
                             classification=MagicMock(routing="system2"))
        assert state.should_bypass is False

    def test_retry_increments_attempt(self):
        """Retry should increment attempt counter."""
        state = PipelineState(task="test", context="", attempt=1)
        new_state = state.with_retry()
        assert new_state.attempt == 2
        assert new_state.execution is None
        assert new_state.evaluation is None

    def test_should_escalate(self):
        """Exceeding max attempts should signal escalation."""
        state = PipelineState(task="test", context="", attempt=3, max_attempts=3)
        assert state.should_escalate is True

    def test_should_not_escalate(self):
        """Below max attempts should not escalate."""
        state = PipelineState(task="test", context="", attempt=1, max_attempts=3)
        assert state.should_escalate is False

    def test_to_dict(self):
        """PipelineState should serialize to dict."""
        state = PipelineState(task="test", context="ctx")
        d = state.to_dict()
        assert d["task"] == "test"
        assert d["attempt"] == 1
        assert d["classification"] is None


class TestCognitivePipeline:
    async def test_classify_stage_system1(self):
        """System 1 classification should set bypass."""
        mock_classifier = AsyncMock()
        mock_classifier.classify_async = AsyncMock(return_value=MagicMock(
            routing="system1", confidence=0.95, reasoning="Read-only",
        ))

        pipeline = CognitivePipeline(classifier=mock_classifier)
        state = await pipeline.classify_stage(
            PipelineState(task="Read file", context="")
        )
        assert state.classification.routing == "system1"
        assert state.should_bypass is True

    async def test_classify_stage_system2(self):
        """System 2 classification should not set bypass."""
        mock_classifier = AsyncMock()
        mock_classifier.classify_async = AsyncMock(return_value=MagicMock(
            routing="system2", confidence=0.4, reasoning="Complex refactor",
        ))

        pipeline = CognitivePipeline(classifier=mock_classifier)
        state = await pipeline.classify_stage(
            PipelineState(task="Refactor auth", context="")
        )
        assert state.classification.routing == "system2"
        assert state.should_bypass is False

    async def test_distill_stage_bypassed_for_system1(self):
        """Distill should be skipped for system1 tasks."""
        mock_distiller = AsyncMock()
        mock_distiller.distill = AsyncMock()

        state = PipelineState(task="test", context="",
                             classification=MagicMock(routing="system1"))
        pipeline = CognitivePipeline(distiller=mock_distiller)
        result = await pipeline.distill_stage(state)
        assert result.template is None
        mock_distiller.distill.assert_not_awaited()

    async def test_distill_stage_runs_for_system2(self):
        """Distill should run for system2 tasks."""
        mock_template = MagicMock(objective="Structured task")
        mock_distiller = AsyncMock()
        mock_distiller.distill = AsyncMock(return_value=mock_template)

        state = PipelineState(task="test", context="",
                             classification=MagicMock(routing="system2"))
        pipeline = CognitivePipeline(distiller=mock_distiller)
        result = await pipeline.distill_stage(state)
        assert result.template is not None
        assert result.template.objective == "Structured task"

    async def test_plan_stage_bypassed_for_system1(self):
        """Plan should be skipped for system1 tasks."""
        mock_planner = AsyncMock()
        mock_planner.create_plan_async = AsyncMock()

        state = PipelineState(task="test", context="",
                             classification=MagicMock(routing="system1"))
        pipeline = CognitivePipeline(planner=mock_planner)
        result = await pipeline.plan_stage(state)
        assert result.plan is None
        mock_planner.create_plan_async.assert_not_awaited()

    async def test_evaluate_stage(self):
        """Evaluate stage should call evaluator with sandbox result."""
        mock_eval_result = MagicMock(verdict="pass")
        mock_evaluator = AsyncMock()
        mock_evaluator.evaluate_async = AsyncMock(return_value=mock_eval_result)

        mock_sandbox_result = MagicMock(exit_code=0)
        state = PipelineState(task="test", context="")
        pipeline = CognitivePipeline(evaluator=mock_evaluator)
        result = await pipeline.evaluate_stage(state, mock_sandbox_result)
        assert result.execution is not None
        assert result.evaluation.verdict == "pass"

    async def test_reflect_stage(self):
        """Reflect stage should call reflector with events."""
        mock_reflection = MagicMock(rules=["rule1"], source_event_count=2)
        mock_reflector = AsyncMock()
        mock_reflector.reflect = AsyncMock(return_value=mock_reflection)

        events = [{"event": "test1"}, {"event": "test2"}]
        state = PipelineState(task="test", context="")
        pipeline = CognitivePipeline(reflector=mock_reflector)
        result = await pipeline.reflect_stage(state, events)
        assert result.reflection is not None
        assert result.reflection.rules == ["rule1"]

    async def test_no_classifier_returns_unmodified_state(self):
        """Without classifier, classify_stage returns state unchanged."""
        pipeline = CognitivePipeline()
        state = PipelineState(task="test", context="")
        result = await pipeline.classify_stage(state)
        assert result.classification is None


class TestAdaptiveCompute:
    """Tests for System 1 / 1.5 / 2 adaptive compute routing."""

    def test_compute_level_defaults_to_none(self):
        """New PipelineState should have compute_level=None."""
        state = PipelineState(task="test", context="")
        assert state.compute_level is None

    def test_system1_bypasses_distill_and_plan(self):
        """compute_level=1 should bypass both distill and plan."""
        state = PipelineState(task="test", context="", compute_level=1)
        assert state.should_bypass_distill is True
        assert state.should_bypass_plan is True
        assert state.should_bypass is True

    def test_system15_runs_distill_skips_plan(self):
        """compute_level=1.5 should run distill but skip plan."""
        state = PipelineState(task="test", context="", compute_level=1.5)
        assert state.should_bypass_distill is False
        assert state.should_bypass_plan is True
        assert state.should_bypass is False

    def test_system2_runs_all(self):
        """compute_level=2 should run both distill and plan."""
        state = PipelineState(task="test", context="", compute_level=2)
        assert state.should_bypass_distill is False
        assert state.should_bypass_plan is False
        assert state.should_bypass is False

    async def test_classify_stage_sets_compute_level(self):
        """classify_stage should extract compute_level from classifier signals."""
        mock_classifier = AsyncMock()
        mock_classifier.classify_async = AsyncMock(return_value=MagicMock(
            routing="system2",
            signals={"compute_level": 1.5},
        ))

        pipeline = CognitivePipeline(classifier=mock_classifier)
        state = await pipeline.classify_stage(
            PipelineState(task="Medium task", context="some context")
        )
        assert state.compute_level == 1.5

    async def test_distill_skipped_for_system1(self):
        """Distill should not be called when compute_level=1."""
        mock_distiller = AsyncMock()
        mock_distiller.distill = AsyncMock()

        state = PipelineState(task="test", context="", compute_level=1)
        pipeline = CognitivePipeline(distiller=mock_distiller)
        result = await pipeline.distill_stage(state)
        assert result.template is None
        mock_distiller.distill.assert_not_awaited()

    async def test_distill_runs_for_system15(self):
        """Distill should run when compute_level=1.5."""
        mock_template = MagicMock(objective="Distilled task")
        mock_distiller = AsyncMock()
        mock_distiller.distill = AsyncMock(return_value=mock_template)

        state = PipelineState(task="test", context="", compute_level=1.5)
        pipeline = CognitivePipeline(distiller=mock_distiller)
        result = await pipeline.distill_stage(state)
        assert result.template is not None
        assert result.template.objective == "Distilled task"
        mock_distiller.distill.assert_awaited_once()
