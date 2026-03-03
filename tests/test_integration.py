"""Integration tests — validate the full cognitive loop end-to-end.

Tests the pipeline: classify -> plan -> sandbox -> evaluate -> telemetry
without requiring a running MCP server or NotebookLM auth.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from metascaffold.classifier import Classifier
from metascaffold.distiller import Distiller
from metascaffold.evaluator import Evaluator
from metascaffold.pipeline import CognitivePipeline, PipelineState
from metascaffold.planner import Planner
from metascaffold.reflector import Reflector
from metascaffold.sandbox import Sandbox, SandboxResult
from metascaffold.telemetry import CognitiveEvent, TelemetryLogger


class TestCognitiveLoop:
    """Tests the full System 2 cognitive loop."""

    def test_full_system2_loop_pass(self, tmp_path):
        """A complete System 2 loop that ends in 'pass'."""
        # 1. Classify
        classifier = Classifier(system2_threshold=0.8, always_system2_tools=["Write"])
        classification = classifier.classify(
            tool_name="Write",
            tool_input={"file_path": str(tmp_path / "test.py")},
            context="Create a new test file for the auth module",
        )
        assert classification.routing == "system2"

        # 2. Plan
        planner = Planner()
        plan = planner.create_plan(
            task="Create a new test file for the auth module",
            context=f"Target: {tmp_path / 'test.py'}",
        )
        assert len(plan.strategies) >= 1

        # 3. Execute in sandbox
        sandbox = Sandbox(work_dir=str(tmp_path), default_timeout_seconds=10)
        # Create a simple Python file and run it
        test_file = tmp_path / "test.py"
        test_file.write_text("print('test passed')")
        result = sandbox.execute(f"python {test_file}")
        assert result.exit_code == 0
        assert "test passed" in result.stdout

        # 4. Evaluate
        evaluator = Evaluator(max_retry_attempts=3)
        evaluation = evaluator.evaluate(sandbox_result=result, attempt=1)
        assert evaluation.verdict == "pass"

        # 5. Telemetry logs everything
        telemetry = TelemetryLogger(
            json_dir=str(tmp_path / "telemetry"),
            sqlite_path=str(tmp_path / "cognitive.db"),
        )
        telemetry.log(CognitiveEvent(
            event_type="classification",
            data={"routing": classification.routing, "confidence": classification.confidence},
        ))
        telemetry.log(CognitiveEvent(
            event_type="plan_created",
            data={"strategies": len(plan.strategies)},
        ))
        telemetry.log(CognitiveEvent(
            event_type="evaluation",
            data={"verdict": evaluation.verdict, "confidence": evaluation.confidence},
        ))
        telemetry.flush()

        # Verify JSON log
        json_files = list((tmp_path / "telemetry").glob("*.json"))
        assert len(json_files) == 1
        with open(json_files[0]) as f:
            events = json.load(f)
        assert len(events) == 3
        assert events[0]["event_type"] == "classification"
        assert events[2]["event_type"] == "evaluation"

    def test_full_system2_loop_retry_then_pass(self, tmp_path):
        """A System 2 loop where first attempt fails, retry succeeds."""
        sandbox = Sandbox(work_dir=str(tmp_path), default_timeout_seconds=10)
        evaluator = Evaluator(max_retry_attempts=3)

        # First attempt: failing script
        fail_script = tmp_path / "fail.py"
        fail_script.write_text("raise ValueError('bug')")
        result1 = sandbox.execute(f"python {fail_script}")
        eval1 = evaluator.evaluate(sandbox_result=result1, attempt=1)
        assert eval1.verdict in ("retry", "backtrack")

        # Second attempt: fixed script
        fix_script = tmp_path / "fix.py"
        fix_script.write_text("print('fixed')")
        result2 = sandbox.execute(f"python {fix_script}")
        eval2 = evaluator.evaluate(sandbox_result=result2, attempt=2)
        assert eval2.verdict == "pass"

    def test_system1_bypasses_planning(self):
        """System 1 tasks should not need planning."""
        classifier = Classifier(system2_threshold=0.8, always_system2_tools=[])
        classification = classifier.classify(
            tool_name="Read",
            tool_input={"file_path": "/tmp/readme.md"},
            context="Read the README file",
        )
        assert classification.routing == "system1"
        # System 1: no planning needed, proceed directly


class TestLLMPipelineIntegration:
    """Integration test for the full LLM cognitive pipeline."""

    async def test_system2_full_pipeline_with_mocked_llm(self):
        """System 2 task should flow through all 6 stages with mocked LLM."""
        mock_llm = AsyncMock()
        mock_llm.enabled = True

        # Stage 1: Classify -> system2
        classify_resp = MagicMock(
            content=json.dumps({"routing": "system2", "confidence": 0.3, "reasoning": "Complex refactor"}),
            error="", model="gpt-5.3-codex", total_tokens=50
        )
        # Stage 2: Distill -> template
        distill_resp = MagicMock(
            content=json.dumps({"objective": "Refactor auth module", "constraints": ["Keep backward compat"], "target_files": ["src/auth.py"], "variables": {}}),
            error=""
        )
        # Stage 3: Plan -> strategies
        plan_resp = MagicMock(
            content=json.dumps({"strategies": [{"id": "A", "description": "TDD refactor", "steps": ["Write test", "Refactor", "Verify"], "confidence": 0.8, "risks": ["Partial migration"], "rollback_plan": "git revert"}], "recommended": "A"}),
            error=""
        )
        # Stage 5: Evaluate -> pass
        eval_resp = MagicMock(
            content=json.dumps({"verdict": "pass", "confidence": 0.9, "feedback": {"failing_tests": [], "error_lines": [], "root_cause": "", "suggested_fix": ""}, "adversarial_findings": [], "revision_allowed": True}),
            error=""
        )

        mock_llm.complete = AsyncMock(side_effect=[classify_resp, distill_resp, plan_resp, eval_resp])

        # Build pipeline with all LLM-powered components
        pipeline = CognitivePipeline(
            classifier=Classifier(llm_client=mock_llm),
            distiller=Distiller(llm_client=mock_llm),
            planner=Planner(llm_client=mock_llm),
            evaluator=Evaluator(llm_client=mock_llm),
        )

        # Stage 1: Classify
        state = PipelineState(task="Refactor auth module", context="src/auth.py needs cleanup")
        state = await pipeline.classify_stage(state)
        assert state.classification.routing == "system2"
        assert state.should_bypass is False

        # Stage 2: Distill
        state = await pipeline.distill_stage(state)
        assert state.template is not None
        assert state.template.objective == "Refactor auth module"

        # Stage 3: Plan
        state = await pipeline.plan_stage(state)
        assert state.plan is not None
        assert len(state.plan.strategies) >= 1

        # Stage 4: Execute (simulated)
        sandbox_result = SandboxResult(exit_code=0, stdout="All 12 tests passed", stderr="", duration_ms=200)

        # Stage 5: Evaluate
        state = await pipeline.evaluate_stage(state, sandbox_result)
        assert state.evaluation is not None
        assert state.evaluation.verdict == "pass"

        # Verify LLM was called 4 times (classify, distill, plan, evaluate)
        assert mock_llm.complete.await_count == 4

    async def test_system1_bypasses_llm_stages(self):
        """System 1 tasks should skip distill and plan even with LLM available."""
        mock_llm = AsyncMock()
        mock_llm.enabled = True

        # Only need classify response (system1 skips distill/plan)
        classify_resp = MagicMock(
            content=json.dumps({"routing": "system1", "confidence": 0.95, "reasoning": "Read-only tool"}),
            error="", model="gpt-5.3-codex", total_tokens=30
        )
        mock_llm.complete = AsyncMock(return_value=classify_resp)

        pipeline = CognitivePipeline(
            classifier=Classifier(llm_client=mock_llm),
            distiller=Distiller(llm_client=mock_llm),
            planner=Planner(llm_client=mock_llm),
        )

        state = PipelineState(task="Read README", context="Simple read")
        state = await pipeline.classify_stage(state)
        assert state.classification.routing == "system1"
        assert state.should_bypass is True

        # Distill and Plan should be skipped
        state = await pipeline.distill_stage(state)
        assert state.template is None

        state = await pipeline.plan_stage(state)
        assert state.plan is None

        # LLM was only called once (classify)
        assert mock_llm.complete.await_count == 1

    async def test_pipeline_with_retry_loop(self):
        """Pipeline should handle retry -> pass flow."""
        mock_llm = AsyncMock()
        mock_llm.enabled = True

        # First evaluate -> retry, second evaluate -> pass
        eval_retry = MagicMock(
            content=json.dumps({"verdict": "retry", "confidence": 0.4, "feedback": {"failing_tests": ["test_login"], "error_lines": [], "root_cause": "Missing null check", "suggested_fix": "Add null check"}, "adversarial_findings": [], "revision_allowed": True}),
            error=""
        )
        eval_pass = MagicMock(
            content=json.dumps({"verdict": "pass", "confidence": 0.9, "feedback": {}, "adversarial_findings": [], "revision_allowed": True}),
            error=""
        )
        mock_llm.complete = AsyncMock(side_effect=[eval_retry, eval_pass])

        pipeline = CognitivePipeline(
            evaluator=Evaluator(llm_client=mock_llm),
        )

        # First attempt -> retry
        state = PipelineState(task="Fix login", context="", attempt=1)
        sandbox_fail = SandboxResult(exit_code=1, stdout="", stderr="AssertionError: test_login", duration_ms=100)
        state = await pipeline.evaluate_stage(state, sandbox_fail)
        assert state.evaluation.verdict == "retry"
        assert state.evaluation.revision_allowed is True

        # Retry
        state = state.with_retry()
        assert state.attempt == 2

        # Second attempt -> pass
        sandbox_pass = SandboxResult(exit_code=0, stdout="All tests passed", stderr="", duration_ms=100)
        state = await pipeline.evaluate_stage(state, sandbox_pass)
        assert state.evaluation.verdict == "pass"
