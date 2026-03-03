"""Integration tests — validate the full cognitive loop end-to-end.

Tests the pipeline: classify -> plan -> sandbox -> evaluate -> telemetry
without requiring a running MCP server or NotebookLM auth.
"""

import json
from pathlib import Path

from metascaffold.classifier import Classifier
from metascaffold.evaluator import Evaluator
from metascaffold.planner import Planner
from metascaffold.sandbox import Sandbox
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
