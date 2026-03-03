"""Integration tests — validate the full cognitive loop end-to-end.

Tests the pipeline: classify -> plan -> sandbox -> evaluate -> telemetry
without requiring a running MCP server or NotebookLM auth.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metascaffold.classifier import Classifier
from metascaffold.distiller import Distiller
from metascaffold.evaluator import Evaluator
from metascaffold.pipeline import CognitivePipeline, PipelineState
from metascaffold.planner import Planner
from metascaffold.reflection_memory import ReflectionMemory
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


class TestV03Integration:
    """Integration tests for v0.3 features: entropy routing, verifiers, memory."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_entropy_routing(self):
        """Full pipeline where entropy classifies System 2, flows through
        distill -> plan -> evaluate -> reflect.  Verify compute_level is set."""
        mock_llm = AsyncMock()
        mock_llm.enabled = True

        # --- Entropy classification response (high entropy -> System 2) ---
        entropy_classify_resp = MagicMock(
            content=json.dumps({
                "routing": "system2",
                "confidence": 0.85,
                "reasoning": "Complex refactor across modules",
            }),
            error="",
            model="gpt-4.1-nano",
            # Logprobs with a routing token at high entropy (> 0.5 threshold)
            token_logprobs=[
                {"token": "system", "logprob": -0.5, "top_logprobs": [
                    {"token": "system", "logprob": -0.5},
                    {"token": "simple", "logprob": -1.2},
                    {"token": "complex", "logprob": -1.8},
                ]},
            ],
        )
        mock_llm.complete_with_logprobs = AsyncMock(return_value=entropy_classify_resp)

        # --- Distill response ---
        distill_resp = MagicMock(
            content=json.dumps({
                "objective": "Refactor authentication module",
                "constraints": ["Keep backward compatibility"],
                "target_files": ["src/auth.py"],
                "variables": [],
            }),
            error="",
        )

        # --- Plan response ---
        plan_resp = MagicMock(
            content=json.dumps({
                "strategies": [{
                    "id": "A",
                    "description": "Incremental TDD refactor",
                    "steps": ["Write tests", "Refactor", "Verify"],
                    "confidence": 0.8,
                    "risks": ["Breaking changes"],
                    "rollback_plan": "git revert",
                }],
                "recommended": "A",
            }),
            error="",
        )

        # --- Evaluate response ---
        eval_resp = MagicMock(
            content=json.dumps({
                "verdict": "pass",
                "confidence": 0.92,
                "feedback": {
                    "failing_tests": [],
                    "error_lines": [],
                    "root_cause": "",
                    "suggested_fix": "",
                },
                "adversarial_findings": [],
                "revision_allowed": True,
            }),
            error="",
        )

        # --- Reflect response ---
        reflect_resp = MagicMock(
            content=json.dumps({
                "rules": ["Always run tests after refactoring"],
                "procedures": ["Use incremental TDD for auth changes"],
            }),
            error="",
        )

        mock_llm.complete = AsyncMock(side_effect=[distill_resp, plan_resp, eval_resp, reflect_resp])

        pipeline = CognitivePipeline(
            classifier=Classifier(llm_client=mock_llm),
            distiller=Distiller(llm_client=mock_llm),
            planner=Planner(llm_client=mock_llm),
            evaluator=Evaluator(llm_client=mock_llm),
            reflector=Reflector(llm_client=mock_llm),
        )

        # Stage 1: Classify (entropy-based)
        state = PipelineState(task="Refactor auth module", context="src/auth.py needs cleanup")
        state = await pipeline.classify_stage(state)
        assert state.classification is not None
        assert state.classification.routing == "system2"
        assert state.compute_level == 2
        assert state.classification.signals.get("source") == "entropy"
        assert state.should_bypass is False

        # Stage 2: Distill (not bypassed for compute_level=2)
        state = await pipeline.distill_stage(state)
        assert state.template is not None
        assert state.template.objective == "Refactor authentication module"

        # Stage 3: Plan (not bypassed for compute_level=2)
        state = await pipeline.plan_stage(state)
        assert state.plan is not None
        assert len(state.plan.strategies) >= 1

        # Stage 4: Execute (simulated)
        sandbox_result = SandboxResult(
            exit_code=0, stdout="All 8 tests passed", stderr="", duration_ms=150,
        )

        # Stage 5: Evaluate
        state = await pipeline.evaluate_stage(state, sandbox_result)
        assert state.evaluation is not None
        assert state.evaluation.verdict == "pass"

        # Stage 6: Reflect
        events = [
            {"event_type": "evaluation", "data": {"verdict": "pass"}},
        ]
        state = await pipeline.reflect_stage(state, events)
        assert state.reflection is not None
        assert "Always run tests after refactoring" in state.reflection.rules

        # Verify entropy probe was called
        mock_llm.complete_with_logprobs.assert_awaited_once()
        # Verify LLM complete was called 4 times (distill, plan, evaluate, reflect)
        assert mock_llm.complete.await_count == 4

    @pytest.mark.asyncio
    async def test_verifier_short_circuits_broken_code(self):
        """Pass broken Python syntax as code_output; verify evaluator returns
        'backtrack' WITHOUT the LLM being called."""
        mock_llm = AsyncMock()
        mock_llm.enabled = True
        mock_llm.complete = AsyncMock()

        evaluator = Evaluator(llm_client=mock_llm)

        # Broken Python: unclosed parenthesis
        broken_code = "def foo(\n    pass"
        sandbox_result = SandboxResult(
            exit_code=1, stdout="", stderr="SyntaxError", duration_ms=10,
        )

        result = await evaluator.evaluate_async(
            sandbox_result=sandbox_result,
            attempt=1,
            code_output=broken_code,
        )

        # Verifier should short-circuit to backtrack
        assert result.verdict == "backtrack"
        assert result.confidence >= 0.9
        assert any(
            issue.type == "verifier" and issue.severity == "critical"
            for issue in result.issues
        )
        # LLM must NOT have been called
        mock_llm.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_system15_skips_plan_but_runs_distill(self):
        """Mock entropy classification to return compute_level=1.5.
        Verify distill runs but plan is skipped."""
        mock_llm = AsyncMock()
        mock_llm.enabled = True

        # Entropy response with medium entropy (0.3 < entropy <= 0.5 -> compute_level=1.5)
        # For 2 tokens: p=0.9 -> H = -0.9*log2(0.9) - 0.1*log2(0.1) ≈ 0.469 bits
        # ln(0.9) ≈ -0.1054, ln(0.1) ≈ -2.3026
        entropy_resp = MagicMock(
            content=json.dumps({
                "routing": "system1",
                "confidence": 0.7,
                "reasoning": "Moderate complexity task",
            }),
            error="",
            model="gpt-4.1-nano",
            token_logprobs=[
                {"token": "system1", "logprob": -0.1054, "top_logprobs": [
                    {"token": "system1", "logprob": -0.1054},
                    {"token": "system2", "logprob": -2.3026},
                ]},
            ],
        )
        mock_llm.complete_with_logprobs = AsyncMock(return_value=entropy_resp)

        # Distill response (will be used since compute_level=1.5 > 1)
        distill_resp = MagicMock(
            content=json.dumps({
                "objective": "Update config values",
                "constraints": [],
                "target_files": ["config.toml"],
                "variables": [],
            }),
            error="",
        )
        mock_llm.complete = AsyncMock(return_value=distill_resp)

        pipeline = CognitivePipeline(
            classifier=Classifier(llm_client=mock_llm),
            distiller=Distiller(llm_client=mock_llm),
            planner=Planner(llm_client=mock_llm),
        )

        state = PipelineState(task="Update config values", context="config.toml changes")

        # Classify -> compute_level=1.5
        state = await pipeline.classify_stage(state)
        assert state.compute_level == 1.5
        assert state.should_bypass_distill is False  # 1.5 > 1 -> distill runs
        assert state.should_bypass_plan is True       # 1.5 < 2 -> plan skipped

        # Distill -> should execute
        state = await pipeline.distill_stage(state)
        assert state.template is not None
        assert state.template.objective == "Update config values"

        # Plan -> should be skipped (compute_level < 2)
        state = await pipeline.plan_stage(state)
        assert state.plan is None

        # LLM.complete called once (for distill), plan was skipped
        assert mock_llm.complete.await_count == 1

    @pytest.mark.asyncio
    async def test_entropy_fallback_to_codex_llm(self):
        """Mock complete_with_logprobs to raise an exception; verify that
        _llm_classify is called as fallback."""
        mock_llm = AsyncMock()
        mock_llm.enabled = True

        # Entropy probe raises an exception
        mock_llm.complete_with_logprobs = AsyncMock(
            side_effect=ConnectionError("OpenAI API unreachable"),
        )

        # Fallback LLM classification response
        llm_classify_resp = MagicMock(
            content=json.dumps({
                "routing": "system2",
                "confidence": 0.8,
                "reasoning": "Complex multi-file refactor",
            }),
            error="",
            model="gpt-5.3-codex",
            total_tokens=45,
        )
        mock_llm.complete = AsyncMock(return_value=llm_classify_resp)

        classifier = Classifier(llm_client=mock_llm)
        result = await classifier.classify_async(
            tool_name="Edit",
            tool_input={"file_path": "src/core.py"},
            context="Refactor the core module across 5 files",
        )

        # Entropy probe was attempted
        mock_llm.complete_with_logprobs.assert_awaited_once()
        # Fallback LLM classify was called
        mock_llm.complete.assert_awaited_once()
        # Result came from the LLM fallback
        assert result.routing == "system2"
        assert result.signals.get("source") == "llm"

    @pytest.mark.asyncio
    async def test_reflection_memory_integration(self, tmp_path):
        """Create reflector with memory_path, run reflection, verify rules
        are saved to disk and can be loaded by a fresh ReflectionMemory."""
        mock_llm = AsyncMock()
        mock_llm.enabled = True

        memory_file = tmp_path / "reflection_memory.json"

        reflect_resp = MagicMock(
            content=json.dumps({
                "rules": [
                    "Always run tests before committing",
                    "Check for import errors before refactoring",
                ],
                "procedures": ["Run full test suite after each change"],
            }),
            error="",
        )
        mock_llm.complete = AsyncMock(return_value=reflect_resp)

        reflector = Reflector(llm_client=mock_llm, memory_path=memory_file)

        events = [
            {"event_type": "evaluation", "data": {"verdict": "backtrack", "reason": "ImportError"}},
            {"event_type": "evaluation", "data": {"verdict": "pass"}},
        ]

        result = await reflector.reflect(events)

        # Reflection returned rules
        assert len(result.rules) == 2
        assert "Always run tests before committing" in result.rules
        assert result.source_event_count == 2

        # Memory file was persisted
        assert memory_file.exists()

        # Load the memory with a fresh instance and verify rules
        fresh_memory = ReflectionMemory(storage_path=memory_file)
        fresh_memory.load()
        assert len(fresh_memory.rules) == 2
        rule_contents = [r.content for r in fresh_memory.rules]
        assert "Always run tests before committing" in rule_contents
        assert "Check for import errors before refactoring" in rule_contents

        # Run reflection again with the same rules -> should reinforce, not duplicate
        result2 = await reflector.reflect(events)
        assert len(reflector.memory.rules) == 2  # still 2, not 4
        # Both rules should have reinforcement_count >= 1
        for rule in reflector.memory.rules:
            assert rule.reinforcement_count >= 1
