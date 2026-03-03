"""MetaScaffold MCP Server — cognitive middleware for Claude Code.

Exposes 6 tools: classify, plan, sandbox_exec, evaluate, nlm_query, telemetry_query.
Run with: uv run python src/metascaffold/server.py
Register with: claude mcp add metascaffold -- uv --directory . run python src/metascaffold/server.py
"""

from __future__ import annotations

import logging
import sys
from typing import Annotated

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from metascaffold.classifier import Classifier
from metascaffold.config import load_config
from metascaffold.evaluator import Evaluator
from metascaffold.notebooklm_bridge import NotebookLMBridge
from metascaffold.planner import Planner
from metascaffold.sandbox import Sandbox, SandboxResult
from metascaffold.telemetry import CognitiveEvent, TelemetryLogger

# Configure logging to stderr (NEVER print to stdout in stdio MCP servers)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("metascaffold")

# Load configuration
config = load_config()

# Initialize components
classifier = Classifier(
    system2_threshold=config.classifier.system2_threshold,
    always_system2_tools=config.classifier.always_system2_tools,
)
planner = Planner()
sandbox = Sandbox(default_timeout_seconds=config.sandbox.default_timeout_seconds)
evaluator = Evaluator(max_retry_attempts=config.sandbox.max_retry_attempts)
telemetry = TelemetryLogger(
    json_dir=config.telemetry.json_dir,
    sqlite_path=config.telemetry.sqlite_path,
)
nlm_bridge = NotebookLMBridge(
    enabled=config.notebooklm.enabled,
    default_notebook=config.notebooklm.default_notebook,
    fallback_on_error=config.notebooklm.fallback_on_error,
)

# Create MCP server
mcp = FastMCP("metascaffold")


@mcp.tool()
def metascaffold_classify(
    tool_name: Annotated[str, Field(description="Name of the tool being called (Bash, Edit, Write, etc.)")],
    tool_input: Annotated[str, Field(description="JSON string of the tool's input parameters")],
    context: Annotated[str, Field(description="Natural language description of what is being done")],
) -> dict:
    """Classify a task as System 1 (fast) or System 2 (deliberate).

    Analyzes complexity, reversibility, uncertainty, and historical success rate
    to determine whether the task needs deep reflection before execution.
    """
    import json as _json
    parsed_input = _json.loads(tool_input) if isinstance(tool_input, str) else tool_input

    # Check historical success rate from telemetry
    historical_rate = telemetry.get_success_rate(context[:50])

    result = classifier.classify(
        tool_name=tool_name,
        tool_input=parsed_input,
        context=context,
        historical_success_rate=historical_rate,
    )

    # Log classification event
    telemetry.log(CognitiveEvent(
        event_type="classification",
        data={
            "routing": result.routing,
            "confidence": result.confidence,
            "tool_name": tool_name,
            "reasoning": result.reasoning,
        },
    ))
    telemetry.flush()

    return {
        "routing": result.routing,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "signals": result.signals,
    }


@mcp.tool()
async def metascaffold_plan(
    task: Annotated[str, Field(description="Description of the task to plan")],
    context: Annotated[str, Field(description="Additional context: files involved, scope, constraints")],
) -> dict:
    """Create a structured execution plan for a System 2 task.

    Decomposes the task into strategies with steps, risks, confidence scores,
    and rollback plans. Optionally consults NotebookLM for domain knowledge.
    """
    # Optionally consult NotebookLM
    nlm_insights = ""
    if nlm_bridge.enabled:
        nlm_result = await nlm_bridge.query(
            f"What are best practices for: {task}? Consider: {context}"
        )
        if nlm_result.success:
            nlm_insights = nlm_result.content

    plan = planner.create_plan(
        task=task,
        context=context,
        notebooklm_insights=nlm_insights,
    )

    # Log plan creation
    telemetry.log(CognitiveEvent(
        event_type="plan_created",
        data={
            "task": task,
            "num_strategies": len(plan.strategies),
            "recommended": plan.recommended,
        },
    ))
    telemetry.flush()

    return plan.to_dict()


@mcp.tool()
def metascaffold_sandbox_exec(
    command: Annotated[str, Field(description="Shell command to execute in the sandbox")],
    timeout_seconds: Annotated[int, Field(description="Timeout in seconds (default: 30)")] = 30,
) -> dict:
    """Execute a command in a sandboxed subprocess with timeout and output capture.

    Provides isolation through restricted subprocess execution.
    Captures stdout, stderr, exit code, and execution duration.
    """
    result = sandbox.execute(command=command, timeout_seconds=timeout_seconds)

    # Log execution
    telemetry.log(CognitiveEvent(
        event_type="execution_result",
        data={
            "command": command[:100],
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "duration_ms": result.duration_ms,
        },
    ))
    telemetry.flush()

    return result.to_dict()


@mcp.tool()
def metascaffold_evaluate(
    exit_code: Annotated[int, Field(description="Exit code from the executed command")],
    stdout: Annotated[str, Field(description="Standard output from the command")],
    stderr: Annotated[str, Field(description="Standard error from the command")],
    duration_ms: Annotated[int, Field(description="Execution duration in milliseconds")],
    attempt: Annotated[int, Field(description="Current attempt number (1-based)")] = 1,
    timed_out: Annotated[bool, Field(description="Whether the command timed out")] = False,
) -> dict:
    """Evaluate the result of a sandboxed execution.

    Produces a verdict: pass, retry, backtrack, or escalate.
    Detects test failures, severe errors, and timeout conditions.
    """
    sandbox_result = SandboxResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=duration_ms,
        timed_out=timed_out,
    )
    result = evaluator.evaluate(sandbox_result=sandbox_result, attempt=attempt)

    # Log evaluation
    telemetry.log(CognitiveEvent(
        event_type="evaluation",
        data={
            "verdict": result.verdict,
            "confidence": result.confidence,
            "attempt": result.attempt,
            "num_issues": len(result.issues),
        },
    ))
    telemetry.flush()

    return result.to_dict()


@mcp.tool()
async def metascaffold_nlm_query(
    question: Annotated[str, Field(description="Question to ask the NotebookLM knowledge base")],
    notebook: Annotated[str, Field(description="Notebook name (uses default if empty)")] = "",
) -> dict:
    """Query the NotebookLM knowledge base for domain-specific insights.

    Returns sourced answers from the MetaScaffold research corpus.
    Degrades gracefully if NotebookLM is unavailable.
    """
    result = await nlm_bridge.query(
        question=question,
        notebook=notebook or None,
    )
    return {"success": result.success, "content": result.content, "reason": result.reason}


@mcp.tool()
def metascaffold_telemetry_query(
    task_type: Annotated[str, Field(description="Task type to query success rate for")],
) -> dict:
    """Query cognitive telemetry for historical success rates.

    Returns the success rate for a given task type based on past evaluations.
    Used by the Classifier to improve routing accuracy over time.
    """
    rate = telemetry.get_success_rate(task_type)
    return {
        "task_type": task_type,
        "success_rate": rate,
        "has_data": rate is not None,
    }


if __name__ == "__main__":
    logger.info("Starting MetaScaffold MCP Server...")
    mcp.run(transport="stdio")
