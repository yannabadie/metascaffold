"""PreToolUse hook — intercepts modifying actions for System 1/2 classification.

This script is called by Claude Code before each tool use.
It reads hook input from stdin (JSON) and outputs a message to stdout.

Exit codes:
- 0: Allow the tool call to proceed
- 2: Block the tool call and show the stdout message to Claude

The hook communicates with the MetaScaffold MCP server via the tools
that Claude already has access to, by injecting a guidance message.
"""

from __future__ import annotations

import json
import sys

# Tools that should trigger classification
_MODIFYING_TOOLS = {"Bash", "Edit", "Write", "NotebookEdit"}


def should_intercept(tool_name: str) -> bool:
    """Check if this tool should be intercepted for classification."""
    return tool_name in _MODIFYING_TOOLS


def format_system2_message(confidence: float, reasoning: str) -> str:
    """Format a message telling Claude to use System 2 deliberation."""
    return (
        f"[MetaScaffold] System 2 activated (confidence: {confidence:.2f}). "
        f"Reason: {reasoning}. "
        f"IMPORTANT: Before proceeding, call metascaffold_plan to create "
        f"a structured plan for this task. Then follow the recommended strategy."
    )


def main():
    """Entry point for the PreToolUse hook."""
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)  # Can't parse input, allow through

    tool_name = hook_input.get("tool_name", "")

    if not should_intercept(tool_name):
        sys.exit(0)  # Allow through

    # For now, output a guidance message suggesting classification.
    # The actual classification happens via the MCP tool that Claude calls.
    print(
        f"[MetaScaffold] Consider calling metascaffold_classify before "
        f"using {tool_name}. This helps determine if deep planning is needed.",
        file=sys.stderr,
    )
    sys.exit(0)  # Allow through (advisory, not blocking)


if __name__ == "__main__":
    main()
