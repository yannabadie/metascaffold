"""PostToolUse hook — sends execution results to the evaluator.

This script is called by Claude Code after each tool use.
It reads hook input from stdin (JSON) and can output guidance to stderr.
"""

from __future__ import annotations

import json
import sys


def parse_tool_result(exit_code: int, stdout: str, stderr: str) -> dict:
    """Parse a tool execution result into a structured dict."""
    return {
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
    }


def main():
    """Entry point for the PostToolUse hook."""
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")

    # Only evaluate modifying tools
    if tool_name not in {"Bash", "Edit", "Write", "NotebookEdit"}:
        sys.exit(0)

    # Advisory message suggesting evaluation
    tool_result = hook_input.get("tool_result", {})
    if isinstance(tool_result, dict) and tool_result.get("exit_code", 0) != 0:
        print(
            f"[MetaScaffold] {tool_name} returned non-zero exit code. "
            f"Consider calling metascaffold_evaluate to assess the result.",
            file=sys.stderr,
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
