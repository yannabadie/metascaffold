"""Tests for the Claude Code hook scripts.

Tests the hook logic in isolation (without requiring a running MCP server).
"""

import json
import sys
from unittest.mock import patch

from hooks.pre_tool_gate import should_intercept, format_system2_message
from hooks.post_tool_evaluate import parse_tool_result


class TestPreToolGate:
    def test_read_tool_not_intercepted(self):
        """Read-only tools should not be intercepted."""
        assert should_intercept("Read") is False

    def test_edit_tool_intercepted(self):
        """Edit tool should be intercepted."""
        assert should_intercept("Edit") is True

    def test_bash_tool_intercepted(self):
        """Bash tool should be intercepted."""
        assert should_intercept("Bash") is True

    def test_write_tool_intercepted(self):
        """Write tool should be intercepted."""
        assert should_intercept("Write") is True

    def test_system2_message_format(self):
        """System 2 activation message should be properly formatted."""
        msg = format_system2_message(confidence=0.65, reasoning="Complex refactor")
        assert "System 2" in msg
        assert "0.65" in msg


class TestPostToolEvaluate:
    def test_parse_success_result(self):
        """Successful tool results should parse correctly."""
        result = parse_tool_result(exit_code=0, stdout="All tests passed", stderr="")
        assert result["exit_code"] == 0
        assert "passed" in result["stdout"].lower()

    def test_parse_failure_result(self):
        """Failed tool results should parse correctly."""
        result = parse_tool_result(exit_code=1, stdout="", stderr="Error: file not found")
        assert result["exit_code"] == 1
