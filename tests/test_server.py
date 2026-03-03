"""Tests for the MCP server tool registration.

Verifies that all MetaScaffold tools are correctly registered on the FastMCP instance.
"""

from metascaffold.server import mcp


class TestMCPServerRegistration:
    def test_server_has_classify_tool(self):
        """metascaffold_classify tool should be registered."""
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "metascaffold_classify" in tool_names

    def test_server_has_plan_tool(self):
        """metascaffold_plan tool should be registered."""
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "metascaffold_plan" in tool_names

    def test_server_has_sandbox_exec_tool(self):
        """metascaffold_sandbox_exec tool should be registered."""
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "metascaffold_sandbox_exec" in tool_names

    def test_server_has_evaluate_tool(self):
        """metascaffold_evaluate tool should be registered."""
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "metascaffold_evaluate" in tool_names

    def test_server_has_nlm_query_tool(self):
        """metascaffold_nlm_query tool should be registered."""
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "metascaffold_nlm_query" in tool_names

    def test_server_has_telemetry_query_tool(self):
        """metascaffold_telemetry_query tool should be registered."""
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "metascaffold_telemetry_query" in tool_names

    def test_server_has_distill_tool(self):
        """metascaffold_distill tool should be registered."""
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "metascaffold_distill" in tool_names

    def test_server_has_reflect_tool(self):
        """metascaffold_reflect tool should be registered."""
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "metascaffold_reflect" in tool_names
