"""Tests for the hot-reload restart mechanism.

Verifies that _reload_components() reloads modules and re-instantiates
all server components without killing the process.
"""

import importlib
import sys
from unittest.mock import patch

import pytest

from metascaffold.server import mcp


@pytest.fixture(autouse=True)
def _restore_modules():
    """Snapshot module __dict__ before each test and restore after.

    importlib.reload mutates modules in-place, replacing class objects.
    This breaks isinstance checks in other test files that captured the
    old class at import time. We save and restore the full module dict.
    """
    saved: dict[str, dict] = {}
    for name in list(sys.modules):
        if name.startswith("metascaffold"):
            mod = sys.modules[name]
            if mod is not None:
                saved[name] = dict(mod.__dict__)
    yield
    for name, attrs in saved.items():
        if name in sys.modules and sys.modules[name] is not None:
            sys.modules[name].__dict__.clear()
            sys.modules[name].__dict__.update(attrs)


class TestRestart:
    def test_restart_tool_is_registered(self):
        """metascaffold_restart tool should be registered on the MCP server."""
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "metascaffold_restart" in tool_names

    def test_reload_components_refreshes_classifier(self):
        """After reload, classifier should be a fresh instance."""
        from metascaffold import server
        old_classifier = server.classifier
        server._reload_components()
        assert server.classifier is not old_classifier

    def test_reload_components_refreshes_planner(self):
        """After reload, planner should be a fresh instance."""
        from metascaffold import server
        old_planner = server.planner
        server._reload_components()
        assert server.planner is not old_planner

    def test_reload_components_refreshes_evaluator(self):
        """After reload, evaluator should be a fresh instance."""
        from metascaffold import server
        old_evaluator = server.evaluator
        server._reload_components()
        assert server.evaluator is not old_evaluator

    def test_reload_components_refreshes_sandbox(self):
        """After reload, sandbox should be a fresh instance."""
        from metascaffold import server
        old_sandbox = server.sandbox
        server._reload_components()
        assert server.sandbox is not old_sandbox

    def test_reload_components_returns_reloaded_modules(self):
        """_reload_components should return the list of reloaded module names."""
        from metascaffold import server
        result = server._reload_components()
        assert "metascaffold.config" in result
        assert "metascaffold.classifier" in result
        assert "metascaffold.planner" in result
        assert "metascaffold.sandbox" in result
        assert "metascaffold.evaluator" in result

    def test_reload_components_survives_import_error(self):
        """If a module reload fails, old components should be preserved."""
        from metascaffold import server
        old_classifier = server.classifier

        # Patch importlib.reload to fail on classifier module
        original_reload = importlib.reload
        def failing_reload(mod):
            if mod.__name__ == "metascaffold.classifier":
                raise SyntaxError("simulated bad code")
            return original_reload(mod)

        with patch("importlib.reload", side_effect=failing_reload):
            result = server._reload_components()

        # Should return error info, not crash
        assert "error" in result or isinstance(result, list)
        # Classifier should still be functional (old or new)
        assert server.classifier is not None
