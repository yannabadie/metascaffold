"""Tests for the NotebookLM bridge module.

These tests mock notebooklm-py to avoid requiring auth.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metascaffold.notebooklm_bridge import NotebookLMBridge, BridgeResult


class TestNotebookLMBridge:
    def test_bridge_creation_with_defaults(self):
        """Bridge should initialize with default config."""
        bridge = NotebookLMBridge(enabled=True, default_notebook="Test")
        assert bridge.enabled is True
        assert bridge.default_notebook == "Test"

    def test_bridge_disabled_returns_empty(self):
        """When disabled, all operations return empty BridgeResult."""
        bridge = NotebookLMBridge(enabled=False, default_notebook="Test")
        result = bridge.query_sync("What is metacognition?")
        assert result.success is False
        assert result.content == ""
        assert "disabled" in result.reason.lower()

    @patch("metascaffold.notebooklm_bridge._get_client")
    def test_query_returns_content_on_success(self, mock_get_client):
        """Successful query should return content from NotebookLM."""
        mock_client = MagicMock()
        mock_client.list_notebooks.return_value = [
            MagicMock(title="Test", id="test-id"),
        ]
        mock_client.chat.return_value = MagicMock(text="Metacognition is thinking about thinking.")
        mock_get_client.return_value = mock_client

        bridge = NotebookLMBridge(enabled=True, default_notebook="Test")
        result = bridge.query_sync("What is metacognition?")
        assert result.success is True
        assert "metacognition" in result.content.lower()

    @patch("metascaffold.notebooklm_bridge._get_client")
    def test_query_graceful_degradation_on_error(self, mock_get_client):
        """Errors should return empty result, not raise exceptions."""
        mock_get_client.side_effect = Exception("Auth expired")

        bridge = NotebookLMBridge(
            enabled=True,
            default_notebook="Test",
            fallback_on_error=True,
        )
        result = bridge.query_sync("test")
        assert result.success is False
        assert "Auth expired" in result.reason

    @patch("metascaffold.notebooklm_bridge._get_client")
    def test_query_raises_when_fallback_disabled(self, mock_get_client):
        """With fallback_on_error=False, errors should propagate."""
        mock_get_client.side_effect = Exception("Auth expired")

        bridge = NotebookLMBridge(
            enabled=True,
            default_notebook="Test",
            fallback_on_error=False,
        )
        with pytest.raises(Exception, match="Auth expired"):
            bridge.query_sync("test")
