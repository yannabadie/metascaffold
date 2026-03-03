"""NotebookLM Bridge — interface to notebooklm-py for knowledge-enriched reflection.

Provides graceful degradation: if NotebookLM is unavailable, returns empty
results and the system continues without external knowledge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BridgeResult:
    """Result from a NotebookLM operation."""

    success: bool
    content: str = ""
    reason: str = ""


def _get_client():
    """Lazy-import and return a NotebookLM client instance.

    This avoids import errors if notebooklm-py is not installed or not authenticated.
    """
    from notebooklm import NotebookLM

    return NotebookLM()


class NotebookLMBridge:
    """Bridge between MetaScaffold and NotebookLM via notebooklm-py."""

    def __init__(
        self,
        enabled: bool = True,
        default_notebook: str = "MetaScaffold_Core",
        fallback_on_error: bool = True,
    ):
        self.enabled = enabled
        self.default_notebook = default_notebook
        self.fallback_on_error = fallback_on_error

    def query_sync(self, question: str, notebook: str | None = None) -> BridgeResult:
        """Query a NotebookLM notebook synchronously.

        Args:
            question: The question to ask
            notebook: Notebook name (uses default if not specified)

        Returns:
            BridgeResult with success status and content
        """
        if not self.enabled:
            return BridgeResult(success=False, reason="NotebookLM bridge is disabled")

        target = notebook or self.default_notebook
        try:
            client = _get_client()
            # Find the notebook by name
            notebooks = client.list_notebooks()
            matching = [nb for nb in notebooks if nb.title == target]
            if not matching:
                return BridgeResult(
                    success=False,
                    reason=f"Notebook '{target}' not found",
                )
            response = client.chat(notebook_id=matching[0].id, message=question)
            return BridgeResult(success=True, content=response.text)

        except Exception as e:
            logger.warning("NotebookLM query failed: %s", e)
            if self.fallback_on_error:
                return BridgeResult(success=False, reason=str(e))
            raise

    def upload_source(self, url: str, notebook: str | None = None) -> BridgeResult:
        """Upload a URL source to a NotebookLM notebook."""
        if not self.enabled:
            return BridgeResult(success=False, reason="NotebookLM bridge is disabled")

        target = notebook or self.default_notebook
        try:
            client = _get_client()
            notebooks = client.list_notebooks()
            matching = [nb for nb in notebooks if nb.title == target]
            if not matching:
                return BridgeResult(
                    success=False, reason=f"Notebook '{target}' not found"
                )
            client.add_source(notebook_id=matching[0].id, url=url)
            return BridgeResult(success=True, content=f"Source uploaded: {url}")
        except Exception as e:
            logger.warning("NotebookLM upload failed: %s", e)
            if self.fallback_on_error:
                return BridgeResult(success=False, reason=str(e))
            raise

    def create_notebook(self, title: str) -> BridgeResult:
        """Create a new NotebookLM notebook."""
        if not self.enabled:
            return BridgeResult(success=False, reason="NotebookLM bridge is disabled")

        try:
            client = _get_client()
            nb = client.create_notebook(title=title)
            return BridgeResult(success=True, content=f"Notebook created: {nb.id}")
        except Exception as e:
            logger.warning("NotebookLM create failed: %s", e)
            if self.fallback_on_error:
                return BridgeResult(success=False, reason=str(e))
            raise
