"""NotebookLM Bridge — interface to notebooklm-py for knowledge-enriched reflection.

Provides graceful degradation: if NotebookLM is unavailable, returns empty
results and the system continues without external knowledge.

The real notebooklm-py API is fully async. This bridge exposes async methods
natively — the MCP server tools are also async, so no event loop conflict.
Corporate SSL is handled via truststore (OS cert store injection).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _inject_truststore() -> None:
    """Inject OS certificate store into Python SSL for corporate proxies."""
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


@dataclass
class BridgeResult:
    """Result from a NotebookLM operation."""
    success: bool
    content: str = ""
    reason: str = ""


async def _get_client():
    """Create and return an authenticated NotebookLMClient.

    Uses saved browser cookies from `notebooklm login`.
    """
    _inject_truststore()
    from notebooklm import NotebookLMClient
    client = await NotebookLMClient.from_storage(timeout=120.0)
    return client


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

    async def query(self, question: str, notebook: str | None = None) -> BridgeResult:
        """Query a NotebookLM notebook asynchronously."""
        if not self.enabled:
            return BridgeResult(success=False, reason="NotebookLM bridge is disabled")

        try:
            target = notebook or self.default_notebook
            client = await _get_client()
            async with client:
                notebooks = await client.notebooks.list()
                matching = [nb for nb in notebooks if nb.title == target]
                if not matching:
                    return BridgeResult(
                        success=False,
                        reason=f"Notebook '{target}' not found",
                    )
                result = await client.chat.ask(matching[0].id, question)
                return BridgeResult(success=True, content=result.text)
        except Exception as e:
            logger.warning("NotebookLM query failed: %s", e)
            if self.fallback_on_error:
                return BridgeResult(success=False, reason=str(e))
            raise

    async def upload_source(self, url: str, notebook: str | None = None) -> BridgeResult:
        """Upload a URL source to a NotebookLM notebook."""
        if not self.enabled:
            return BridgeResult(success=False, reason="NotebookLM bridge is disabled")

        try:
            target = notebook or self.default_notebook
            client = await _get_client()
            async with client:
                notebooks = await client.notebooks.list()
                matching = [nb for nb in notebooks if nb.title == target]
                if not matching:
                    return BridgeResult(success=False, reason=f"Notebook '{target}' not found")
                await client.sources.add_url(matching[0].id, url)
                return BridgeResult(success=True, content=f"Source uploaded: {url}")
        except Exception as e:
            logger.warning("NotebookLM upload failed: %s", e)
            if self.fallback_on_error:
                return BridgeResult(success=False, reason=str(e))
            raise

    async def create_notebook(self, title: str) -> BridgeResult:
        """Create a new NotebookLM notebook."""
        if not self.enabled:
            return BridgeResult(success=False, reason="NotebookLM bridge is disabled")

        try:
            client = await _get_client()
            async with client:
                nb = await client.notebooks.create(title=title)
                return BridgeResult(success=True, content=f"Notebook created: {nb.id}")
        except Exception as e:
            logger.warning("NotebookLM create failed: %s", e)
            if self.fallback_on_error:
                return BridgeResult(success=False, reason=str(e))
            raise
