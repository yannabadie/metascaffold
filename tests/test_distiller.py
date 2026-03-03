"""Tests for the Task Distiller component."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from metascaffold.distiller import Distiller, TaskTemplate


class TestDistiller:
    def test_task_template_dataclass(self):
        """TaskTemplate should hold structured task information."""
        t = TaskTemplate(
            objective="Add user authentication",
            constraints=["Must use JWT", "No external auth service"],
            target_files=["src/auth.py", "tests/test_auth.py"],
            variables={"framework": "FastAPI"},
        )
        assert t.objective == "Add user authentication"
        assert len(t.constraints) == 2
        assert len(t.target_files) == 2
        assert t.variables["framework"] == "FastAPI"

    def test_task_template_to_dict(self):
        """TaskTemplate should serialize to dict."""
        t = TaskTemplate(objective="Test")
        d = t.to_dict()
        assert isinstance(d, dict)
        assert d["objective"] == "Test"
        assert d["constraints"] == []

    async def test_distill_produces_template(self):
        """Distiller should produce a structured TaskTemplate from raw task text."""
        mock_client = AsyncMock()
        mock_client.enabled = True
        mock_client.complete = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "objective": "Add JWT-based authentication to the FastAPI application",
                "constraints": ["Use python-jose for JWT", "Support token refresh"],
                "target_files": ["src/api/auth.py", "src/api/middleware.py"],
                "variables": [{"key": "token_expiry", "value": "30m"}, {"key": "algorithm", "value": "HS256"}],
            }),
            error=""
        ))

        distiller = Distiller(llm_client=mock_client)
        template = await distiller.distill(
            task="Add JWT auth to the API",
            context="FastAPI app with SQLAlchemy",
        )
        assert template.objective != ""
        assert len(template.target_files) > 0
        assert len(template.constraints) > 0
        mock_client.complete.assert_awaited_once()

    async def test_fallback_produces_basic_template(self):
        """When LLM is disabled, return a basic template from the raw input."""
        mock_client = AsyncMock()
        mock_client.enabled = False

        distiller = Distiller(llm_client=mock_client)
        template = await distiller.distill(
            task="Fix the login bug",
            context="src/auth.py has an issue",
        )
        assert template.objective == "Fix the login bug"
        assert template.target_files == []  # No LLM to infer files

    async def test_fallback_on_llm_error(self):
        """When LLM returns an error, fall back to basic template."""
        mock_client = AsyncMock()
        mock_client.enabled = True
        mock_client.complete = AsyncMock(return_value=MagicMock(
            content="",
            error="codex exec timed out"
        ))

        distiller = Distiller(llm_client=mock_client)
        template = await distiller.distill(task="Do something", context="")
        assert template.objective == "Do something"
        assert template.target_files == []
