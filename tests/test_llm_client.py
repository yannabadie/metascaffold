"""Tests for the LLM client abstraction."""

import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from metascaffold.llm_client import LLMClient, LLMResponse


class TestLLMClient:
    def test_llm_response_dataclass(self):
        """LLMResponse should hold content, model, and usage info."""
        resp = LLMResponse(content="hello", model="gpt-4.1-nano", prompt_tokens=10, completion_tokens=5)
        assert resp.content == "hello"
        assert resp.model == "gpt-4.1-nano"
        assert resp.total_tokens == 15

    def test_client_loads_token_from_codex_auth(self, tmp_path):
        """Client should read OAuth token from ~/.codex/auth.json."""
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({
            "tokens": {"access_token": "test-token-123"}
        }))
        client = LLMClient(auth_path=auth_file)
        assert client._token == "test-token-123"

    def test_client_disabled_when_no_auth(self, tmp_path):
        """Client should gracefully disable when auth file is missing."""
        client = LLMClient(auth_path=tmp_path / "nonexistent.json")
        assert client.enabled is False

    @patch("metascaffold.llm_client.AsyncOpenAI")
    async def test_complete_returns_llm_response(self, mock_openai_cls, tmp_path):
        """complete() should return structured LLMResponse."""
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({"tokens": {"access_token": "tok"}}))

        mock_msg = MagicMock()
        mock_msg.content = '{"verdict": "pass"}'
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 50
        mock_usage.completion_tokens = 20
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = mock_usage
        mock_resp.model = "gpt-4.1-nano"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_resp)
        mock_openai_cls.return_value = mock_client

        client = LLMClient(auth_path=auth_file)
        result = await client.complete(
            model="gpt-4.1-nano",
            system_prompt="You are an evaluator.",
            user_prompt="Evaluate this code.",
        )
        assert result.content == '{"verdict": "pass"}'
        assert result.total_tokens == 70

    @patch("metascaffold.llm_client.AsyncOpenAI")
    async def test_complete_fallback_on_error(self, mock_openai_cls, tmp_path):
        """complete() should return empty LLMResponse on API error, not raise."""
        auth_file = tmp_path / "auth.json"
        auth_file.write_text(json.dumps({"tokens": {"access_token": "tok"}}))

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API down"))
        mock_openai_cls.return_value = mock_client

        client = LLMClient(auth_path=auth_file)
        result = await client.complete(
            model="gpt-4.1-nano",
            system_prompt="test",
            user_prompt="test",
        )
        assert result.content == ""
        assert result.error == "API down"

    async def test_complete_returns_empty_when_disabled(self, tmp_path):
        """complete() should return empty response when client is disabled."""
        client = LLMClient(auth_path=tmp_path / "nonexistent.json")
        result = await client.complete(
            model="gpt-4.1-nano",
            system_prompt="test",
            user_prompt="test",
        )
        assert result.content == ""
        assert "disabled" in result.error.lower()
