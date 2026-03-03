"""Tests for the LLM client abstraction (codex exec subprocess)."""

import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from metascaffold.llm_client import LLMClient, LLMResponse


class TestLLMResponse:
    def test_llm_response_dataclass(self):
        """LLMResponse should hold content, model, and usage info."""
        resp = LLMResponse(content="hello", model="gpt-5.3-codex", prompt_tokens=10, completion_tokens=5)
        assert resp.content == "hello"
        assert resp.model == "gpt-5.3-codex"
        assert resp.total_tokens == 15

    def test_llm_response_defaults(self):
        """LLMResponse should have sensible defaults."""
        resp = LLMResponse()
        assert resp.content == ""
        assert resp.error == ""
        assert resp.total_tokens == 0


class TestLLMClient:
    def test_client_enabled_when_codex_found(self):
        """Client should be enabled when codex binary is provided."""
        client = LLMClient(codex_path="/usr/bin/codex")
        assert client.enabled is True

    def test_client_disabled_when_codex_not_found(self):
        """Client should disable when codex binary not found."""
        client = LLMClient(codex_path="")
        assert client.enabled is False

    async def test_complete_returns_empty_when_disabled(self):
        """complete() should return empty response when client is disabled."""
        client = LLMClient(codex_path="")
        result = await client.complete(
            model="gpt-5.3-codex",
            system_prompt="test",
            user_prompt="test",
        )
        assert result.content == ""
        assert "disabled" in result.error.lower()

    @patch("metascaffold.llm_client.asyncio.create_subprocess_exec")
    async def test_complete_returns_parsed_response(self, mock_exec):
        """complete() should parse codex exec output correctly."""
        codex_output = (
            "OpenAI Codex v0.107.0\n"
            "--------\n"
            "workdir: /tmp\n"
            "model: gpt-5.3-codex\n"
            "--------\n"
            "user\n"
            "some prompt\n"
            "\n"
            "thinking\n"
            "**Analyzing**\n"
            "codex\n"
            '{"verdict": "pass", "confidence": 0.9}\n'
            "tokens used\n"
            "1234\n"
            '{"verdict": "pass", "confidence": 0.9}\n'
        )
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(codex_output.encode(), b""))
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        client = LLMClient(codex_path="/usr/bin/codex")
        result = await client.complete(
            model="gpt-5.3-codex",
            system_prompt="You are a judge.",
            user_prompt="Evaluate this.",
        )
        assert result.content == '{"verdict": "pass", "confidence": 0.9}'
        assert result.error == ""
        assert result.model == "gpt-5.3-codex"

    @patch("metascaffold.llm_client.asyncio.create_subprocess_exec")
    async def test_complete_handles_codex_error(self, mock_exec):
        """complete() should return error when codex exec fails."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"ERROR: model not supported"))
        mock_proc.returncode = 1
        mock_exec.return_value = mock_proc

        client = LLMClient(codex_path="/usr/bin/codex")
        result = await client.complete(
            model="gpt-5.3-codex",
            system_prompt="test",
            user_prompt="test",
        )
        assert result.content == ""
        assert "failed" in result.error.lower()

    @patch("metascaffold.llm_client.asyncio.create_subprocess_exec")
    async def test_complete_handles_timeout(self, mock_exec):
        """complete() should handle timeout gracefully."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_exec.return_value = mock_proc

        client = LLMClient(codex_path="/usr/bin/codex")
        result = await client.complete(
            model="gpt-5.3-codex",
            system_prompt="test",
            user_prompt="test",
        )
        assert result.content == ""
        assert "timed out" in result.error.lower()


class TestParseCodexOutput:
    def test_parse_standard_output(self):
        """Should extract content between 'codex' marker and 'tokens used'."""
        raw = "header\ncodex\nHello World\ntokens used\n100\nHello World"
        assert LLMClient._parse_codex_output(raw) == "Hello World"

    def test_parse_multiline_response(self):
        """Should handle multiline responses."""
        raw = "header\ncodex\nline1\nline2\nline3\ntokens used\n100"
        assert LLMClient._parse_codex_output(raw) == "line1\nline2\nline3"

    def test_parse_no_codex_marker(self):
        """Without codex marker, return raw output."""
        raw = "just some output"
        assert LLMClient._parse_codex_output(raw) == "just some output"

    def test_parse_json_response(self):
        """Should handle JSON responses correctly."""
        raw = 'header\ncodex\n{"verdict": "pass"}\ntokens used\n50'
        assert LLMClient._parse_codex_output(raw) == '{"verdict": "pass"}'
