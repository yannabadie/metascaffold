"""Tests for the LLM client abstraction (codex exec subprocess)."""

import asyncio
import json
from pathlib import Path
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


class TestCompleteWithSchema:
    """Tests for structured JSON output via --output-schema."""

    @patch("metascaffold.llm_client.asyncio.create_subprocess_exec")
    async def test_schema_path_returns_json_from_output_file(self, mock_exec):
        """complete() with response_format should use --output-schema and read from -o file."""
        expected_json = '{"routing": "system2", "confidence": 0.95, "reasoning": "test"}'

        async def fake_communicate():
            # Simulate codex writing the output file
            args = mock_exec.call_args[0]
            for i, a in enumerate(args):
                if str(a) == "-o" and i + 1 < len(args):
                    Path(args[i + 1]).write_text(expected_json)
            return (b"", b"")

        mock_proc = AsyncMock()
        mock_proc.communicate = fake_communicate
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        schema = {
            "type": "object",
            "properties": {
                "routing": {"type": "string"},
                "confidence": {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": ["routing", "confidence", "reasoning"],
            "additionalProperties": False,
        }

        client = LLMClient(codex_path="/usr/bin/codex")
        result = await client.complete(
            model="test",
            system_prompt="sys",
            user_prompt="usr",
            response_format=schema,
        )

        assert result.content == expected_json
        assert result.error == ""
        assert result.model == "gpt-5.3-codex"

        # Verify --output-schema and -o flags were passed
        call_args = mock_exec.call_args[0]
        assert "--output-schema" in call_args
        assert "-o" in call_args

    @patch("metascaffold.llm_client.asyncio.create_subprocess_exec")
    async def test_schema_path_adds_additional_properties(self, mock_exec):
        """Schema without additionalProperties should get it auto-added."""
        async def fake_communicate():
            args = mock_exec.call_args[0]
            for i, a in enumerate(args):
                if str(a) == "--output-schema" and i + 1 < len(args):
                    written_schema = json.loads(Path(args[i + 1]).read_text())
                    assert written_schema["additionalProperties"] is False
                if str(a) == "-o" and i + 1 < len(args):
                    Path(args[i + 1]).write_text('{"result": "ok"}')
            return (b"", b"")

        mock_proc = AsyncMock()
        mock_proc.communicate = fake_communicate
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        # Schema WITHOUT additionalProperties
        schema = {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "required": ["result"],
        }

        client = LLMClient(codex_path="/usr/bin/codex")
        result = await client.complete(
            model="test", system_prompt="s", user_prompt="u",
            response_format=schema,
        )
        assert result.content == '{"result": "ok"}'

    @patch("metascaffold.llm_client.asyncio.create_subprocess_exec")
    async def test_schema_path_handles_no_output_file(self, mock_exec):
        """When codex produces no output file, return error."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": False,
        }

        client = LLMClient(codex_path="/usr/bin/codex")
        result = await client.complete(
            model="test", system_prompt="s", user_prompt="u",
            response_format=schema,
        )
        assert result.content == ""
        assert "no output" in result.error.lower()

    @patch("metascaffold.llm_client.asyncio.create_subprocess_exec")
    async def test_schema_path_handles_subprocess_error(self, mock_exec):
        """When codex exec fails with non-zero exit, return error."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"schema validation error"))
        mock_proc.returncode = 1
        mock_exec.return_value = mock_proc

        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
            "additionalProperties": False,
        }

        client = LLMClient(codex_path="/usr/bin/codex")
        result = await client.complete(
            model="test", system_prompt="s", user_prompt="u",
            response_format=schema,
        )
        assert result.content == ""
        assert "failed" in result.error.lower()
        assert "schema validation" in result.error.lower()


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
