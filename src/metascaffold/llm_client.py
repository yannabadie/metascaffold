"""LLM Client — abstraction over Codex CLI for LLM inference.

Uses `codex exec` subprocess to call OpenAI models via the user's ChatGPT
subscription. No API key needed — authentication is handled by Codex CLI.

When a JSON schema is provided (response_format), uses --output-schema and -o
flags to get reliable structured JSON output. Otherwise, parses raw stdout.

Corporate SSL is transparent since Codex handles its own TLS.
Provides async completion with graceful degradation when Codex is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("metascaffold.llm")


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""
    content: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""
    token_logprobs: list[dict] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """Async LLM client using Codex CLI subprocess."""

    def __init__(self, codex_path: str | None = None):
        if codex_path is not None:
            self._codex = codex_path
        else:
            self._codex = shutil.which("codex") or ""
        self.enabled: bool = bool(self._codex)

    async def complete(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Call LLM via codex exec subprocess.

        Returns LLMResponse with content on success, or empty content + error on failure.
        The model parameter is ignored — Codex uses the model from its config.

        When response_format is a JSON Schema dict, uses --output-schema for
        reliable structured output. The schema MUST include "additionalProperties": false
        at the top level.
        """
        if not self.enabled:
            return LLMResponse(error="LLM client disabled (codex not found)")

        prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}\n\nRespond with ONLY the requested output. No explanations, no tool calls, no file edits."

        try:
            if response_format:
                return await self._complete_with_schema(prompt, response_format)
            return await self._complete_raw(prompt)
        except asyncio.TimeoutError:
            return LLMResponse(error="codex exec timed out after 120s")
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return LLMResponse(error=str(e))

    async def _complete_with_schema(
        self, prompt: str, schema: dict,
    ) -> LLMResponse:
        """Use --output-schema and -o for reliable structured JSON output."""
        # Ensure schema has additionalProperties: false (required by API)
        if "additionalProperties" not in schema:
            schema = {**schema, "additionalProperties": False}

        with tempfile.TemporaryDirectory() as tmpdir:
            schema_path = Path(tmpdir) / "schema.json"
            output_path = Path(tmpdir) / "result.json"
            schema_path.write_text(json.dumps(schema))

            proc = await asyncio.create_subprocess_exec(
                self._codex, "exec",
                "-c", "sandbox_permissions=[]",
                "--output-schema", str(schema_path),
                "-o", str(output_path),
                prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=120.0,
            )

            if proc.returncode != 0:
                err_text = stderr.decode("utf-8", errors="replace").strip()
                return LLMResponse(error=f"codex exec failed (exit {proc.returncode}): {err_text[:500]}")

            # Read clean JSON from output file
            if output_path.exists():
                content = output_path.read_text(encoding="utf-8").strip()
                if content:
                    return LLMResponse(content=content, model="gpt-5.3-codex")

            return LLMResponse(error="codex exec produced no output")

    async def _complete_raw(self, prompt: str) -> LLMResponse:
        """Fallback: parse raw stdout (for non-schema calls)."""
        proc = await asyncio.create_subprocess_exec(
            self._codex, "exec",
            "-c", "sandbox_permissions=[]",
            prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=120.0,
        )
        output = stdout.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            err_text = stderr.decode("utf-8", errors="replace").strip()
            return LLMResponse(error=f"codex exec failed (exit {proc.returncode}): {err_text[:500]}")

        content = self._parse_codex_output(output)
        return LLMResponse(content=content, model="gpt-5.3-codex")

    async def complete_with_logprobs(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 64,
        top_logprobs: int = 5,
    ) -> LLMResponse:
        """Call OpenAI API directly to get token logprobs.

        This does NOT use codex exec. It requires the OPEN_API_KEY env var.
        Used by the Classifier for entropy-based System 1/System 2 routing.
        """
        api_key = os.environ.get("OPEN_API_KEY", "")
        if not api_key:
            return LLMResponse(
                error="OPEN_API_KEY not set — cannot call OpenAI API for logprobs"
            )

        try:
            return self._call_openai_with_logprobs(
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_logprobs=top_logprobs,
            )
        except Exception as e:
            logger.warning("OpenAI logprobs call failed: %s", e)
            return LLMResponse(error=str(e))

    def _call_openai_with_logprobs(
        self,
        api_key: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        top_logprobs: int,
    ) -> LLMResponse:
        """Synchronous helper that creates the OpenAI client and calls the API."""
        # Inject corporate SSL certs (non-critical — catch and continue)
        try:
            import truststore
            truststore.inject_into_ssl()
        except Exception:
            pass

        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_logprobs,
        )

        choice = response.choices[0]
        content = choice.message.content or ""

        # Extract token logprobs
        token_logprobs: list[dict] = []
        if choice.logprobs and choice.logprobs.content:
            for token_info in choice.logprobs.content:
                entry: dict = {
                    "token": token_info.token,
                    "logprob": token_info.logprob,
                    "top_logprobs": [
                        {"token": tp.token, "logprob": tp.logprob}
                        for tp in (token_info.top_logprobs or [])
                    ],
                }
                token_logprobs.append(entry)

        usage = response.usage
        return LLMResponse(
            content=content,
            model=response.model or model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            token_logprobs=token_logprobs,
        )

    @staticmethod
    def _parse_codex_output(raw: str) -> str:
        """Extract the model response from codex exec output.

        Codex exec outputs:
          ... header lines ...
          codex
          <actual response>
          tokens used
          <number>
          <final repeated text or empty>

        We extract the text between the last 'codex' marker and 'tokens used'.
        """
        lines = raw.split("\n")

        # Find last "codex" line (marks start of response)
        codex_idx = -1
        for i, line in enumerate(lines):
            if line.strip() == "codex":
                codex_idx = i

        if codex_idx == -1:
            # No codex marker — return entire output as best effort
            return raw

        # Find "tokens used" after the codex marker
        tokens_idx = len(lines)
        for i in range(codex_idx + 1, len(lines)):
            if lines[i].strip() == "tokens used":
                tokens_idx = i
                break

        # Extract content between markers
        content_lines = lines[codex_idx + 1:tokens_idx]
        return "\n".join(content_lines).strip()
