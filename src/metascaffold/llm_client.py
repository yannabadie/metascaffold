"""LLM Client — abstraction over Codex CLI for LLM inference.

Uses `codex exec` subprocess to call OpenAI models via the user's ChatGPT
subscription. No API key needed — authentication is handled by Codex CLI.

Corporate SSL is transparent since Codex handles its own TLS.
Provides async completion with graceful degradation when Codex is unavailable.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass

logger = logging.getLogger("metascaffold.llm")


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""
    content: str = ""
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""

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
        """
        if not self.enabled:
            return LLMResponse(error="LLM client disabled (codex not found)")

        prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}\n\nRespond with ONLY the requested output. No explanations, no tool calls, no file edits."

        try:
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

            # Parse codex exec output: the actual response is after the header block
            # Codex outputs headers, then "codex\n<response>\ntokens used\n<count>"
            content = self._parse_codex_output(output)

            return LLMResponse(
                content=content,
                model="gpt-5.3-codex",
            )
        except asyncio.TimeoutError:
            return LLMResponse(error="codex exec timed out after 120s")
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return LLMResponse(error=str(e))

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
