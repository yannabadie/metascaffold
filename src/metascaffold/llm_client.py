"""LLM Client — abstraction over OpenAI API using Codex CLI OAuth tokens.

Reads the access_token from ~/.codex/auth.json (created by `codex login`).
Provides async completion with graceful degradation when LLM is unavailable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

logger = logging.getLogger("metascaffold.llm")

_DEFAULT_AUTH_PATH = Path.home() / ".codex" / "auth.json"


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
    """Async OpenAI API client using Codex CLI OAuth tokens."""

    def __init__(self, auth_path: Path | None = None):
        self._auth_path = auth_path or _DEFAULT_AUTH_PATH
        self._token: str = ""
        self.enabled: bool = False
        self._load_token()

    def _load_token(self) -> None:
        """Load OAuth access_token from Codex auth.json."""
        try:
            data = json.loads(self._auth_path.read_text())
            self._token = data.get("tokens", {}).get("access_token", "")
            self.enabled = bool(self._token)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            self.enabled = False

    async def complete(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """Call OpenAI Chat Completions API.

        Returns LLMResponse with content on success, or empty content + error on failure.
        """
        if not self.enabled:
            return LLMResponse(error="LLM client disabled (no auth token)")

        try:
            client = AsyncOpenAI(api_key=self._token)
            kwargs: dict = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format

            resp = await client.chat.completions.create(**kwargs)

            return LLMResponse(
                content=resp.choices[0].message.content or "",
                model=resp.model,
                prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
            )
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return LLMResponse(error=str(e))
