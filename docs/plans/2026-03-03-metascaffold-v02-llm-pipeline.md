# MetaScaffold v0.2 — LLM-Powered Cognitive Pipeline

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace all regex heuristics with real LLM calls (via OpenAI API + Codex subscription) and orchestrate them in a 6-stage cognitive pipeline: Classify → Distill → Plan → Execute → Evaluate → Reflect.

**Architecture:** Each cognitive component (Classifier, Evaluator, Planner) gets an LLM backend that performs semantic analysis instead of pattern matching. A new `llm_client.py` module abstracts OpenAI API calls using the OAuth token from Codex CLI (`~/.codex/auth.json`). Two new components (Distiller, Reflector) are added. A `pipeline.py` orchestrator manages the full cognitive flow with System 1 bypass and retry/backtrack loops.

**Tech Stack:** Python 3.11+, OpenAI Python SDK (`openai`), FastMCP, existing MetaScaffold components, Codex CLI OAuth tokens.

**Models (configurable in config.toml):**
- Classifier: `gpt-4.1-nano` (ultra-fast, ~5ms for simple routing)
- Evaluator: `o3-mini` (strong reasoning for LLM-as-judge on code)
- Planner: `gpt-4.1-mini` (fast, good at structured output, 1M context)
- Distiller: `gpt-4.1-nano` (fast task structuring)
- Reflector: `o3-mini` (deep pattern analysis from telemetry)

**Fallback:** When LLM is unavailable (no token, network error, timeout), all components fall back to v0.1 heuristic behavior. The system never hard-fails on LLM unavailability.

---

## Task 1: Add `openai` dependency and create `llm_client.py` scaffold

**Files:**
- Modify: `pyproject.toml:6-11` (add openai dependency)
- Create: `src/metascaffold/llm_client.py`
- Create: `tests/test_llm_client.py`

**Step 1: Write the failing test**

```python
# tests/test_llm_client.py
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'metascaffold.llm_client'`

**Step 3: Add openai dependency**

In `pyproject.toml`, add `"openai>=1.60.0"` to dependencies:

```toml
dependencies = [
    "mcp[cli]>=1.2.0",
    "pydantic>=2.0",
    "notebooklm-py[browser]>=0.3.0",
    "truststore>=0.10.4",
    "openai>=1.60.0",
]
```

Then run: `uv sync`

**Step 4: Write minimal implementation**

```python
# src/metascaffold/llm_client.py
"""LLM Client — abstraction over OpenAI API using Codex CLI OAuth tokens.

Reads the access_token from ~/.codex/auth.json (created by `codex login`).
Provides async completion with graceful degradation when LLM is unavailable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

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
            from openai import AsyncOpenAI

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
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_llm_client.py -v`
Expected: 6 PASS

**Step 6: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass (60 existing + 6 new = 66)

**Step 7: Commit**

```bash
git add pyproject.toml src/metascaffold/llm_client.py tests/test_llm_client.py
git commit -m "feat(v0.2): add LLM client with OpenAI SDK + Codex OAuth tokens"
```

---

## Task 2: Add LLM model config to `config.py` and `default_config.toml`

**Files:**
- Modify: `config/default_config.toml:1-25`
- Modify: `src/metascaffold/config.py:18-58`
- Modify: `tests/test_config.py`

**Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_llm_config_defaults(self):
    """LLM config should have default models for each component."""
    cfg = load_config()
    assert cfg.llm.classifier_model == "gpt-4.1-nano"
    assert cfg.llm.evaluator_model == "o3-mini"
    assert cfg.llm.planner_model == "gpt-4.1-mini"
    assert cfg.llm.distiller_model == "gpt-4.1-nano"
    assert cfg.llm.reflector_model == "o3-mini"
    assert cfg.llm.enabled is True
    assert cfg.llm.fallback_to_heuristics is True
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::TestConfig::test_llm_config_defaults -v`
Expected: FAIL — `AttributeError: 'MetaScaffoldConfig' has no attribute 'llm'`

**Step 3: Write minimal implementation**

Add to `config/default_config.toml`:

```toml
[llm]
enabled = true
fallback_to_heuristics = true
classifier_model = "gpt-4.1-nano"
evaluator_model = "o3-mini"
planner_model = "gpt-4.1-mini"
distiller_model = "gpt-4.1-nano"
reflector_model = "o3-mini"
```

Add to `src/metascaffold/config.py` after `McpServerConfig`:

```python
@dataclass
class LLMConfig:
    enabled: bool = True
    fallback_to_heuristics: bool = True
    classifier_model: str = "gpt-4.1-nano"
    evaluator_model: str = "o3-mini"
    planner_model: str = "gpt-4.1-mini"
    distiller_model: str = "gpt-4.1-nano"
    reflector_model: str = "o3-mini"
```

Add `llm` field to `MetaScaffoldConfig`:

```python
@dataclass
class MetaScaffoldConfig:
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    notebooklm: NotebookLMConfig = field(default_factory=NotebookLMConfig)
    mcp_server: McpServerConfig = field(default_factory=McpServerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
```

Add LLM handling to `_dict_to_config()`:

```python
if "llm" in data:
    cfg.llm = LLMConfig(**data["llm"])
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: 5 PASS (4 existing + 1 new)

**Step 5: Commit**

```bash
git add config/default_config.toml src/metascaffold/config.py tests/test_config.py
git commit -m "feat(v0.2): add LLM model configuration for all cognitive components"
```

---

## Task 3: Spike — verify OpenAI API works with Codex OAuth token

**Files:**
- Create: `scripts/test_openai_auth.py` (temporary, not committed)

**Step 1: Write a quick validation script**

```python
# scripts/test_openai_auth.py
"""Spike: verify OpenAI API accepts Codex OAuth token."""
import asyncio
import json
from pathlib import Path

async def main():
    from openai import AsyncOpenAI

    auth = json.loads(Path.home().joinpath(".codex/auth.json").read_text())
    token = auth["tokens"]["access_token"]

    client = AsyncOpenAI(api_key=token)
    resp = await client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "Reply with exactly: AUTH_OK"}],
        max_tokens=10,
    )
    print(f"Model: {resp.model}")
    print(f"Response: {resp.choices[0].message.content}")
    print(f"Tokens: {resp.usage.prompt_tokens} + {resp.usage.completion_tokens}")

asyncio.run(main())
```

**Step 2: Run it**

Run: `uv run python scripts/test_openai_auth.py`
Expected: `AUTH_OK` response (confirms Codex OAuth token works with OpenAI API)

**Step 3: If the token doesn't work**

If we get an auth error, the fallback approach is to use `codex exec` as a subprocess:

```python
# Alternative: subprocess approach
import subprocess
result = subprocess.run(
    ["codex", "exec", "--model", "gpt-4.1-nano", "-c", 'sandbox_permissions=[]', "Reply with exactly: AUTH_OK"],
    capture_output=True, text=True, timeout=30,
)
```

Document which approach works and adjust `llm_client.py` accordingly.

**Step 4: Clean up**

Do NOT commit the spike script. Delete or gitignore it.

---

## Task 4: Classifier v2 — LLM-powered semantic classification

**Files:**
- Modify: `src/metascaffold/classifier.py`
- Modify: `tests/test_classifier.py`

**Step 1: Write the failing test for LLM classification**

Add to `tests/test_classifier.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch

class TestClassifierLLM:
    """Tests for LLM-enhanced classification."""

    @patch("metascaffold.classifier.LLMClient")
    async def test_llm_classifies_ambiguous_task(self, mock_llm_cls):
        """Ambiguous tasks should be classified via LLM, not just regex."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = '{"routing": "system2", "confidence": 0.35, "reasoning": "This task involves modifying authentication logic which has high blast radius"}'
        mock_response.error = ""
        mock_client.complete = AsyncMock(return_value=mock_response)
        mock_client.enabled = True

        classifier = Classifier(llm_client=mock_client)
        result = await classifier.classify_async(
            tool_name="Edit",
            tool_input={"file_path": "/src/auth.py", "old_string": "def login", "new_string": "def login_v2"},
            context="Refactoring authentication module",
        )
        assert result.routing == "system2"
        assert result.confidence < 0.5
        assert "authentication" in result.reasoning.lower() or "blast radius" in result.reasoning.lower()

    async def test_readonly_tools_skip_llm(self):
        """Read-only tools should still use fast-path, never call LLM."""
        mock_client = AsyncMock()
        mock_client.enabled = True
        mock_client.complete = AsyncMock()

        classifier = Classifier(llm_client=mock_client)
        result = await classifier.classify_async(
            tool_name="Read",
            tool_input={"file_path": "/src/main.py"},
            context="Reading a file",
        )
        assert result.routing == "system1"
        mock_client.complete.assert_not_called()

    async def test_fallback_to_heuristics_when_llm_disabled(self):
        """When LLM is disabled, fall back to v0.1 heuristic behavior."""
        mock_client = AsyncMock()
        mock_client.enabled = False

        classifier = Classifier(llm_client=mock_client)
        result = await classifier.classify_async(
            tool_name="Bash",
            tool_input={"command": "rm -rf /tmp/test"},
            context="Cleaning temp files",
        )
        # Should use heuristic: destructive command → low confidence → system2
        assert result.routing == "system2"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_classifier.py::TestClassifierLLM -v`
Expected: FAIL — `classify_async` doesn't exist, `llm_client` parameter not accepted

**Step 3: Implement Classifier v2**

Modify `src/metascaffold/classifier.py`:

```python
"""System 1/2 Classifier — routes tasks to fast or deliberate processing.

v0.2: Uses LLM for semantic classification of ambiguous tasks.
Falls back to heuristics when LLM is unavailable.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger("metascaffold.classifier")

_COMPLEX_KEYWORDS = re.compile(
    r"refactor|architect|redesign|migrate|across\s+\d+|entire|all\s+modules|system-wide",
    re.IGNORECASE,
)

_DESTRUCTIVE_COMMANDS = re.compile(
    r"rm\s+-rf|drop\s+table|delete|git\s+reset\s+--hard|git\s+push\s+--force|truncate|format",
    re.IGNORECASE,
)

_SIMPLE_COMMANDS = re.compile(
    r"^(ls|pwd|echo|cat|head|tail|wc|date|whoami|which|type|git\s+status|git\s+log|git\s+diff)(\s|$)",
    re.IGNORECASE,
)

_READ_ONLY_TOOLS = frozenset({"Read", "Grep", "Glob", "WebSearch", "WebFetch"})

_CLASSIFIER_SYSTEM_PROMPT = """You are a metacognitive classifier for a coding AI agent.
Given a tool call and its context, assess whether the task needs careful deliberation (System 2)
or can proceed quickly (System 1).

Evaluate these signals:
- **Complexity**: How many files/systems are affected? Is the logic intricate?
- **Reversibility**: Can this action be easily undone? (git revert vs data loss)
- **Uncertainty**: Is the intent clear? Are requirements ambiguous?
- **Blast radius**: Could this break other parts of the codebase?

Return ONLY valid JSON:
{"routing": "system1" or "system2", "confidence": 0.0-1.0, "reasoning": "one sentence explanation"}"""


@dataclass
class ClassificationResult:
    """Result of the System 1/2 classification."""
    routing: str
    confidence: float
    reasoning: str
    signals: dict = field(default_factory=dict)


class Classifier:
    """Hybrid classifier: LLM for ambiguous tasks, heuristics for fast-path."""

    def __init__(
        self,
        system2_threshold: float = 0.8,
        always_system2_tools: list[str] | None = None,
        llm_client: object | None = None,
    ):
        self.system2_threshold = system2_threshold
        self.always_system2_tools = always_system2_tools or []
        self._llm = llm_client

    def classify(
        self,
        tool_name: str,
        tool_input: dict,
        context: str,
        historical_success_rate: float | None = None,
    ) -> ClassificationResult:
        """Synchronous classification using heuristics only (v0.1 compat)."""
        return self._heuristic_classify(tool_name, tool_input, context, historical_success_rate)

    async def classify_async(
        self,
        tool_name: str,
        tool_input: dict,
        context: str,
        historical_success_rate: float | None = None,
    ) -> ClassificationResult:
        """Async classification: LLM for ambiguous tasks, heuristics for fast-path."""
        # Fast-path: read-only tools never need LLM
        if tool_name in _READ_ONLY_TOOLS:
            return ClassificationResult(
                routing="system1",
                confidence=0.95,
                reasoning=f"Read-only tool '{tool_name}'",
            )

        # Fast-path: forced System 2 tools
        if tool_name in self.always_system2_tools:
            return ClassificationResult(
                routing="system2",
                confidence=0.5,
                reasoning=f"Tool '{tool_name}' is configured for mandatory System 2",
            )

        # Try LLM classification for ambiguous cases
        if self._llm and getattr(self._llm, "enabled", False):
            llm_result = await self._llm_classify(tool_name, tool_input, context)
            if llm_result is not None:
                return llm_result

        # Fallback to heuristics
        return self._heuristic_classify(tool_name, tool_input, context, historical_success_rate)

    async def _llm_classify(
        self,
        tool_name: str,
        tool_input: dict,
        context: str,
    ) -> ClassificationResult | None:
        """Use LLM for semantic classification. Returns None on failure."""
        user_prompt = f"Tool: {tool_name}\nInput: {json.dumps(tool_input)[:500]}\nContext: {context[:500]}"
        try:
            resp = await self._llm.complete(
                model="gpt-4.1-nano",
                system_prompt=_CLASSIFIER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=256,
            )
            if resp.error:
                return None
            data = json.loads(resp.content)
            return ClassificationResult(
                routing=data["routing"],
                confidence=data["confidence"],
                reasoning=data.get("reasoning", "LLM classification"),
                signals={"source": "llm", "model": resp.model, "tokens": resp.total_tokens},
            )
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.warning("LLM classification failed, falling back to heuristics: %s", e)
            return None

    def _heuristic_classify(
        self,
        tool_name: str,
        tool_input: dict,
        context: str,
        historical_success_rate: float | None = None,
    ) -> ClassificationResult:
        """v0.1 heuristic classification (regex-based)."""
        signals = {
            "complexity": "low",
            "reversibility": "high",
            "uncertainty": "low",
            "historical_success_rate": historical_success_rate,
            "source": "heuristic",
        }
        confidence = 0.9
        reasons: list[str] = []

        if tool_name in self.always_system2_tools:
            return ClassificationResult(
                routing="system2", confidence=0.5,
                reasoning=f"Tool '{tool_name}' is configured for mandatory System 2",
                signals=signals,
            )

        if tool_name in _READ_ONLY_TOOLS:
            return ClassificationResult(
                routing="system1", confidence=0.95,
                reasoning=f"Read-only tool '{tool_name}'",
                signals=signals,
            )

        command = tool_input.get("command", "")
        if _DESTRUCTIVE_COMMANDS.search(command):
            confidence -= 0.35
            signals["reversibility"] = "low"
            reasons.append("Destructive command detected")

        if _SIMPLE_COMMANDS.match(command):
            confidence += 0.05
            reasons.append("Simple read-only command")

        if _COMPLEX_KEYWORDS.search(context):
            confidence -= 0.25
            signals["complexity"] = "high"
            reasons.append("Complex task keywords detected")

        if historical_success_rate is not None:
            if historical_success_rate < 0.5:
                confidence -= 0.2
                signals["uncertainty"] = "high"
                reasons.append(f"Low historical success rate ({historical_success_rate:.0%})")
            elif historical_success_rate < 0.7:
                confidence -= 0.1
                reasons.append(f"Moderate historical success rate ({historical_success_rate:.0%})")

        confidence = max(0.0, min(1.0, confidence))
        routing = "system1" if confidence >= self.system2_threshold else "system2"
        reasoning = "; ".join(reasons) if reasons else f"Standard {tool_name} operation"

        return ClassificationResult(
            routing=routing, confidence=confidence,
            reasoning=reasoning, signals=signals,
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: 9 PASS (6 existing + 3 new)

**Step 5: Commit**

```bash
git add src/metascaffold/classifier.py tests/test_classifier.py
git commit -m "feat(v0.2): add LLM-powered classification with heuristic fallback"
```

---

## Task 5: Evaluator v2 — LLM-as-Judge with SOFAI-LM feedback

**Files:**
- Modify: `src/metascaffold/evaluator.py`
- Modify: `tests/test_evaluator.py`

**Step 1: Write the failing tests**

Add to `tests/test_evaluator.py`:

```python
from unittest.mock import AsyncMock, MagicMock

class TestEvaluatorLLM:
    """Tests for LLM-enhanced evaluation."""

    @patch("metascaffold.evaluator.LLMClient")
    async def test_llm_evaluates_with_semantic_feedback(self, mock_llm_cls):
        """LLM should provide structured feedback, not just a verdict."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "verdict": "retry",
            "confidence": 0.4,
            "feedback": {
                "failing_tests": ["test_login_redirect"],
                "error_lines": [{"line": 42, "message": "redirect_url is None"}],
                "root_cause": "The redirect URL is not set when session expires",
                "suggested_fix": "Add a default redirect URL fallback in the login handler",
            },
            "adversarial_findings": [],
            "revision_allowed": True,
        })
        mock_response.error = ""
        mock_client.complete = AsyncMock(return_value=mock_response)
        mock_client.enabled = True

        evaluator = Evaluator(llm_client=mock_client)
        result = await evaluator.evaluate_async(
            sandbox_result=SandboxResult(
                exit_code=1, stdout="", stderr="AssertionError: test_login_redirect",
                duration_ms=500,
            ),
        )
        assert result.verdict == "retry"
        assert result.feedback["root_cause"] != ""
        assert result.feedback["suggested_fix"] != ""
        assert result.revision_allowed is True

    async def test_adversarial_check_downgrades_pass(self):
        """LLM adversarial check should downgrade pass to retry if issues found."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "verdict": "pass",
            "confidence": 0.6,
            "feedback": {},
            "adversarial_findings": [
                {"type": "sql_injection", "detail": "User input passed directly to query at line 15"},
            ],
            "revision_allowed": True,
        })
        mock_response.error = ""
        mock_client.complete = AsyncMock(return_value=mock_response)
        mock_client.enabled = True

        evaluator = Evaluator(llm_client=mock_client)
        result = await evaluator.evaluate_async(
            sandbox_result=SandboxResult(
                exit_code=0, stdout="All tests passed", stderr="",
                duration_ms=200,
            ),
        )
        # Adversarial findings should downgrade pass → retry
        assert result.verdict == "retry"
        assert len(result.adversarial_findings) > 0

    async def test_pag_blocks_empty_retry(self):
        """PAG: retry without concrete error should set revision_allowed=False."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "verdict": "retry",
            "confidence": 0.5,
            "feedback": {"failing_tests": [], "error_lines": [], "root_cause": "", "suggested_fix": ""},
            "adversarial_findings": [],
            "revision_allowed": False,
        })
        mock_response.error = ""
        mock_client.complete = AsyncMock(return_value=mock_response)
        mock_client.enabled = True

        evaluator = Evaluator(llm_client=mock_client)
        result = await evaluator.evaluate_async(
            sandbox_result=SandboxResult(
                exit_code=1, stdout="", stderr="Unknown error",
                duration_ms=100,
            ),
        )
        assert result.revision_allowed is False

    async def test_fallback_to_heuristics_when_llm_fails(self):
        """When LLM fails, should fall back to v0.1 heuristic evaluation."""
        mock_client = AsyncMock()
        mock_client.enabled = False

        evaluator = Evaluator(llm_client=mock_client)
        result = await evaluator.evaluate_async(
            sandbox_result=SandboxResult(
                exit_code=0, stdout="OK", stderr="",
                duration_ms=100,
            ),
        )
        assert result.verdict == "pass"
        assert result.feedback == {}  # No LLM feedback in heuristic mode
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluator.py::TestEvaluatorLLM -v`
Expected: FAIL — `evaluate_async` doesn't exist

**Step 3: Implement Evaluator v2**

The key changes to `evaluator.py`:
- Add `EvaluationResult.feedback` dict, `adversarial_findings` list, `revision_allowed` bool
- Add `evaluate_async()` method that calls LLM
- Keep `evaluate()` as sync heuristic fallback
- LLM system prompt asks for structured JSON analysis
- If LLM returns adversarial findings on a "pass", downgrade to "retry"
- If LLM returns "retry" with no concrete errors, set `revision_allowed=False`

```python
"""Evaluator v2 — LLM-as-Judge with SOFAI-LM feedback, adversarial check, and PAG revision gate.

v0.1 heuristics are kept as fallback when LLM is unavailable.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from metascaffold.sandbox import SandboxResult

logger = logging.getLogger("metascaffold.evaluator")

_SEVERE_ERRORS = re.compile(
    r"ModuleNotFoundError|ImportError|SyntaxError|IndentationError|RecursionError|MemoryError|PermissionError",
    re.IGNORECASE,
)

_TEST_FAILURE = re.compile(
    r"FAIL|failed|error|assert|AssertionError",
    re.IGNORECASE,
)

_EVALUATOR_SYSTEM_PROMPT = """You are a code evaluation judge for a coding AI agent.
Analyze the execution output and produce a structured verdict.

Your evaluation must include:
1. **Verdict**: pass (code works correctly), retry (fixable issues found), backtrack (fundamental approach is wrong), or escalate (human needed)
2. **Feedback**: Structured analysis with failing_tests, error_lines, root_cause, suggested_fix
3. **Adversarial check**: Look for security issues (injection, XSS, race conditions, resource leaks), missing edge cases (null, empty, overflow), and logic errors even if tests pass
4. **Revision gate**: Set revision_allowed=false if verdict is retry but you cannot identify a concrete, actionable error to fix (prevents infinite retry loops)

Return ONLY valid JSON:
{
  "verdict": "pass|retry|backtrack|escalate",
  "confidence": 0.0-1.0,
  "feedback": {
    "failing_tests": ["test names"],
    "error_lines": [{"line": 0, "message": "..."}],
    "root_cause": "explanation",
    "suggested_fix": "what to change"
  },
  "adversarial_findings": [{"type": "issue_type", "detail": "description"}],
  "revision_allowed": true
}"""


@dataclass
class Issue:
    """A single issue found during evaluation."""
    type: str
    detail: str
    severity: str

    def to_dict(self) -> dict:
        return {"type": self.type, "detail": self.detail, "severity": self.severity}


@dataclass
class EvaluationResult:
    """Result of the auto-evaluation."""
    verdict: str
    confidence: float
    issues: list[Issue] = field(default_factory=list)
    corrections: list[dict] = field(default_factory=list)
    attempt: int = 1
    max_attempts: int = 3
    feedback: dict = field(default_factory=dict)
    adversarial_findings: list[dict] = field(default_factory=list)
    revision_allowed: bool = True

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "issues": [i.to_dict() for i in self.issues],
            "corrections": self.corrections,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "feedback": self.feedback,
            "adversarial_findings": self.adversarial_findings,
            "revision_allowed": self.revision_allowed,
        }


class Evaluator:
    """Evaluates sandbox results — LLM-as-judge with heuristic fallback."""

    def __init__(self, max_retry_attempts: int = 3, llm_client: object | None = None):
        self.max_retry_attempts = max_retry_attempts
        self._llm = llm_client

    def evaluate(
        self,
        sandbox_result: SandboxResult,
        attempt: int = 1,
    ) -> EvaluationResult:
        """Synchronous heuristic evaluation (v0.1 compat)."""
        return self._heuristic_evaluate(sandbox_result, attempt)

    async def evaluate_async(
        self,
        sandbox_result: SandboxResult,
        attempt: int = 1,
    ) -> EvaluationResult:
        """Async LLM-powered evaluation with heuristic fallback."""
        # Try LLM evaluation
        if self._llm and getattr(self._llm, "enabled", False):
            llm_result = await self._llm_evaluate(sandbox_result, attempt)
            if llm_result is not None:
                return llm_result

        # Fallback to heuristics
        return self._heuristic_evaluate(sandbox_result, attempt)

    async def _llm_evaluate(
        self,
        sandbox_result: SandboxResult,
        attempt: int,
    ) -> EvaluationResult | None:
        """Use LLM-as-judge for semantic evaluation. Returns None on failure."""
        user_prompt = (
            f"Exit code: {sandbox_result.exit_code}\n"
            f"Timed out: {sandbox_result.timed_out}\n"
            f"Duration: {sandbox_result.duration_ms}ms\n"
            f"Attempt: {attempt}/{self.max_retry_attempts}\n"
            f"STDOUT:\n{sandbox_result.stdout[:2000]}\n"
            f"STDERR:\n{sandbox_result.stderr[:2000]}"
        )
        try:
            resp = await self._llm.complete(
                model="o3-mini",
                system_prompt=_EVALUATOR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=1024,
            )
            if resp.error:
                return None

            data = json.loads(resp.content)
            verdict = data["verdict"]
            adversarial = data.get("adversarial_findings", [])

            # Adversarial override: downgrade pass to retry if issues found
            if verdict == "pass" and adversarial:
                verdict = "retry"

            # PAG: enforce revision gate
            revision_allowed = data.get("revision_allowed", True)

            # Max attempts override
            if attempt >= self.max_retry_attempts and verdict == "retry":
                verdict = "escalate"

            return EvaluationResult(
                verdict=verdict,
                confidence=data.get("confidence", 0.5),
                feedback=data.get("feedback", {}),
                adversarial_findings=adversarial,
                revision_allowed=revision_allowed,
                attempt=attempt,
                max_attempts=self.max_retry_attempts,
            )
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.warning("LLM evaluation failed, falling back to heuristics: %s", e)
            return None

    def _heuristic_evaluate(
        self,
        sandbox_result: SandboxResult,
        attempt: int,
    ) -> EvaluationResult:
        """v0.1 heuristic evaluation (regex-based)."""
        issues: list[Issue] = []
        combined_output = sandbox_result.stdout + "\n" + sandbox_result.stderr

        if sandbox_result.timed_out:
            issues.append(Issue(
                type="timeout",
                detail=f"Command timed out after {sandbox_result.duration_ms}ms",
                severity="medium",
            ))

        if _SEVERE_ERRORS.search(sandbox_result.stderr):
            issues.append(Issue(
                type="severe_error",
                detail=sandbox_result.stderr.strip()[:200],
                severity="critical",
            ))

        if sandbox_result.exit_code != 0 and _TEST_FAILURE.search(combined_output):
            issues.append(Issue(
                type="test_failure",
                detail=sandbox_result.stderr.strip()[:200],
                severity="medium",
            ))

        if sandbox_result.exit_code == 0 and not issues:
            return EvaluationResult(
                verdict="pass", confidence=0.9,
                issues=issues, attempt=attempt, max_attempts=self.max_retry_attempts,
            )

        if any(i.severity == "critical" for i in issues):
            return EvaluationResult(
                verdict="backtrack", confidence=0.3,
                issues=issues, attempt=attempt, max_attempts=self.max_retry_attempts,
            )

        if attempt >= self.max_retry_attempts:
            return EvaluationResult(
                verdict="escalate", confidence=0.2,
                issues=issues, attempt=attempt, max_attempts=self.max_retry_attempts,
            )

        return EvaluationResult(
            verdict="retry", confidence=0.5,
            issues=issues, attempt=attempt, max_attempts=self.max_retry_attempts,
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: 10 PASS (6 existing + 4 new)

**Step 5: Commit**

```bash
git add src/metascaffold/evaluator.py tests/test_evaluator.py
git commit -m "feat(v0.2): LLM-as-judge evaluator with SOFAI feedback, adversarial check, PAG gate"
```

---

## Task 6: Planner v2 — LLM-generated context-aware strategies

**Files:**
- Modify: `src/metascaffold/planner.py`
- Modify: `tests/test_planner.py`

**Step 1: Write the failing tests**

Add to `tests/test_planner.py`:

```python
from unittest.mock import AsyncMock, MagicMock

class TestPlannerLLM:
    """Tests for LLM-enhanced planning."""

    @patch("metascaffold.planner.LLMClient")
    async def test_llm_generates_contextual_strategies(self, mock_llm_cls):
        """LLM should generate strategies specific to the actual task context."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "strategies": [
                {
                    "id": "A",
                    "description": "Add JWT validation middleware to FastAPI",
                    "steps": [
                        "Install python-jose dependency",
                        "Create auth/jwt.py with token validation logic",
                        "Write test for valid and expired tokens",
                        "Add Depends(verify_token) to protected routes",
                        "Run full test suite",
                    ],
                    "confidence": 0.85,
                    "risks": ["Token rotation not handled yet"],
                    "rollback_plan": "Remove middleware and dependency",
                },
            ],
            "recommended": "A",
        })
        mock_response.error = ""
        mock_client.complete = AsyncMock(return_value=mock_response)
        mock_client.enabled = True

        planner = Planner(llm_client=mock_client)
        plan = await planner.create_plan_async(
            task="Add JWT authentication to the API",
            context="FastAPI app in src/api/, using SQLAlchemy ORM",
        )
        assert len(plan.strategies) >= 1
        assert "JWT" in plan.strategies[0].description or "jwt" in plan.strategies[0].steps[0].lower()

    async def test_fallback_to_heuristics_when_llm_disabled(self):
        """When LLM is disabled, use v0.1 template strategies."""
        mock_client = AsyncMock()
        mock_client.enabled = False

        planner = Planner(llm_client=mock_client)
        plan = await planner.create_plan_async(
            task="Fix the login bug",
            context="Users can't log in after password reset",
        )
        # Should use heuristic bugfix template
        assert len(plan.strategies) >= 1
        assert plan.strategies[0].id == "A"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_planner.py::TestPlannerLLM -v`
Expected: FAIL — `create_plan_async` doesn't exist

**Step 3: Implement Planner v2**

Add to `src/metascaffold/planner.py`:
- Add `llm_client` parameter to `__init__`
- Add `create_plan_async()` method
- LLM system prompt asks for structured strategies with steps, risks, rollback
- Parse LLM JSON response into Strategy/Plan dataclasses
- Keep `create_plan()` as sync heuristic fallback

The LLM system prompt:
```
_PLANNER_SYSTEM_PROMPT = """You are a strategic planner for a coding AI agent.
Given a task and context, decompose it into 1-3 execution strategies.

Each strategy must include:
- id: letter (A, B, C)
- description: one-line summary of the approach
- steps: ordered list of concrete actions (5-8 steps each)
- confidence: 0.0-1.0 (how likely this strategy succeeds)
- risks: list of potential failure modes
- rollback_plan: how to undo if this strategy fails

Return ONLY valid JSON:
{"strategies": [...], "recommended": "A"}"""
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_planner.py -v`
Expected: 7 PASS (5 existing + 2 new)

**Step 5: Commit**

```bash
git add src/metascaffold/planner.py tests/test_planner.py
git commit -m "feat(v0.2): LLM-powered planner with context-aware strategy generation"
```

---

## Task 7: Create Distiller — LLM-powered task structuring

**Files:**
- Create: `src/metascaffold/distiller.py`
- Create: `tests/test_distiller.py`

**Step 1: Write the failing tests**

```python
# tests/test_distiller.py
"""Tests for the Task Distiller component."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

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

    @patch("metascaffold.distiller.LLMClient")
    async def test_distill_produces_template(self, mock_llm_cls):
        """Distiller should produce a structured TaskTemplate from raw task text."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "objective": "Add JWT-based authentication to the FastAPI application",
            "constraints": ["Use python-jose for JWT", "Support token refresh"],
            "target_files": ["src/api/auth.py", "src/api/middleware.py"],
            "variables": {"token_expiry": "30m", "algorithm": "HS256"},
        })
        mock_response.error = ""
        mock_client.complete = AsyncMock(return_value=mock_response)
        mock_client.enabled = True

        distiller = Distiller(llm_client=mock_client)
        template = await distiller.distill(
            task="Add JWT auth to the API",
            context="FastAPI app with SQLAlchemy",
        )
        assert template.objective != ""
        assert len(template.target_files) > 0

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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_distiller.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'metascaffold.distiller'`

**Step 3: Implement Distiller**

```python
# src/metascaffold/distiller.py
"""Distiller — LLM-powered task structuring (Self-Thought Task Distillation).

Transforms raw task descriptions into structured TaskTemplates before planning.
Falls back to passthrough when LLM is unavailable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("metascaffold.distiller")

_DISTILLER_SYSTEM_PROMPT = """You are a task analyst for a coding AI agent.
Given a raw task description and context, extract a structured task template.

Extract:
- objective: clear one-sentence goal
- constraints: technical or business rules that must be respected
- target_files: files likely affected (infer from context)
- variables: key parameters, values, or configuration mentioned

Return ONLY valid JSON:
{
  "objective": "...",
  "constraints": ["..."],
  "target_files": ["..."],
  "variables": {"key": "value"}
}"""


@dataclass
class TaskTemplate:
    """Structured task representation produced by the Distiller."""
    objective: str
    constraints: list[str] = field(default_factory=list)
    target_files: list[str] = field(default_factory=list)
    variables: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "objective": self.objective,
            "constraints": self.constraints,
            "target_files": self.target_files,
            "variables": self.variables,
        }


class Distiller:
    """Transforms raw task text into structured TaskTemplate."""

    def __init__(self, llm_client: object | None = None):
        self._llm = llm_client

    async def distill(self, task: str, context: str) -> TaskTemplate:
        """Distill a raw task into a structured template."""
        if self._llm and getattr(self._llm, "enabled", False):
            result = await self._llm_distill(task, context)
            if result is not None:
                return result

        # Fallback: passthrough with no enrichment
        return TaskTemplate(objective=task)

    async def _llm_distill(self, task: str, context: str) -> TaskTemplate | None:
        """Use LLM for task distillation. Returns None on failure."""
        user_prompt = f"Task: {task}\nContext: {context[:1000]}"
        try:
            resp = await self._llm.complete(
                model="gpt-4.1-nano",
                system_prompt=_DISTILLER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=512,
            )
            if resp.error:
                return None
            data = json.loads(resp.content)
            return TaskTemplate(
                objective=data.get("objective", task),
                constraints=data.get("constraints", []),
                target_files=data.get("target_files", []),
                variables=data.get("variables", {}),
            )
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("LLM distillation failed: %s", e)
            return None
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_distiller.py -v`
Expected: 3 PASS

**Step 5: Commit**

```bash
git add src/metascaffold/distiller.py tests/test_distiller.py
git commit -m "feat(v0.2): add Distiller component for LLM-powered task structuring"
```

---

## Task 8: Create Reflector — MARS reflection loop

**Files:**
- Create: `src/metascaffold/reflector.py`
- Create: `tests/test_reflector.py`

**Step 1: Write the failing tests**

```python
# tests/test_reflector.py
"""Tests for the MARS Reflector component."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metascaffold.reflector import Reflector, ReflectionResult


class TestReflector:
    def test_reflection_result_dataclass(self):
        """ReflectionResult should hold rules and procedures."""
        r = ReflectionResult(
            rules=["Always validate input before DB queries"],
            procedures=["1. Write test 2. Implement 3. Verify"],
            source_event_count=10,
        )
        assert len(r.rules) == 1
        assert r.source_event_count == 10

    @patch("metascaffold.reflector.LLMClient")
    async def test_reflect_extracts_rules_from_telemetry(self, mock_llm_cls):
        """Reflector should analyze telemetry events and extract patterns."""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "rules": [
                "Always run the full test suite after modifying shared modules",
                "SyntaxError usually indicates a missing import — check imports first",
            ],
            "procedures": [
                "When a test fails: 1) Read the error message 2) Check the test assertion 3) Fix the minimal code path",
            ],
        })
        mock_response.error = ""
        mock_client.complete = AsyncMock(return_value=mock_response)
        mock_client.enabled = True

        # Fake telemetry events
        events = [
            {"event_type": "evaluation", "data": {"verdict": "retry", "num_issues": 2}},
            {"event_type": "evaluation", "data": {"verdict": "pass", "num_issues": 0}},
            {"event_type": "evaluation", "data": {"verdict": "backtrack", "num_issues": 1}},
        ]

        reflector = Reflector(llm_client=mock_client)
        result = await reflector.reflect(events)
        assert len(result.rules) > 0
        assert len(result.procedures) > 0
        assert result.source_event_count == 3

    async def test_reflect_returns_empty_when_no_events(self):
        """Reflector should return empty result with no events."""
        mock_client = AsyncMock()
        mock_client.enabled = True

        reflector = Reflector(llm_client=mock_client)
        result = await reflector.reflect([])
        assert result.rules == []
        assert result.source_event_count == 0

    async def test_reflect_fallback_when_llm_disabled(self):
        """When LLM is disabled, return empty reflection."""
        mock_client = AsyncMock()
        mock_client.enabled = False

        events = [{"event_type": "evaluation", "data": {"verdict": "pass"}}]
        reflector = Reflector(llm_client=mock_client)
        result = await reflector.reflect(events)
        assert result.rules == []
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_reflector.py -v`
Expected: FAIL — module not found

**Step 3: Implement Reflector**

```python
# src/metascaffold/reflector.py
"""Reflector — MARS reflection loop for learning from telemetry.

Analyzes recent cognitive events (evaluations, backtracks, escalations)
and extracts reusable rules and procedures via LLM analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("metascaffold.reflector")

_REFLECTOR_SYSTEM_PROMPT = """You are a metacognitive reflector for a coding AI agent.
Analyze the batch of cognitive telemetry events below and extract learning patterns.

Extract two types of artifacts:
1. **Rules**: Normative constraints the agent should always follow (e.g., "Always run tests after modifying shared code")
2. **Procedures**: Step-by-step strategies that worked well or should replace failed approaches

Focus on:
- Patterns in failures (what keeps going wrong?)
- Patterns in successes (what keeps working?)
- Recurring backtracks or escalations (what should the agent avoid?)

Return ONLY valid JSON:
{"rules": ["..."], "procedures": ["..."]}"""


@dataclass
class ReflectionResult:
    """Result of the MARS reflection analysis."""
    rules: list[str] = field(default_factory=list)
    procedures: list[str] = field(default_factory=list)
    source_event_count: int = 0

    def to_dict(self) -> dict:
        return {
            "rules": self.rules,
            "procedures": self.procedures,
            "source_event_count": self.source_event_count,
        }


class Reflector:
    """Analyzes telemetry to extract reusable rules and procedures."""

    def __init__(self, llm_client: object | None = None):
        self._llm = llm_client

    async def reflect(self, events: list[dict]) -> ReflectionResult:
        """Analyze a batch of telemetry events and extract patterns."""
        if not events:
            return ReflectionResult(source_event_count=0)

        if self._llm and getattr(self._llm, "enabled", False):
            result = await self._llm_reflect(events)
            if result is not None:
                return result

        # Fallback: no reflection without LLM
        return ReflectionResult(source_event_count=len(events))

    async def _llm_reflect(self, events: list[dict]) -> ReflectionResult | None:
        """Use LLM to analyze telemetry and extract patterns."""
        # Summarize events for the prompt (limit to last 50)
        recent = events[-50:]
        events_text = json.dumps(recent, indent=2)[:4000]
        user_prompt = f"Telemetry events ({len(recent)} most recent):\n{events_text}"

        try:
            resp = await self._llm.complete(
                model="o3-mini",
                system_prompt=_REFLECTOR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=1024,
            )
            if resp.error:
                return None
            data = json.loads(resp.content)
            return ReflectionResult(
                rules=data.get("rules", []),
                procedures=data.get("procedures", []),
                source_event_count=len(events),
            )
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("LLM reflection failed: %s", e)
            return None
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_reflector.py -v`
Expected: 4 PASS

**Step 5: Commit**

```bash
git add src/metascaffold/reflector.py tests/test_reflector.py
git commit -m "feat(v0.2): add Reflector component for MARS telemetry learning"
```

---

## Task 9: Create Pipeline orchestrator

**Files:**
- Create: `src/metascaffold/pipeline.py`
- Create: `tests/test_pipeline.py`

**Step 1: Write the failing tests**

```python
# tests/test_pipeline.py
"""Tests for the cognitive pipeline orchestrator."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

import pytest

from metascaffold.pipeline import CognitivePipeline, PipelineState


class TestPipeline:
    def test_pipeline_state_dataclass(self):
        """PipelineState should track all 6 stages."""
        state = PipelineState(task="Fix bug", context="src/main.py")
        assert state.task == "Fix bug"
        assert state.classification is None
        assert state.template is None
        assert state.plan is None
        assert state.execution is None
        assert state.evaluation is None
        assert state.reflection is None

    async def test_system1_bypasses_distill_and_plan(self):
        """System 1 tasks should skip Distill and Plan stages."""
        mock_classifier = AsyncMock()
        mock_classifier.classify_async = AsyncMock(return_value=MagicMock(
            routing="system1", confidence=0.95, reasoning="Read-only",
        ))

        pipeline = CognitivePipeline(classifier=mock_classifier)
        state = await pipeline.classify_stage(
            PipelineState(task="Read file", context="")
        )
        assert state.classification.routing == "system1"
        assert state.should_bypass is True

    async def test_system2_runs_full_pipeline(self):
        """System 2 tasks should proceed through all stages."""
        mock_classifier = AsyncMock()
        mock_classifier.classify_async = AsyncMock(return_value=MagicMock(
            routing="system2", confidence=0.4, reasoning="Complex refactor",
        ))

        pipeline = CognitivePipeline(classifier=mock_classifier)
        state = await pipeline.classify_stage(
            PipelineState(task="Refactor auth", context="")
        )
        assert state.classification.routing == "system2"
        assert state.should_bypass is False

    async def test_retry_increments_attempt(self):
        """Retry verdict should increment attempt counter in state."""
        state = PipelineState(task="test", context="", attempt=1)
        state = state.with_retry()
        assert state.attempt == 2

    async def test_max_attempts_triggers_escalation(self):
        """Exceeding max attempts should signal escalation."""
        state = PipelineState(task="test", context="", attempt=3, max_attempts=3)
        assert state.should_escalate is True
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL — module not found

**Step 3: Implement Pipeline**

```python
# src/metascaffold/pipeline.py
"""Pipeline — 6-stage cognitive orchestrator.

Manages the flow: Classify → Distill → Plan → Execute → Evaluate → Reflect
with System 1 bypass and retry/backtrack loops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

logger = logging.getLogger("metascaffold.pipeline")


@dataclass
class PipelineState:
    """State that flows through the 6-stage pipeline."""
    task: str
    context: str
    classification: object | None = None
    template: object | None = None
    plan: object | None = None
    execution: object | None = None
    evaluation: object | None = None
    reflection: object | None = None
    attempt: int = 1
    max_attempts: int = 3

    @property
    def should_bypass(self) -> bool:
        """True if classification says System 1 (skip Distill+Plan)."""
        if self.classification is None:
            return False
        return getattr(self.classification, "routing", "") == "system1"

    @property
    def should_escalate(self) -> bool:
        """True if attempts exhausted."""
        return self.attempt >= self.max_attempts

    def with_retry(self) -> PipelineState:
        """Return new state with incremented attempt, cleared execution/evaluation."""
        return replace(
            self,
            attempt=self.attempt + 1,
            execution=None,
            evaluation=None,
        )

    def to_dict(self) -> dict:
        def _safe_dict(obj: object | None) -> dict | None:
            if obj is None:
                return None
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if hasattr(obj, "__dict__"):
                return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
            return None

        return {
            "task": self.task,
            "context": self.context[:200],
            "attempt": self.attempt,
            "classification": _safe_dict(self.classification),
            "template": _safe_dict(self.template),
            "plan": _safe_dict(self.plan),
            "evaluation": _safe_dict(self.evaluation),
            "reflection": _safe_dict(self.reflection),
        }


class CognitivePipeline:
    """Orchestrates the 6-stage cognitive pipeline."""

    def __init__(
        self,
        classifier: object | None = None,
        distiller: object | None = None,
        planner: object | None = None,
        evaluator: object | None = None,
        reflector: object | None = None,
    ):
        self.classifier = classifier
        self.distiller = distiller
        self.planner = planner
        self.evaluator = evaluator
        self.reflector = reflector

    async def classify_stage(self, state: PipelineState) -> PipelineState:
        """Stage 1: Classify the task as System 1 or System 2."""
        if self.classifier is None:
            return state
        result = await self.classifier.classify_async(
            tool_name="pipeline",
            tool_input={},
            context=state.task + " " + state.context,
        )
        return replace(state, classification=result)

    async def distill_stage(self, state: PipelineState) -> PipelineState:
        """Stage 2: Distill the task into a structured template."""
        if state.should_bypass or self.distiller is None:
            return state
        template = await self.distiller.distill(state.task, state.context)
        return replace(state, template=template)

    async def plan_stage(self, state: PipelineState) -> PipelineState:
        """Stage 3: Generate execution strategies."""
        if state.should_bypass or self.planner is None:
            return state
        nlm_insights = ""
        task_text = state.task
        if state.template and hasattr(state.template, "objective"):
            task_text = state.template.objective
        plan = await self.planner.create_plan_async(
            task=task_text,
            context=state.context,
            notebooklm_insights=nlm_insights,
        )
        return replace(state, plan=plan)

    async def evaluate_stage(self, state: PipelineState, sandbox_result: object) -> PipelineState:
        """Stage 5: Evaluate execution results."""
        if self.evaluator is None:
            return replace(state, execution=sandbox_result)
        evaluation = await self.evaluator.evaluate_async(
            sandbox_result=sandbox_result,
            attempt=state.attempt,
        )
        return replace(state, execution=sandbox_result, evaluation=evaluation)

    async def reflect_stage(self, state: PipelineState, events: list[dict]) -> PipelineState:
        """Stage 6: Extract learning from telemetry."""
        if self.reflector is None:
            return state
        reflection = await self.reflector.reflect(events)
        return replace(state, reflection=reflection)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 5 PASS

**Step 5: Commit**

```bash
git add src/metascaffold/pipeline.py tests/test_pipeline.py
git commit -m "feat(v0.2): add CognitivePipeline orchestrator with 6-stage flow"
```

---

## Task 10: Wire everything into `server.py`

**Files:**
- Modify: `src/metascaffold/server.py`
- Modify: `tests/test_server.py`

**Step 1: Write the failing tests**

Add to `tests/test_server.py`:

```python
def test_server_has_reflect_tool(self):
    """metascaffold_reflect tool should be registered."""
    tool_names = [t.name for t in mcp._tool_manager.list_tools()]
    assert "metascaffold_reflect" in tool_names

def test_server_has_nine_tools(self):
    """v0.2 should have 9 tools total."""
    tool_names = [t.name for t in mcp._tool_manager.list_tools()]
    assert len(tool_names) == 9

def test_server_has_distill_tool(self):
    """metascaffold_distill tool should be registered."""
    tool_names = [t.name for t in mcp._tool_manager.list_tools()]
    assert "metascaffold_distill" in tool_names
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -v`
Expected: FAIL — tools don't exist yet

**Step 3: Implement server integration**

Key changes to `server.py`:
1. Import and instantiate `LLMClient`, `Distiller`, `Reflector`, `CognitivePipeline`
2. Pass `llm_client` to `Classifier`, `Evaluator`, `Planner`, `Distiller`, `Reflector`
3. Make `metascaffold_classify` async (call `classify_async`)
4. Make `metascaffold_evaluate` async (call `evaluate_async`)
5. Add new tools: `metascaffold_distill`, `metascaffold_reflect`
6. Update `_reload_components()` and `_RELOAD_ORDER` for new modules
7. Update docstring to say "9 tools"

The server initialization becomes:

```python
# Initialize LLM client
from metascaffold.llm_client import LLMClient
llm_client = LLMClient()

# Initialize components with LLM
classifier = Classifier(
    system2_threshold=config.classifier.system2_threshold,
    always_system2_tools=config.classifier.always_system2_tools,
    llm_client=llm_client if config.llm.enabled else None,
)
planner = Planner(llm_client=llm_client if config.llm.enabled else None)
evaluator = Evaluator(
    max_retry_attempts=config.sandbox.max_retry_attempts,
    llm_client=llm_client if config.llm.enabled else None,
)

from metascaffold.distiller import Distiller
distiller = Distiller(llm_client=llm_client if config.llm.enabled else None)

from metascaffold.reflector import Reflector
reflector = Reflector(llm_client=llm_client if config.llm.enabled else None)
```

New tools:

```python
@mcp.tool()
async def metascaffold_distill(
    task: Annotated[str, Field(description="Raw task description to structure")],
    context: Annotated[str, Field(description="Additional context about the codebase")],
) -> dict:
    """Distill a raw task into a structured template with objective, constraints, files, and variables.

    Uses LLM to semantically analyze the task. Falls back to passthrough when LLM is unavailable.
    """
    template = await distiller.distill(task=task, context=context)
    return template.to_dict()


@mcp.tool()
async def metascaffold_reflect(
    limit: Annotated[int, Field(description="Number of recent events to analyze")] = 50,
) -> dict:
    """Analyze recent telemetry events and extract reusable rules and procedures.

    Uses LLM to find patterns in successes, failures, backtracks, and escalations.
    Returns rules (normative constraints) and procedures (step-by-step strategies).
    """
    events = telemetry.get_recent_events(limit)
    result = await reflector.reflect(events)
    return result.to_dict()
```

Note: `telemetry.get_recent_events(limit)` needs to be added to `TelemetryLogger`.

**Step 4: Add `get_recent_events` to telemetry.py**

```python
def get_recent_events(self, limit: int = 50) -> list[dict]:
    """Retrieve recent events from SQLite for reflection analysis."""
    conn = sqlite3.connect(self._sqlite_path)
    rows = conn.execute(
        "SELECT event_type, data_json FROM events ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [
        {"event_type": row[0], "data": json.loads(row[1])}
        for row in reversed(rows)
    ]
```

**Step 5: Update `metascaffold_classify` to use async**

```python
@mcp.tool()
async def metascaffold_classify(...) -> dict:
    result = await classifier.classify_async(...)
    ...
```

**Step 6: Update `metascaffold_evaluate` to use async**

```python
@mcp.tool()
async def metascaffold_evaluate(...) -> dict:
    result = await evaluator.evaluate_async(...)
    ...
```

**Step 7: Update `_RELOAD_ORDER` and `_reload_components`**

Add to `_RELOAD_ORDER`:
```python
_RELOAD_ORDER = [
    "metascaffold.config",
    "metascaffold.telemetry",
    "metascaffold.notebooklm_bridge",
    "metascaffold.llm_client",
    "metascaffold.classifier",
    "metascaffold.planner",
    "metascaffold.sandbox",
    "metascaffold.evaluator",
    "metascaffold.distiller",
    "metascaffold.reflector",
    "metascaffold.pipeline",
]
```

Add re-instantiation of new components in `_reload_components()`.

**Step 8: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass

**Step 9: Commit**

```bash
git add src/metascaffold/server.py src/metascaffold/telemetry.py tests/test_server.py
git commit -m "feat(v0.2): wire LLM pipeline into server — 9 tools, async classify/evaluate"
```

---

## Task 11: Integration test — full LLM pipeline end-to-end

**Files:**
- Modify: `tests/test_integration.py`

**Step 1: Write the integration test**

Add a new test class to `tests/test_integration.py`:

```python
class TestLLMPipelineIntegration:
    """Integration test for the full LLM cognitive pipeline."""

    async def test_system2_full_pipeline_with_mocked_llm(self):
        """System 2 task should flow through all 6 stages with LLM."""
        # Mock LLM client
        mock_llm = AsyncMock()
        mock_llm.enabled = True

        # Stage 1: Classify → system2
        classify_resp = MagicMock()
        classify_resp.content = json.dumps({"routing": "system2", "confidence": 0.3, "reasoning": "Complex refactor"})
        classify_resp.error = ""

        # Stage 2: Distill → template
        distill_resp = MagicMock()
        distill_resp.content = json.dumps({"objective": "Refactor auth module", "constraints": [], "target_files": ["src/auth.py"], "variables": {}})
        distill_resp.error = ""

        # Stage 3: Plan → strategies
        plan_resp = MagicMock()
        plan_resp.content = json.dumps({"strategies": [{"id": "A", "description": "TDD refactor", "steps": ["test", "refactor", "verify"], "confidence": 0.8, "risks": [], "rollback_plan": "revert"}], "recommended": "A"})
        plan_resp.error = ""

        # Stage 5: Evaluate → pass
        eval_resp = MagicMock()
        eval_resp.content = json.dumps({"verdict": "pass", "confidence": 0.9, "feedback": {}, "adversarial_findings": [], "revision_allowed": True})
        eval_resp.error = ""

        mock_llm.complete = AsyncMock(side_effect=[classify_resp, distill_resp, plan_resp, eval_resp])

        from metascaffold.classifier import Classifier
        from metascaffold.distiller import Distiller
        from metascaffold.planner import Planner
        from metascaffold.evaluator import Evaluator
        from metascaffold.pipeline import CognitivePipeline, PipelineState
        from metascaffold.sandbox import SandboxResult

        pipeline = CognitivePipeline(
            classifier=Classifier(llm_client=mock_llm),
            distiller=Distiller(llm_client=mock_llm),
            planner=Planner(llm_client=mock_llm),
            evaluator=Evaluator(llm_client=mock_llm),
        )

        state = PipelineState(task="Refactor auth module", context="src/auth.py needs cleanup")
        state = await pipeline.classify_stage(state)
        assert state.classification.routing == "system2"

        state = await pipeline.distill_stage(state)
        assert state.template is not None

        state = await pipeline.plan_stage(state)
        assert state.plan is not None

        # Simulate execution
        sandbox_result = SandboxResult(exit_code=0, stdout="All tests passed", stderr="", duration_ms=200)
        state = await pipeline.evaluate_stage(state, sandbox_result)
        assert state.evaluation.verdict == "pass"
```

**Step 2: Run test**

Run: `uv run pytest tests/test_integration.py -v`
Expected: All integration tests pass

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test(v0.2): add LLM pipeline integration test"
```

---

## Task 12: Update CLAUDE.md, version, and documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `pyproject.toml:3` (version → 0.2.0)
- Modify: `docs/plans/2026-03-03-metascaffold-design.md` (add v0.2 section)

**Step 1: Bump version**

In `pyproject.toml`, change `version = "0.1.0"` to `version = "0.2.0"`.

**Step 2: Update CLAUDE.md**

Update the overview to mention LLM-powered components, 9 tools, and the cognitive pipeline.

**Step 3: Commit**

```bash
git add CLAUDE.md pyproject.toml docs/
git commit -m "docs(v0.2): update CLAUDE.md, bump version to 0.2.0"
```

---

## Task 13: Live validation — test with real LLM calls

**Step 1: Hot-reload the server**

Call `metascaffold_restart` to reload all modules.

**Step 2: Test classify with real LLM**

Call `metascaffold_classify` with a complex task and verify the response includes `"source": "llm"` in signals.

**Step 3: Test evaluate with real LLM**

Call `metascaffold_evaluate` with a failed test output and verify the response includes semantic feedback (not just regex patterns).

**Step 4: Test distill**

Call `metascaffold_distill` with a raw task description and verify it returns a structured template.

**Step 5: Test reflect**

Call `metascaffold_reflect` and verify it returns rules extracted from telemetry history.

**Step 6: Final commit**

```bash
git add -A
git commit -m "feat: MetaScaffold v0.2.0 — LLM-powered cognitive pipeline"
```

---

## Summary

| Task | Component | Key Change | Tests Added |
|------|-----------|------------|-------------|
| 1 | llm_client.py | OpenAI SDK + Codex OAuth tokens | 6 |
| 2 | config.py | LLM model config per component | 1 |
| 3 | Spike | Verify OAuth token works with API | 0 |
| 4 | classifier.py | LLM semantic classification | 3 |
| 5 | evaluator.py | LLM-as-judge + SOFAI + adversarial + PAG | 4 |
| 6 | planner.py | LLM context-aware strategies | 2 |
| 7 | distiller.py | NEW: task structuring | 3 |
| 8 | reflector.py | NEW: MARS telemetry learning | 4 |
| 9 | pipeline.py | NEW: 6-stage orchestrator | 5 |
| 10 | server.py | Wire 9 tools, async classify/evaluate | 3 |
| 11 | integration | Full pipeline E2E test | 1 |
| 12 | Docs | Version bump, CLAUDE.md | 0 |
| 13 | Live | Real LLM validation | 0 |
| **Total** | | | **32** |

**Architecture diagram:**
```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ CLASSIFY │───▶│ DISTILL  │───▶│  PLAN    │───▶│ EXECUTE  │───▶│ EVALUATE │───▶│ REFLECT  │
│  LLM:    │    │  LLM:    │    │  LLM:    │    │          │    │  LLM:    │    │  LLM:    │
│  nano    │    │  nano    │    │  4.1-mini│    │ Sandbox  │    │  o3-mini │    │  o3-mini │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
      │                                                              │               │
      │  S1 ──────────────────────────────────▶ Bypass direct        │               │
      │                                                              │               │
      └──────────────── Retry/Backtrack ◀────────────────────────────┘               │
                                                                                      │
                                                                     NLM ◀────────────┘
                                                               (rules + procedures)
```
