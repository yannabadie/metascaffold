# MetaScaffold v0.3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Evolve MetaScaffold from prompt-based confidence to grounded, measurable cognitive signals — entropy routing, deterministic verifiers, memory decay, and adaptive compute.

**Architecture:** Add a dual-backend LLM client (codex exec default + OpenAI API for logprobs), entropy-based classifier routing via gpt-4.1-nano, a `VerificationSuite` that runs AST/Ruff/pytest/mypy before LLM-as-Judge, an Ebbinghaus decay memory system for Reflector rules, and 3-level adaptive compute (System 1/1.5/2) in the pipeline.

**Tech Stack:** Python 3.11+, OpenAI SDK (`openai>=1.60.0`), `truststore`, Ruff, mypy, pytest, `math` stdlib for entropy/decay.

**Design doc:** `docs/plans/2026-03-03-metascaffold-v03-design.md`

---

### Task 1: Add `complete_with_logprobs()` to LLM Client

**Files:**
- Modify: `src/metascaffold/llm_client.py`
- Test: `tests/test_llm_client.py`

**Context:** The existing `LLMClient` only uses `codex exec` subprocess. We need a second code path that calls the OpenAI API directly with `logprobs=True` via the `openai` SDK. The API key is in `.env` as `OPEN_API_KEY` (non-standard name). Corporate SSL requires `truststore`. This method is ONLY used for entropy routing in the Classifier — all other LLM calls continue to use `codex exec`.

**Step 1: Write failing tests for `complete_with_logprobs()`**

Add to `tests/test_llm_client.py` a new `TestCompleteWithLogprobs` class:

```python
class TestCompleteWithLogprobs:
    """Tests for the OpenAI API logprobs path."""

    async def test_returns_logprobs_on_success(self):
        """complete_with_logprobs() should return content + token_logprobs."""
        mock_choice = MagicMock()
        mock_choice.message.content = '{"routing": "system2"}'
        mock_logprob_content = MagicMock()
        mock_logprob_content.token = "system"
        mock_logprob_content.logprob = -0.05
        top_lp = MagicMock()
        top_lp.token = "system"
        top_lp.logprob = -0.05
        mock_logprob_content.top_logprobs = [top_lp]
        mock_choice.logprobs.content = [mock_logprob_content]

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4.1-nano"
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 10

        with patch("metascaffold.llm_client.OpenAI") as MockOpenAI:
            mock_client_instance = MagicMock()
            mock_client_instance.chat.completions.create.return_value = mock_response
            MockOpenAI.return_value = mock_client_instance

            client = LLMClient(codex_path="/usr/bin/codex")
            result = await client.complete_with_logprobs(
                model="gpt-4.1-nano",
                system_prompt="Classify this.",
                user_prompt="Edit a file.",
                max_tokens=64,
            )

        assert result.content == '{"routing": "system2"}'
        assert result.error == ""
        assert len(result.token_logprobs) == 1
        assert result.token_logprobs[0]["token"] == "system"

    async def test_returns_error_when_no_api_key(self):
        """complete_with_logprobs() should return error if OPEN_API_KEY not set."""
        with patch.dict("os.environ", {}, clear=True):
            client = LLMClient(codex_path="/usr/bin/codex")
            result = await client.complete_with_logprobs(
                model="gpt-4.1-nano",
                system_prompt="test",
                user_prompt="test",
            )
        assert result.content == ""
        assert "api key" in result.error.lower()

    async def test_returns_error_on_api_failure(self):
        """complete_with_logprobs() should handle API errors gracefully."""
        with patch("metascaffold.llm_client.OpenAI") as MockOpenAI:
            mock_client_instance = MagicMock()
            mock_client_instance.chat.completions.create.side_effect = Exception("API error")
            MockOpenAI.return_value = mock_client_instance

            client = LLMClient(codex_path="/usr/bin/codex")
            # Ensure env var is set so we get past the key check
            with patch.dict("os.environ", {"OPEN_API_KEY": "sk-test"}):
                result = await client.complete_with_logprobs(
                    model="gpt-4.1-nano",
                    system_prompt="test",
                    user_prompt="test",
                )
        assert result.content == ""
        assert "api error" in result.error.lower()
```

You'll also need to add these imports at the top of the test file:

```python
import os
from unittest.mock import patch, AsyncMock, MagicMock
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm_client.py::TestCompleteWithLogprobs -v`
Expected: FAIL — `LLMClient` has no `complete_with_logprobs` method, no `token_logprobs` on `LLMResponse`.

**Step 3: Implement `complete_with_logprobs()` in `llm_client.py`**

First, add `token_logprobs` field to `LLMResponse`:

```python
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
```

Add import at the top of `llm_client.py`:

```python
import os
from dataclasses import dataclass, field
```

Then add the method to `LLMClient`:

```python
async def complete_with_logprobs(
    self,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 64,
    top_logprobs: int = 5,
) -> LLMResponse:
    """Call OpenAI API directly with logprobs=True.

    Used ONLY for entropy-based routing in the Classifier.
    All other LLM calls use codex exec (the default path).

    Requires OPEN_API_KEY environment variable.
    Uses truststore for corporate SSL proxy compatibility.
    """
    api_key = os.environ.get("OPEN_API_KEY", "")
    if not api_key:
        return LLMResponse(error="No API key: OPEN_API_KEY env var not set")

    try:
        import truststore
        truststore.inject_into_ssl()
    except Exception:
        pass  # Non-critical — may work without truststore

    try:
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
        token_logprobs = []
        if choice.logprobs and choice.logprobs.content:
            for token_info in choice.logprobs.content:
                entry = {
                    "token": token_info.token,
                    "logprob": token_info.logprob,
                    "top_logprobs": [
                        {"token": tp.token, "logprob": tp.logprob}
                        for tp in (token_info.top_logprobs or [])
                    ],
                }
                token_logprobs.append(entry)

        return LLMResponse(
            content=content,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            token_logprobs=token_logprobs,
        )
    except Exception as e:
        logger.warning("OpenAI API call failed: %s", e)
        return LLMResponse(error=str(e))
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm_client.py -v`
Expected: ALL PASS (including all existing tests — the new `token_logprobs` field has a default)

**Step 5: Commit**

```bash
git add src/metascaffold/llm_client.py tests/test_llm_client.py
git commit -m "feat(v0.3): add complete_with_logprobs() for OpenAI API direct path"
```

---

### Task 2: Entropy Computation Utility

**Files:**
- Create: `src/metascaffold/entropy.py`
- Create: `tests/test_entropy.py`

**Context:** We need a pure function that computes Shannon entropy from token logprobs. This is used by the Classifier to measure model uncertainty. The function takes a list of `{token, logprob}` dicts (top-k logprobs for a single token position) and returns the entropy value in bits.

**Step 1: Write failing tests**

Create `tests/test_entropy.py`:

```python
"""Tests for the entropy computation utility."""

import math

import pytest

from metascaffold.entropy import compute_entropy, find_routing_token_entropy


class TestComputeEntropy:
    def test_single_token_zero_entropy(self):
        """A single token with probability 1.0 has zero entropy."""
        logprobs = [{"token": "system2", "logprob": 0.0}]
        assert compute_entropy(logprobs) == pytest.approx(0.0, abs=0.001)

    def test_two_tokens_equal_probability(self):
        """Two tokens with equal probability have entropy = 1.0 bit."""
        logprobs = [
            {"token": "system1", "logprob": math.log(0.5)},
            {"token": "system2", "logprob": math.log(0.5)},
        ]
        assert compute_entropy(logprobs) == pytest.approx(1.0, abs=0.01)

    def test_high_confidence_low_entropy(self):
        """One dominant token (p=0.95) should have low entropy."""
        logprobs = [
            {"token": "system2", "logprob": math.log(0.95)},
            {"token": "system1", "logprob": math.log(0.05)},
        ]
        result = compute_entropy(logprobs)
        assert result < 0.4

    def test_uncertain_high_entropy(self):
        """Close probabilities (p=0.6 vs 0.4) should have higher entropy."""
        logprobs = [
            {"token": "system1", "logprob": math.log(0.6)},
            {"token": "system2", "logprob": math.log(0.4)},
        ]
        result = compute_entropy(logprobs)
        assert result > 0.9

    def test_empty_logprobs_returns_zero(self):
        """Empty logprobs list should return 0.0."""
        assert compute_entropy([]) == 0.0

    def test_handles_negative_infinity_logprob(self):
        """Tokens with -inf logprob (p=0) should be skipped."""
        logprobs = [
            {"token": "system2", "logprob": 0.0},
            {"token": "system1", "logprob": float("-inf")},
        ]
        assert compute_entropy(logprobs) == pytest.approx(0.0, abs=0.001)


class TestFindRoutingTokenEntropy:
    def test_finds_system_token(self):
        """Should find entropy at the token position containing 'system'."""
        token_logprobs = [
            {
                "token": '{"',
                "logprob": -0.01,
                "top_logprobs": [{"token": '{"', "logprob": -0.01}],
            },
            {
                "token": "routing",
                "logprob": -0.02,
                "top_logprobs": [{"token": "routing", "logprob": -0.02}],
            },
            {
                "token": "system",
                "logprob": math.log(0.7),
                "top_logprobs": [
                    {"token": "system", "logprob": math.log(0.7)},
                    {"token": "complex", "logprob": math.log(0.3)},
                ],
            },
        ]
        entropy = find_routing_token_entropy(token_logprobs)
        assert entropy is not None
        assert entropy > 0.5

    def test_returns_none_when_no_routing_token(self):
        """Should return None if no token matches routing keywords."""
        token_logprobs = [
            {
                "token": "hello",
                "logprob": -0.01,
                "top_logprobs": [{"token": "hello", "logprob": -0.01}],
            },
        ]
        assert find_routing_token_entropy(token_logprobs) is None

    def test_returns_none_for_empty_list(self):
        """Should return None for empty token list."""
        assert find_routing_token_entropy([]) is None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_entropy.py -v`
Expected: FAIL — `metascaffold.entropy` module does not exist.

**Step 3: Implement `entropy.py`**

Create `src/metascaffold/entropy.py`:

```python
"""Entropy computation for token logprobs — measures LLM uncertainty.

Shannon entropy H = -sum(p * log2(p)) over the probability distribution
of top-k tokens at a given position. Used by the Classifier for
entropy-based routing.
"""

from __future__ import annotations

import math


def compute_entropy(logprobs: list[dict]) -> float:
    """Compute Shannon entropy (in bits) from a list of {token, logprob} dicts.

    Each logprob is a natural log probability (as returned by the OpenAI API).
    Converts to probabilities, normalizes, and computes H = -sum(p * log2(p)).

    Returns 0.0 for empty input or single-token distributions.
    """
    if not logprobs:
        return 0.0

    # Convert logprobs to probabilities
    probs = []
    for entry in logprobs:
        lp = entry.get("logprob", float("-inf"))
        if lp == float("-inf"):
            continue
        probs.append(math.exp(lp))

    if not probs:
        return 0.0

    # Normalize (top-k logprobs may not sum to 1.0)
    total = sum(probs)
    if total <= 0:
        return 0.0
    probs = [p / total for p in probs]

    # Shannon entropy in bits
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def find_routing_token_entropy(token_logprobs: list[dict]) -> float | None:
    """Find the entropy at the token position most relevant to routing.

    Scans through the token stream looking for tokens that match routing
    keywords (system1, system2, system, simple, complex). When found,
    computes entropy from that position's top_logprobs.

    Returns None if no routing token is found.
    """
    if not token_logprobs:
        return None

    routing_keywords = {"system", "system1", "system2", "simple", "complex"}

    for token_info in token_logprobs:
        token = token_info.get("token", "").lower().strip()
        if any(kw in token for kw in routing_keywords):
            top_lps = token_info.get("top_logprobs", [])
            if top_lps:
                return compute_entropy(top_lps)

    return None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_entropy.py -v`
Expected: ALL PASS

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS (120+ existing tests + new entropy tests)

**Step 6: Commit**

```bash
git add src/metascaffold/entropy.py tests/test_entropy.py
git commit -m "feat(v0.3): add Shannon entropy computation for logprobs"
```

---

### Task 3: Entropy-Based Classifier with 3-Level Routing

**Files:**
- Modify: `src/metascaffold/classifier.py`
- Modify: `tests/test_classifier.py`

**Context:** Replace the Classifier's LLM self-reported confidence with entropy from logprobs. Introduce 3-level routing: System 1 (low entropy), System 1.5 (medium), System 2 (high). Fast-paths for read-only and always_system2 tools remain unchanged. The entropy probe uses `complete_with_logprobs()` → gpt-4.1-nano. Falls back to heuristic classification if the API call fails.

**Step 1: Write failing tests for entropy-based classification**

Add to `tests/test_classifier.py`:

```python
import math
from unittest.mock import patch


class TestClassifierEntropy:
    """Tests for entropy-based 3-level routing (v0.3)."""

    @pytest.mark.asyncio
    async def test_low_entropy_routes_to_system1(self):
        """Low entropy (model confident) should route to system1."""
        llm_response = _FakeLLMResponse(
            content='{"routing": "system1", "confidence": 0.9, "reasoning": "simple"}',
        )
        # Add token_logprobs with very low entropy (one dominant token)
        llm_response.token_logprobs = [
            {
                "token": "system",
                "logprob": math.log(0.98),
                "top_logprobs": [
                    {"token": "system", "logprob": math.log(0.98)},
                    {"token": "complex", "logprob": math.log(0.02)},
                ],
            },
        ]
        mock_llm = _make_llm_client(enabled=True, response=llm_response)
        mock_llm.complete_with_logprobs = AsyncMock(return_value=llm_response)

        c = Classifier(llm_client=mock_llm, entropy_threshold=0.5)
        result = await c.classify_async(
            tool_name="Bash",
            tool_input={"command": "echo hello"},
            context="Print greeting",
        )
        assert result.routing == "system1"
        assert result.signals.get("compute_level") == 1

    @pytest.mark.asyncio
    async def test_high_entropy_routes_to_system2(self):
        """High entropy (model uncertain) should force system2."""
        llm_response = _FakeLLMResponse(
            content='{"routing": "system1", "confidence": 0.6, "reasoning": "maybe simple"}',
        )
        # Nearly equal probabilities → high entropy
        llm_response.token_logprobs = [
            {
                "token": "system",
                "logprob": math.log(0.55),
                "top_logprobs": [
                    {"token": "system", "logprob": math.log(0.55)},
                    {"token": "complex", "logprob": math.log(0.45)},
                ],
            },
        ]
        mock_llm = _make_llm_client(enabled=True, response=llm_response)
        mock_llm.complete_with_logprobs = AsyncMock(return_value=llm_response)

        c = Classifier(llm_client=mock_llm, entropy_threshold=0.5, medium_entropy_threshold=0.3)
        result = await c.classify_async(
            tool_name="Edit",
            tool_input={"file_path": "/src/app.py"},
            context="Change something",
        )
        assert result.routing == "system2"
        assert result.signals.get("compute_level") == 2

    @pytest.mark.asyncio
    async def test_medium_entropy_routes_to_system15(self):
        """Medium entropy should route to system1.5 (system1 routing, compute_level 1.5)."""
        llm_response = _FakeLLMResponse(
            content='{"routing": "system1", "confidence": 0.75, "reasoning": "moderately simple"}',
        )
        # Medium entropy: dominant but not overwhelming
        llm_response.token_logprobs = [
            {
                "token": "system",
                "logprob": math.log(0.82),
                "top_logprobs": [
                    {"token": "system", "logprob": math.log(0.82)},
                    {"token": "complex", "logprob": math.log(0.18)},
                ],
            },
        ]
        mock_llm = _make_llm_client(enabled=True, response=llm_response)
        mock_llm.complete_with_logprobs = AsyncMock(return_value=llm_response)

        c = Classifier(llm_client=mock_llm, entropy_threshold=0.5, medium_entropy_threshold=0.3)
        result = await c.classify_async(
            tool_name="Edit",
            tool_input={"file_path": "/src/utils.py"},
            context="Update utility",
        )
        assert result.signals.get("compute_level") == 1.5

    @pytest.mark.asyncio
    async def test_fallback_to_codex_when_logprobs_fail(self):
        """If logprobs API fails, fall back to codex exec classification."""
        # Logprobs call fails
        logprobs_response = _FakeLLMResponse(error="API key not set")
        logprobs_response.token_logprobs = []

        # Codex classification succeeds
        codex_response = _FakeLLMResponse(
            content='{"routing": "system2", "confidence": 0.85, "reasoning": "LLM said so"}',
        )

        mock_llm = _make_llm_client(enabled=True, response=codex_response)
        mock_llm.complete_with_logprobs = AsyncMock(return_value=logprobs_response)

        c = Classifier(llm_client=mock_llm, entropy_threshold=0.5)
        result = await c.classify_async(
            tool_name="Edit",
            tool_input={"file_path": "/src/app.py"},
            context="Complex refactor",
        )
        assert result.routing == "system2"
        assert result.signals.get("source") == "llm"
```

Also update `_FakeLLMResponse` to support `token_logprobs`:

```python
@dataclass
class _FakeLLMResponse:
    content: str = ""
    model: str = "fake-model"
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    token_logprobs: list = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_classifier.py::TestClassifierEntropy -v`
Expected: FAIL — Classifier doesn't accept `entropy_threshold`, doesn't have logprobs-based routing.

**Step 3: Implement entropy-based classification in `classifier.py`**

Update `Classifier.__init__()`:

```python
def __init__(
    self,
    system2_threshold: float = 0.8,
    always_system2_tools: list[str] | None = None,
    llm_client: object | None = None,
    entropy_threshold: float = 0.5,
    medium_entropy_threshold: float = 0.3,
):
    self.system2_threshold = system2_threshold
    self.always_system2_tools = always_system2_tools or []
    self._llm = llm_client
    self.entropy_threshold = entropy_threshold
    self.medium_entropy_threshold = medium_entropy_threshold
```

Add the entropy import at the top:

```python
from metascaffold.entropy import find_routing_token_entropy
```

Add a new `_entropy_classify` method and modify `classify_async`:

```python
async def classify_async(
    self,
    tool_name: str,
    tool_input: dict,
    context: str,
    historical_success_rate: float | None = None,
) -> ClassificationResult:
    # Fast-paths (unchanged)
    if tool_name in self.always_system2_tools:
        return ClassificationResult(
            routing="system2",
            confidence=0.5,
            reasoning=f"Tool '{tool_name}' is configured for mandatory System 2",
            signals={"source": "fast-path", "compute_level": 2},
        )
    if tool_name in _READ_ONLY_TOOLS:
        return ClassificationResult(
            routing="system1",
            confidence=0.95,
            reasoning=f"Read-only tool '{tool_name}'",
            signals={"source": "fast-path", "compute_level": 1},
        )

    # Try entropy-based classification (logprobs via OpenAI API)
    if getattr(self._llm, "enabled", False):
        entropy_result = await self._entropy_classify(tool_name, tool_input, context)
        if entropy_result is not None:
            return entropy_result

        # Fallback: codex exec LLM classification
        llm_result = await self._llm_classify(tool_name, tool_input, context)
        if llm_result is not None:
            return llm_result

    # Final fallback: heuristic
    logger.debug("Falling back to heuristic classification for %s", tool_name)
    return self._heuristic_classify(
        tool_name, tool_input, context, historical_success_rate,
    )

async def _entropy_classify(
    self,
    tool_name: str,
    tool_input: dict,
    context: str,
) -> ClassificationResult | None:
    """Entropy-based 3-level routing using logprobs from gpt-4.1-nano."""
    if not hasattr(self._llm, "complete_with_logprobs"):
        return None

    user_prompt = (
        f"Tool: {tool_name}\n"
        f"Input: {json.dumps(tool_input, default=str)}\n"
        f"Context: {context}"
    )

    try:
        response = await self._llm.complete_with_logprobs(
            model="gpt-4.1-nano",
            system_prompt=_CLASSIFIER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=256,
            top_logprobs=5,
        )
        if response.error:
            logger.warning("Entropy probe failed: %s", response.error)
            return None

        # Parse textual response for routing decision
        data = json.loads(response.content)
        routing = data.get("routing", "system2")
        confidence = float(data.get("confidence", 0.5))
        reasoning = data.get("reasoning", "entropy-based classification")

        if routing not in ("system1", "system2"):
            routing = "system2"

        # Compute entropy from logprobs
        entropy = find_routing_token_entropy(response.token_logprobs)

        if entropy is not None:
            if entropy > self.entropy_threshold:
                # High entropy → force System 2
                routing = "system2"
                compute_level = 2
                reasoning = f"Entropy override: H={entropy:.3f} > {self.entropy_threshold} → System 2. {reasoning}"
            elif entropy > self.medium_entropy_threshold:
                # Medium entropy → System 1.5
                compute_level = 1.5
                reasoning = f"Medium entropy: H={entropy:.3f} → System 1.5. {reasoning}"
            else:
                # Low entropy → trust the model's routing
                compute_level = 1 if routing == "system1" else 2
        else:
            # No routing token found → trust textual answer
            compute_level = 1 if routing == "system1" else 2

        return ClassificationResult(
            routing=routing,
            confidence=max(0.0, min(1.0, confidence)),
            reasoning=reasoning,
            signals={
                "source": "entropy",
                "entropy": entropy,
                "compute_level": compute_level,
            },
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Entropy classification parse error: %s", exc)
        return None
    except Exception as exc:
        logger.warning("Entropy classification failed: %s", exc)
        return None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: ALL PASS (existing + new entropy tests)

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/metascaffold/classifier.py tests/test_classifier.py
git commit -m "feat(v0.3): entropy-based 3-level routing (System 1/1.5/2)"
```

---

### Task 4: Deterministic Verification Suite

**Files:**
- Create: `src/metascaffold/verifiers.py`
- Create: `tests/test_verifiers.py`

**Context:** Create a `VerificationSuite` that runs 4 deterministic checks: AST parse, Ruff lint, pytest execution, and mypy type check. Each verifier is independent and produces a `VerifierResult`. The suite aggregates results. This runs BEFORE the LLM-as-Judge in the evaluator.

**Step 1: Write failing tests**

Create `tests/test_verifiers.py`:

```python
"""Tests for the deterministic verification suite."""

import ast
from unittest.mock import patch, MagicMock

import pytest

from metascaffold.verifiers import (
    VerifierResult,
    VerificationSuite,
    ast_verify,
    ruff_verify,
    pytest_verify,
    mypy_verify,
)


class TestAstVerify:
    def test_valid_python_passes(self):
        """Valid Python code should pass AST verification."""
        result = ast_verify("def foo():\n    return 42\n")
        assert result.passed is True
        assert result.verifier == "ast"

    def test_syntax_error_fails(self):
        """Invalid Python should fail AST verification."""
        result = ast_verify("def foo(\n")
        assert result.passed is False
        assert "SyntaxError" in result.detail or "syntax" in result.detail.lower()

    def test_empty_string_passes(self):
        """Empty string is valid Python."""
        result = ast_verify("")
        assert result.passed is True


class TestRuffVerify:
    @patch("metascaffold.verifiers.subprocess.run")
    def test_ruff_clean_passes(self, mock_run):
        """No ruff errors should pass."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        result = ruff_verify("/tmp/test.py")
        assert result.passed is True
        assert result.verifier == "ruff"

    @patch("metascaffold.verifiers.subprocess.run")
    def test_ruff_errors_fail(self, mock_run):
        """Ruff errors should fail."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="test.py:1:1: F401 `os` imported but unused",
            stderr="",
        )
        result = ruff_verify("/tmp/test.py")
        assert result.passed is False
        assert "F401" in result.detail

    @patch("metascaffold.verifiers.subprocess.run")
    def test_ruff_not_installed(self, mock_run):
        """Missing ruff binary should return skipped result."""
        mock_run.side_effect = FileNotFoundError("ruff not found")
        result = ruff_verify("/tmp/test.py")
        assert result.passed is True  # Skipped = not a failure
        assert result.skipped is True


class TestPytestVerify:
    @patch("metascaffold.verifiers.subprocess.run")
    def test_all_tests_pass(self, mock_run):
        """All tests passing should pass verification."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="5 passed in 1.2s",
            stderr="",
        )
        result = pytest_verify("tests/")
        assert result.passed is True

    @patch("metascaffold.verifiers.subprocess.run")
    def test_test_failures(self, mock_run):
        """Test failures should fail verification."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="3 passed, 2 failed",
            stderr="FAILED test_auth",
        )
        result = pytest_verify("tests/")
        assert result.passed is False
        assert "2 failed" in result.detail or "FAILED" in result.detail


class TestMypyVerify:
    @patch("metascaffold.verifiers.subprocess.run")
    def test_mypy_clean(self, mock_run):
        """No mypy errors should pass."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Success: no issues found",
            stderr="",
        )
        result = mypy_verify("/tmp/test.py")
        assert result.passed is True

    @patch("metascaffold.verifiers.subprocess.run")
    def test_mypy_errors(self, mock_run):
        """Type errors should fail."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="test.py:5: error: Incompatible types",
            stderr="",
        )
        result = mypy_verify("/tmp/test.py")
        assert result.passed is False


class TestVerificationSuite:
    def test_suite_with_passing_code(self):
        """Suite should pass when code is syntactically valid."""
        suite = VerificationSuite()
        results = suite.verify_code("def foo():\n    return 42\n")
        # At minimum, AST should pass
        ast_results = [r for r in results if r.verifier == "ast"]
        assert len(ast_results) == 1
        assert ast_results[0].passed is True

    def test_suite_with_broken_syntax(self):
        """Suite should fail AST check on invalid code."""
        suite = VerificationSuite()
        results = suite.verify_code("def foo(\n")
        ast_results = [r for r in results if r.verifier == "ast"]
        assert len(ast_results) == 1
        assert ast_results[0].passed is False

    def test_suite_has_critical_failures(self):
        """has_critical_failures should be True when AST fails."""
        suite = VerificationSuite()
        results = suite.verify_code("def foo(\n")
        assert suite.has_critical_failures(results) is True

    def test_suite_no_critical_on_valid_code(self):
        """has_critical_failures should be False on valid code."""
        suite = VerificationSuite()
        results = suite.verify_code("x = 1\n")
        assert suite.has_critical_failures(results) is False
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_verifiers.py -v`
Expected: FAIL — `metascaffold.verifiers` does not exist.

**Step 3: Implement `verifiers.py`**

Create `src/metascaffold/verifiers.py`:

```python
"""Deterministic verification suite — AST, Ruff, pytest, mypy.

Runs before LLM-as-Judge in the evaluator. Provides ground truth
that anchors the LLM evaluation.
"""

from __future__ import annotations

import ast
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("metascaffold.verifiers")


@dataclass
class VerifierResult:
    """Result from a single deterministic verifier."""
    verifier: str       # "ast", "ruff", "pytest", "mypy"
    passed: bool
    detail: str = ""
    severity: str = "info"   # "info", "warning", "critical"
    skipped: bool = False

    def to_dict(self) -> dict:
        return {
            "verifier": self.verifier,
            "passed": self.passed,
            "detail": self.detail,
            "severity": self.severity,
            "skipped": self.skipped,
        }


def ast_verify(code: str) -> VerifierResult:
    """Check Python syntax by parsing the AST."""
    try:
        ast.parse(code)
        return VerifierResult(verifier="ast", passed=True, detail="Syntax OK")
    except SyntaxError as e:
        return VerifierResult(
            verifier="ast",
            passed=False,
            detail=f"SyntaxError at line {e.lineno}: {e.msg}",
            severity="critical",
        )


def ruff_verify(file_path: str, timeout: int = 15) -> VerifierResult:
    """Run Ruff linter on a file."""
    try:
        proc = subprocess.run(
            ["ruff", "check", file_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0:
            return VerifierResult(verifier="ruff", passed=True, detail="No lint errors")
        return VerifierResult(
            verifier="ruff",
            passed=False,
            detail=proc.stdout[:500].strip(),
            severity="warning",
        )
    except FileNotFoundError:
        return VerifierResult(
            verifier="ruff", passed=True, detail="ruff not installed", skipped=True,
        )
    except subprocess.TimeoutExpired:
        return VerifierResult(
            verifier="ruff", passed=True, detail="ruff timed out", skipped=True,
        )


def pytest_verify(test_path: str, timeout: int = 60) -> VerifierResult:
    """Run pytest on a test path."""
    try:
        proc = subprocess.run(
            ["python", "-m", "pytest", test_path, "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout + proc.stderr
        if proc.returncode == 0:
            return VerifierResult(verifier="pytest", passed=True, detail=output[:300].strip())
        return VerifierResult(
            verifier="pytest",
            passed=False,
            detail=output[:500].strip(),
            severity="critical",
        )
    except FileNotFoundError:
        return VerifierResult(
            verifier="pytest", passed=True, detail="pytest not found", skipped=True,
        )
    except subprocess.TimeoutExpired:
        return VerifierResult(
            verifier="pytest",
            passed=False,
            detail=f"pytest timed out after {timeout}s",
            severity="warning",
        )


def mypy_verify(file_path: str, timeout: int = 30) -> VerifierResult:
    """Run mypy type checker on a file."""
    try:
        proc = subprocess.run(
            ["python", "-m", "mypy", file_path, "--ignore-missing-imports"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0:
            return VerifierResult(verifier="mypy", passed=True, detail="No type errors")
        return VerifierResult(
            verifier="mypy",
            passed=False,
            detail=proc.stdout[:500].strip(),
            severity="warning",
        )
    except FileNotFoundError:
        return VerifierResult(
            verifier="mypy", passed=True, detail="mypy not installed", skipped=True,
        )
    except subprocess.TimeoutExpired:
        return VerifierResult(
            verifier="mypy", passed=True, detail="mypy timed out", skipped=True,
        )


class VerificationSuite:
    """Aggregates deterministic verifiers."""

    def verify_code(self, code: str) -> list[VerifierResult]:
        """Run AST verification on inline code. Returns list of results."""
        results = [ast_verify(code)]
        return results

    def verify_file(
        self,
        file_path: str,
        run_ruff: bool = True,
        run_mypy: bool = False,
    ) -> list[VerifierResult]:
        """Run verifiers on a file. Returns list of results."""
        results = []

        # Read file for AST check
        try:
            code = Path(file_path).read_text(encoding="utf-8")
            results.append(ast_verify(code))
        except (OSError, UnicodeDecodeError) as e:
            results.append(VerifierResult(
                verifier="ast", passed=True, detail=f"Could not read file: {e}", skipped=True,
            ))

        if run_ruff:
            results.append(ruff_verify(file_path))

        if run_mypy:
            results.append(mypy_verify(file_path))

        return results

    def verify_tests(self, test_path: str) -> VerifierResult:
        """Run pytest on a test path."""
        return pytest_verify(test_path)

    @staticmethod
    def has_critical_failures(results: list[VerifierResult]) -> bool:
        """Check if any verifier reported a critical failure."""
        return any(
            not r.passed and r.severity == "critical" and not r.skipped
            for r in results
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_verifiers.py -v`
Expected: ALL PASS

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/metascaffold/verifiers.py tests/test_verifiers.py
git commit -m "feat(v0.3): add deterministic verification suite (AST, Ruff, pytest, mypy)"
```

---

### Task 5: Integrate Verifiers into Evaluator

**Files:**
- Modify: `src/metascaffold/evaluator.py`
- Modify: `tests/test_evaluator.py`

**Context:** The evaluator should run deterministic verifiers BEFORE calling LLM-as-Judge. If verifiers find critical failures (e.g., AST parse error), return "backtrack" immediately without wasting an LLM call. Verifier findings are included in the LLM prompt context for richer evaluation.

**Step 1: Write failing tests**

Add to `tests/test_evaluator.py`:

```python
from metascaffold.verifiers import VerificationSuite, VerifierResult


class TestEvaluatorWithVerifiers:
    """Tests for deterministic verifiers integration in evaluator."""

    @pytest.mark.asyncio
    async def test_ast_failure_returns_backtrack_without_llm(self):
        """When code has syntax error, evaluator should backtrack without calling LLM."""
        mock_client = AsyncMock()
        mock_client.enabled = True

        evaluator = Evaluator(max_retry_attempts=3, llm_client=mock_client)
        result = await evaluator.evaluate_async(
            sandbox_result=SandboxResult(
                exit_code=1,
                stdout="",
                stderr="SyntaxError: unexpected EOF while parsing",
                duration_ms=100,
            ),
            attempt=1,
            code_output="def foo(\n",  # Broken syntax
        )
        assert result.verdict == "backtrack"
        # LLM should NOT have been called
        mock_client.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_valid_code_proceeds_to_llm(self):
        """When code passes verifiers, LLM-as-Judge should still be called."""
        llm_response = {
            "verdict": "pass",
            "confidence": 0.9,
            "feedback": {
                "failing_tests": [],
                "error_lines": [],
                "root_cause": "",
                "suggested_fix": "",
            },
            "adversarial_findings": [],
            "revision_allowed": True,
        }
        mock_client = AsyncMock()
        mock_client.enabled = True
        mock_client.complete = AsyncMock(return_value=MagicMock(
            content=json.dumps(llm_response),
            error="",
        ))

        evaluator = Evaluator(max_retry_attempts=3, llm_client=mock_client)
        result = await evaluator.evaluate_async(
            sandbox_result=SandboxResult(
                exit_code=0,
                stdout="All tests passed",
                stderr="",
                duration_ms=500,
            ),
            attempt=1,
            code_output="def foo():\n    return 42\n",
        )
        assert result.verdict == "pass"
        mock_client.complete.assert_awaited_once()

    def test_heuristic_evaluation_unchanged_without_code(self):
        """Heuristic evaluation should still work when no code_output provided."""
        evaluator = Evaluator(max_retry_attempts=3)
        result = evaluator.evaluate(
            sandbox_result=SandboxResult(
                exit_code=0, stdout="ok", stderr="", duration_ms=100,
            ),
            attempt=1,
        )
        assert result.verdict == "pass"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluator.py::TestEvaluatorWithVerifiers -v`
Expected: FAIL — `evaluate_async()` doesn't accept `code_output` parameter.

**Step 3: Modify evaluator to integrate verifiers**

In `src/metascaffold/evaluator.py`, add import:

```python
from metascaffold.verifiers import VerificationSuite
```

Update `evaluate_async()` signature and logic:

```python
async def evaluate_async(
    self,
    sandbox_result: SandboxResult,
    attempt: int = 1,
    code_output: str | None = None,
) -> EvaluationResult:
    """Async evaluation with deterministic verifiers + LLM-as-Judge.

    1. If code_output provided, run deterministic verifiers first
    2. If verifiers find critical failures → return "backtrack" immediately
    3. Otherwise, proceed to LLM-as-Judge (or heuristic fallback)
    """
    # Step 1: Deterministic verification
    verifier_findings = []
    if code_output is not None:
        suite = VerificationSuite()
        verifier_results = suite.verify_code(code_output)
        verifier_findings = [r.to_dict() for r in verifier_results]

        if suite.has_critical_failures(verifier_results):
            critical = [r for r in verifier_results if not r.passed and r.severity == "critical"]
            detail = "; ".join(r.detail for r in critical)
            return EvaluationResult(
                verdict="backtrack",
                confidence=0.95,
                issues=[Issue(type="verifier", detail=detail, severity="critical")],
                attempt=attempt,
                max_attempts=self.max_retry_attempts,
                feedback={"verifier_failures": verifier_findings},
            )

    # Step 2: LLM-as-Judge (with verifier context)
    if self._llm_client and self._llm_client.enabled:
        llm_result = await self._llm_evaluate(
            sandbox_result, attempt, verifier_findings,
        )
        if llm_result is not None:
            return llm_result
        logger.warning("LLM evaluation failed, falling back to heuristics")

    return self._heuristic_evaluate(sandbox_result, attempt)
```

Update `_llm_evaluate` to accept and include verifier findings:

```python
async def _llm_evaluate(
    self,
    sandbox_result: SandboxResult,
    attempt: int = 1,
    verifier_findings: list[dict] | None = None,
) -> EvaluationResult | None:
```

And include verifier findings in the user prompt:

```python
user_prompt = (
    f"exit_code: {sandbox_result.exit_code}\n"
    f"timed_out: {sandbox_result.timed_out}\n"
    f"duration_ms: {sandbox_result.duration_ms}\n"
    f"attempt: {attempt} / {self.max_retry_attempts}\n"
    f"\n--- stdout (truncated) ---\n{sandbox_result.stdout[:2000]}\n"
    f"\n--- stderr (truncated) ---\n{sandbox_result.stderr[:2000]}"
)
if verifier_findings:
    import json as _json
    user_prompt += f"\n\n--- deterministic verifiers ---\n{_json.dumps(verifier_findings, indent=2)}"
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluator.py -v`
Expected: ALL PASS (all existing tests + new verifier integration tests)

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/metascaffold/evaluator.py tests/test_evaluator.py
git commit -m "feat(v0.3): integrate deterministic verifiers into evaluator pipeline"
```

---

### Task 6: Reflection Memory with Ebbinghaus Decay

**Files:**
- Create: `src/metascaffold/reflection_memory.py`
- Create: `tests/test_reflection_memory.py`

**Context:** Create a `ReflectionMemory` that stores rules with retention strength and applies Ebbinghaus forgetting curve decay. Rules that are reinforced resist decay; rules that aren't used fade and eventually get pruned. Storage is JSON file for hot rules, with SQLite archival.

**Step 1: Write failing tests**

Create `tests/test_reflection_memory.py`:

```python
"""Tests for the Ebbinghaus-decay reflection memory."""

import json
import math
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from metascaffold.reflection_memory import ReflectionMemory, ReflectionRule


class TestReflectionRule:
    def test_default_retention_is_one(self):
        """New rules should have full retention strength."""
        rule = ReflectionRule(content="Always run tests")
        assert rule.retention_strength == 1.0

    def test_compute_retention_decays_over_time(self):
        """Retention should decay with time since last reinforcement."""
        past = datetime.now(timezone.utc) - timedelta(hours=168)  # 1 week ago
        rule = ReflectionRule(
            content="Test rule",
            last_reinforced=past,
            reinforcement_count=0,
        )
        retention = rule.compute_retention()
        assert 0.0 < retention < 1.0

    def test_reinforced_rule_decays_slower(self):
        """A reinforced rule should have higher retention than unreinforced."""
        past = datetime.now(timezone.utc) - timedelta(hours=168)
        unreinforced = ReflectionRule(content="A", last_reinforced=past, reinforcement_count=0)
        reinforced = ReflectionRule(content="B", last_reinforced=past, reinforcement_count=5)
        assert reinforced.compute_retention() > unreinforced.compute_retention()

    def test_fresh_rule_has_full_retention(self):
        """A just-created rule should have retention ~1.0."""
        rule = ReflectionRule(content="New rule")
        assert rule.compute_retention() == pytest.approx(1.0, abs=0.01)

    def test_to_dict_roundtrip(self):
        """Rule should serialize and deserialize correctly."""
        rule = ReflectionRule(content="Test", reinforcement_count=3)
        d = rule.to_dict()
        restored = ReflectionRule.from_dict(d)
        assert restored.content == "Test"
        assert restored.reinforcement_count == 3


class TestReflectionMemory:
    def test_add_rule(self):
        """Adding a rule should store it in memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ReflectionMemory(storage_path=Path(tmpdir) / "memory.json")
            mem.add_rule("Always run tests after edits")
            assert len(mem.rules) == 1
            assert mem.rules[0].content == "Always run tests after edits"

    def test_reinforce_existing_rule(self):
        """Reinforcing a rule should increment its count and refresh timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ReflectionMemory(storage_path=Path(tmpdir) / "memory.json")
            mem.add_rule("Use TDD")
            old_count = mem.rules[0].reinforcement_count
            mem.reinforce("Use TDD")
            assert mem.rules[0].reinforcement_count == old_count + 1

    def test_prune_removes_decayed_rules(self):
        """Prune should remove rules with low retention."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ReflectionMemory(
                storage_path=Path(tmpdir) / "memory.json",
                prune_threshold=0.5,
            )
            # Add a rule with old timestamp (will have decayed)
            old_time = datetime.now(timezone.utc) - timedelta(days=30)
            rule = ReflectionRule(content="Old rule", last_reinforced=old_time)
            mem.rules.append(rule)
            mem.prune()
            assert len(mem.rules) == 0

    def test_save_and_load(self):
        """Memory should persist to JSON and reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.json"
            mem = ReflectionMemory(storage_path=path)
            mem.add_rule("Rule 1")
            mem.add_rule("Rule 2")
            mem.save()

            mem2 = ReflectionMemory(storage_path=path)
            mem2.load()
            assert len(mem2.rules) == 2
            assert mem2.rules[0].content == "Rule 1"

    def test_get_active_rules_filters_by_retention(self):
        """get_active_rules should only return rules above threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ReflectionMemory(storage_path=Path(tmpdir) / "m.json")
            mem.add_rule("Fresh rule")
            old_time = datetime.now(timezone.utc) - timedelta(days=60)
            mem.rules.append(ReflectionRule(content="Dead rule", last_reinforced=old_time))
            active = mem.get_active_rules(min_retention=0.3)
            assert len(active) == 1
            assert active[0].content == "Fresh rule"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_reflection_memory.py -v`
Expected: FAIL — `metascaffold.reflection_memory` does not exist.

**Step 3: Implement `reflection_memory.py`**

Create `src/metascaffold/reflection_memory.py`:

```python
"""Reflection memory with Ebbinghaus forgetting curve decay.

Rules that are reinforced (confirmed useful) resist decay.
Rules that are not used gradually fade and get pruned.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("metascaffold.reflection_memory")

# Default half-life in hours (1 week)
_DEFAULT_STABILITY_HOURS = 168.0


@dataclass
class ReflectionRule:
    """A learned rule with Ebbinghaus decay tracking."""
    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_reinforced: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retention_strength: float = 1.0
    reinforcement_count: int = 0
    source_events: list[str] = field(default_factory=list)

    def compute_retention(self, stability_hours: float = _DEFAULT_STABILITY_HOURS) -> float:
        """Compute current retention using Ebbinghaus forgetting curve.

        retention = e^(-t / (stability * reinforcement_factor))
        - t: hours since last reinforcement
        - stability: base half-life in hours
        - reinforcement_factor: 1 + log(1 + reinforcement_count)
        """
        now = datetime.now(timezone.utc)
        t_hours = (now - self.last_reinforced).total_seconds() / 3600.0
        reinforcement_factor = 1.0 + math.log(1.0 + self.reinforcement_count)
        denominator = stability_hours * reinforcement_factor
        if denominator <= 0:
            return 0.0
        return math.exp(-t_hours / denominator)

    def reinforce(self) -> None:
        """Mark this rule as confirmed useful — resets decay timer."""
        self.last_reinforced = datetime.now(timezone.utc)
        self.reinforcement_count += 1
        self.retention_strength = 1.0

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "last_reinforced": self.last_reinforced.isoformat(),
            "retention_strength": self.retention_strength,
            "reinforcement_count": self.reinforcement_count,
            "source_events": self.source_events,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReflectionRule:
        return cls(
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_reinforced=datetime.fromisoformat(data["last_reinforced"]),
            retention_strength=data.get("retention_strength", 1.0),
            reinforcement_count=data.get("reinforcement_count", 0),
            source_events=data.get("source_events", []),
        )


class ReflectionMemory:
    """Manages reflection rules with Ebbinghaus decay."""

    def __init__(
        self,
        storage_path: Path | None = None,
        prune_threshold: float = 0.1,
        stability_hours: float = _DEFAULT_STABILITY_HOURS,
    ):
        self.storage_path = storage_path or (Path.home() / ".metascaffold" / "reflection_memory.json")
        self.prune_threshold = prune_threshold
        self.stability_hours = stability_hours
        self.rules: list[ReflectionRule] = []

    def add_rule(self, content: str, source_events: list[str] | None = None) -> ReflectionRule:
        """Add a new rule to memory."""
        rule = ReflectionRule(content=content, source_events=source_events or [])
        self.rules.append(rule)
        return rule

    def reinforce(self, content: str) -> bool:
        """Reinforce a rule by content match. Returns True if found."""
        for rule in self.rules:
            if rule.content == content:
                rule.reinforce()
                return True
        return False

    def prune(self) -> list[ReflectionRule]:
        """Remove rules whose retention has fallen below threshold.

        Returns the list of pruned rules (for archival).
        """
        pruned = []
        remaining = []
        for rule in self.rules:
            retention = rule.compute_retention(self.stability_hours)
            rule.retention_strength = retention
            if retention < self.prune_threshold:
                pruned.append(rule)
            else:
                remaining.append(rule)
        self.rules = remaining
        if pruned:
            logger.info("Pruned %d rules below retention threshold %.2f", len(pruned), self.prune_threshold)
        return pruned

    def get_active_rules(self, min_retention: float = 0.3) -> list[ReflectionRule]:
        """Return rules with retention above min_retention."""
        active = []
        for rule in self.rules:
            retention = rule.compute_retention(self.stability_hours)
            if retention >= min_retention:
                active.append(rule)
        return active

    def save(self) -> None:
        """Persist rules to JSON file."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = [rule.to_dict() for rule in self.rules]
        self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self) -> None:
        """Load rules from JSON file."""
        if not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            self.rules = [ReflectionRule.from_dict(d) for d in data]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load reflection memory: %s", e)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_reflection_memory.py -v`
Expected: ALL PASS

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/metascaffold/reflection_memory.py tests/test_reflection_memory.py
git commit -m "feat(v0.3): add Ebbinghaus decay reflection memory"
```

---

### Task 7: Integrate Reflection Memory into Reflector

**Files:**
- Modify: `src/metascaffold/reflector.py`
- Modify: `tests/test_reflector.py`

**Context:** The Reflector should use `ReflectionMemory` to store/reinforce rules from LLM reflection. When reflecting, it loads existing rules, compares with new LLM-extracted rules, reinforces matches, and adds new ones. Pruning happens at the end.

**Step 1: Write failing tests**

Add to `tests/test_reflector.py`:

```python
import tempfile
from pathlib import Path


class TestReflectorWithMemory:
    """Tests for memory-integrated reflection."""

    @pytest.mark.asyncio
    async def test_reflect_stores_rules_in_memory(self):
        """New rules from LLM should be stored in reflection memory."""
        mock_client = AsyncMock()
        mock_client.enabled = True
        mock_client.complete = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "rules": ["Always run tests after modifying shared code"],
                "procedures": ["Use TDD for new features"],
            }),
            error="",
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = Path(tmpdir) / "memory.json"
            reflector = Reflector(llm_client=mock_client, memory_path=mem_path)
            result = await reflector.reflect([{"event": "test"}])

            assert len(result.rules) == 1
            assert reflector.memory.rules[0].content == "Always run tests after modifying shared code"

    @pytest.mark.asyncio
    async def test_reflect_reinforces_existing_rules(self):
        """Rules that appear again should be reinforced, not duplicated."""
        mock_client = AsyncMock()
        mock_client.enabled = True
        mock_client.complete = AsyncMock(return_value=MagicMock(
            content=json.dumps({
                "rules": ["Use TDD"],
                "procedures": [],
            }),
            error="",
        ))

        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = Path(tmpdir) / "memory.json"
            reflector = Reflector(llm_client=mock_client, memory_path=mem_path)

            # First reflection: adds the rule
            await reflector.reflect([{"event": "test1"}])
            assert len(reflector.memory.rules) == 1
            assert reflector.memory.rules[0].reinforcement_count == 0

            # Second reflection: reinforces the rule
            await reflector.reflect([{"event": "test2"}])
            assert len(reflector.memory.rules) == 1
            assert reflector.memory.rules[0].reinforcement_count == 1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_reflector.py::TestReflectorWithMemory -v`
Expected: FAIL — `Reflector` doesn't accept `memory_path`.

**Step 3: Modify `reflector.py`**

Add import:

```python
from metascaffold.reflection_memory import ReflectionMemory
from pathlib import Path
```

Update `__init__` and `_llm_reflect`:

```python
class Reflector:
    """Analyzes telemetry to extract reusable rules and procedures."""

    def __init__(
        self,
        llm_client: object | None = None,
        memory_path: Path | None = None,
    ):
        self._llm = llm_client
        self.memory = ReflectionMemory(storage_path=memory_path)
        self.memory.load()

    async def reflect(self, events: list[dict]) -> ReflectionResult:
        if not events:
            return ReflectionResult(source_event_count=0)

        if self._llm and getattr(self._llm, "enabled", False):
            result = await self._llm_reflect(events)
            if result is not None:
                # Store/reinforce rules in memory
                for rule_text in result.rules:
                    if not self.memory.reinforce(rule_text):
                        self.memory.add_rule(rule_text)
                self.memory.prune()
                self.memory.save()
                return result

        return ReflectionResult(source_event_count=len(events))
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_reflector.py -v`
Expected: ALL PASS

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/metascaffold/reflector.py tests/test_reflector.py
git commit -m "feat(v0.3): integrate Ebbinghaus memory into reflector"
```

---

### Task 8: Adaptive Compute in Pipeline (System 1/1.5/2)

**Files:**
- Modify: `src/metascaffold/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Context:** Add `compute_level` to `PipelineState`. The pipeline uses this to decide which stages to run: System 1 skips Distill+Plan, System 1.5 skips Plan but runs Distill, System 2 runs all stages.

**Step 1: Write failing tests**

Add to `tests/test_pipeline.py`:

```python
class TestAdaptiveCompute:
    """Tests for 3-level adaptive compute (System 1/1.5/2)."""

    def test_compute_level_defaults_to_none(self):
        """New state should have no compute_level."""
        state = PipelineState(task="test", context="")
        assert state.compute_level is None

    def test_system1_bypasses_distill_and_plan(self):
        """System 1 should bypass both distill and plan."""
        state = PipelineState(
            task="test", context="",
            classification=MagicMock(routing="system1"),
            compute_level=1,
        )
        assert state.should_bypass_distill is True
        assert state.should_bypass_plan is True

    def test_system15_runs_distill_skips_plan(self):
        """System 1.5 should run distill but skip plan."""
        state = PipelineState(
            task="test", context="",
            classification=MagicMock(routing="system1"),
            compute_level=1.5,
        )
        assert state.should_bypass_distill is False
        assert state.should_bypass_plan is True

    def test_system2_runs_all(self):
        """System 2 should run all stages."""
        state = PipelineState(
            task="test", context="",
            classification=MagicMock(routing="system2"),
            compute_level=2,
        )
        assert state.should_bypass_distill is False
        assert state.should_bypass_plan is False

    async def test_classify_stage_sets_compute_level(self):
        """Classify stage should extract compute_level from signals."""
        mock_classifier = AsyncMock()
        mock_classifier.classify_async = AsyncMock(return_value=MagicMock(
            routing="system1",
            confidence=0.95,
            reasoning="Low entropy",
            signals={"compute_level": 1.5, "source": "entropy"},
        ))

        pipeline = CognitivePipeline(classifier=mock_classifier)
        state = await pipeline.classify_stage(PipelineState(task="test", context=""))
        assert state.compute_level == 1.5

    async def test_distill_skipped_for_system1(self):
        """Distill should be skipped when compute_level=1."""
        mock_distiller = AsyncMock()
        mock_distiller.distill = AsyncMock()

        state = PipelineState(
            task="test", context="",
            classification=MagicMock(routing="system1"),
            compute_level=1,
        )
        pipeline = CognitivePipeline(distiller=mock_distiller)
        result = await pipeline.distill_stage(state)
        mock_distiller.distill.assert_not_awaited()

    async def test_distill_runs_for_system15(self):
        """Distill should run when compute_level=1.5."""
        mock_template = MagicMock(objective="Structured")
        mock_distiller = AsyncMock()
        mock_distiller.distill = AsyncMock(return_value=mock_template)

        state = PipelineState(
            task="test", context="",
            classification=MagicMock(routing="system1"),
            compute_level=1.5,
        )
        pipeline = CognitivePipeline(distiller=mock_distiller)
        result = await pipeline.distill_stage(state)
        mock_distiller.distill.assert_awaited_once()
        assert result.template is not None

    async def test_plan_skipped_for_system15(self):
        """Plan should be skipped when compute_level=1.5."""
        mock_planner = AsyncMock()
        mock_planner.create_plan_async = AsyncMock()

        state = PipelineState(
            task="test", context="",
            classification=MagicMock(routing="system1"),
            compute_level=1.5,
        )
        pipeline = CognitivePipeline(planner=mock_planner)
        result = await pipeline.plan_stage(state)
        mock_planner.create_plan_async.assert_not_awaited()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py::TestAdaptiveCompute -v`
Expected: FAIL — `PipelineState` has no `compute_level`, `should_bypass_distill`, `should_bypass_plan`.

**Step 3: Modify `pipeline.py`**

Update `PipelineState`:

```python
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
    compute_level: float | None = None  # 1, 1.5, or 2

    @property
    def should_bypass(self) -> bool:
        """True if classification says System 1 (legacy compat)."""
        return self.should_bypass_distill and self.should_bypass_plan

    @property
    def should_bypass_distill(self) -> bool:
        """True if compute_level is 1 (System 1 — skip distill+plan)."""
        if self.compute_level is not None:
            return self.compute_level <= 1
        if self.classification is None:
            return False
        return getattr(self.classification, "routing", "") == "system1"

    @property
    def should_bypass_plan(self) -> bool:
        """True if compute_level <= 1.5 (System 1 or 1.5 — skip plan)."""
        if self.compute_level is not None:
            return self.compute_level < 2
        if self.classification is None:
            return False
        return getattr(self.classification, "routing", "") == "system1"

    @property
    def should_escalate(self) -> bool:
        return self.attempt >= self.max_attempts

    def with_retry(self) -> PipelineState:
        return replace(
            self,
            attempt=self.attempt + 1,
            execution=None,
            evaluation=None,
        )

    def to_dict(self) -> dict:
        def _safe_dict(obj):
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
            "compute_level": self.compute_level,
            "classification": _safe_dict(self.classification),
            "template": _safe_dict(self.template),
            "plan": _safe_dict(self.plan),
            "evaluation": _safe_dict(self.evaluation),
            "reflection": _safe_dict(self.reflection),
        }
```

Update `classify_stage` to extract compute_level from signals:

```python
async def classify_stage(self, state: PipelineState) -> PipelineState:
    """Stage 1: Classify and set compute level."""
    if self.classifier is None:
        return state
    result = await self.classifier.classify_async(
        tool_name="pipeline",
        tool_input={},
        context=state.task + " " + state.context,
    )
    compute_level = None
    if hasattr(result, "signals") and isinstance(result.signals, dict):
        compute_level = result.signals.get("compute_level")
    return replace(state, classification=result, compute_level=compute_level)
```

Update `distill_stage` and `plan_stage`:

```python
async def distill_stage(self, state: PipelineState) -> PipelineState:
    if state.should_bypass_distill or self.distiller is None:
        return state
    template = await self.distiller.distill(state.task, state.context)
    return replace(state, template=template)

async def plan_stage(self, state: PipelineState) -> PipelineState:
    if state.should_bypass_plan or self.planner is None:
        return state
    task_text = state.task
    if state.template and hasattr(state.template, "objective"):
        task_text = state.template.objective
    plan = await self.planner.create_plan_async(task=task_text, context=state.context)
    return replace(state, plan=plan)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: ALL PASS (existing tests need checking — `should_bypass` still works via the new properties)

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/metascaffold/pipeline.py tests/test_pipeline.py
git commit -m "feat(v0.3): adaptive compute — System 1/1.5/2 pipeline routing"
```

---

### Task 9: Update Configuration for v0.3

**Files:**
- Modify: `src/metascaffold/config.py`
- Modify: `config/default_config.toml`
- Modify: `tests/test_config.py`

**Context:** Add new config sections for entropy thresholds, verifier settings, and memory config. These are used by the Classifier, Evaluator, and Reflector respectively.

**Step 1: Write failing tests**

Add to `tests/test_config.py`:

```python
class TestV03Config:
    def test_entropy_config_has_defaults(self):
        """EntropyConfig should have sensible defaults."""
        from metascaffold.config import ClassifierConfig
        cfg = ClassifierConfig()
        assert cfg.entropy_threshold == 0.5
        assert cfg.medium_entropy_threshold == 0.3

    def test_verifier_config_has_defaults(self):
        """VerifierConfig should have sensible defaults."""
        from metascaffold.config import VerifierConfig
        cfg = VerifierConfig()
        assert cfg.run_ast is True
        assert cfg.run_ruff is True
        assert cfg.run_mypy is False
        assert cfg.run_pytest is False

    def test_memory_config_has_defaults(self):
        """MemoryConfig should have sensible defaults."""
        from metascaffold.config import MemoryConfig
        cfg = MemoryConfig()
        assert cfg.prune_threshold == 0.1
        assert cfg.stability_hours == 168.0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py::TestV03Config -v`
Expected: FAIL — missing `entropy_threshold`, `VerifierConfig`, `MemoryConfig`.

**Step 3: Update `config.py`**

Add `entropy_threshold` and `medium_entropy_threshold` to `ClassifierConfig`:

```python
@dataclass
class ClassifierConfig:
    system2_threshold: float = 0.8
    always_system2_tools: list[str] = field(default_factory=lambda: ["Write"])
    entropy_threshold: float = 0.5
    medium_entropy_threshold: float = 0.3
```

Add new config dataclasses:

```python
@dataclass
class VerifierConfig:
    run_ast: bool = True
    run_ruff: bool = True
    run_mypy: bool = False
    run_pytest: bool = False
    ruff_timeout: int = 15
    mypy_timeout: int = 30
    pytest_timeout: int = 60


@dataclass
class MemoryConfig:
    prune_threshold: float = 0.1
    stability_hours: float = 168.0
    storage_path: str = ""
```

Add to `MetaScaffoldConfig`:

```python
@dataclass
class MetaScaffoldConfig:
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    notebooklm: NotebookLMConfig = field(default_factory=NotebookLMConfig)
    mcp_server: McpServerConfig = field(default_factory=McpServerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
```

Update `_dict_to_config` to handle new sections:

```python
if "verifier" in data:
    cfg.verifier = VerifierConfig(**data["verifier"])

if "memory" in data:
    m = data["memory"]
    cfg.memory = MemoryConfig(
        prune_threshold=m.get("prune_threshold", 0.1),
        stability_hours=m.get("stability_hours", 168.0),
        storage_path=_expand_path(m.get("storage_path", "~/.metascaffold/reflection_memory.json")),
    )
else:
    cfg.memory = MemoryConfig(
        storage_path=_expand_path("~/.metascaffold/reflection_memory.json"),
    )
```

Update `config/default_config.toml`:

```toml
[verifier]
run_ast = true
run_ruff = true
run_mypy = false
run_pytest = false
ruff_timeout = 15
mypy_timeout = 30
pytest_timeout = 60

[memory]
prune_threshold = 0.1
stability_hours = 168.0
storage_path = "~/.metascaffold/reflection_memory.json"
```

Also add entropy thresholds to `[classifier]`:

```toml
[classifier]
system2_threshold = 0.8
always_system2_tools = ["Write"]
entropy_threshold = 0.5
medium_entropy_threshold = 0.3
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: ALL PASS

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/metascaffold/config.py config/default_config.toml tests/test_config.py
git commit -m "feat(v0.3): add entropy, verifier, and memory configuration"
```

---

### Task 10: Wire v0.3 Components in Server

**Files:**
- Modify: `src/metascaffold/server.py`
- Modify: `src/metascaffold/__init__.py`

**Context:** Update server.py to pass new config values to components. Wire entropy thresholds to Classifier, verifier config to Evaluator, memory path to Reflector. Update hot-reload order to include new modules. Bump version to 0.3.0.

**Step 1: Update `server.py` component initialization**

Pass entropy thresholds to Classifier:

```python
classifier = Classifier(
    system2_threshold=config.classifier.system2_threshold,
    always_system2_tools=config.classifier.always_system2_tools,
    llm_client=_llm,
    entropy_threshold=config.classifier.entropy_threshold,
    medium_entropy_threshold=config.classifier.medium_entropy_threshold,
)
```

Pass memory path to Reflector:

```python
from pathlib import Path

reflector = Reflector(
    llm_client=_llm,
    memory_path=Path(config.memory.storage_path) if config.memory.storage_path else None,
)
```

Update `_RELOAD_ORDER` to include new modules:

```python
_RELOAD_ORDER = [
    "metascaffold.config",
    "metascaffold.telemetry",
    "metascaffold.notebooklm_bridge",
    "metascaffold.llm_client",
    "metascaffold.entropy",
    "metascaffold.classifier",
    "metascaffold.planner",
    "metascaffold.sandbox",
    "metascaffold.verifiers",
    "metascaffold.evaluator",
    "metascaffold.distiller",
    "metascaffold.reflection_memory",
    "metascaffold.reflector",
    "metascaffold.pipeline",
]
```

Update `_reload_components()` similarly — pass entropy config to Classifier, memory path to Reflector.

**Step 2: Bump version**

In `src/metascaffold/__init__.py`:

```python
__version__ = "0.3.0"
```

**Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/metascaffold/server.py src/metascaffold/__init__.py
git commit -m "feat(v0.3): wire entropy routing, verifiers, memory into server"
```

---

### Task 11: Update Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `docs/architecture.md`

**Context:** Update all docs to reflect v0.3 changes: dual backend, entropy routing, verifiers, memory decay, 3-level compute.

**Step 1: Update CLAUDE.md**

Add sections for:
- v0.3 new components: entropy.py, verifiers.py, reflection_memory.py
- Dual backend architecture: codex exec (default) + OpenAI API (logprobs only)
- Entropy threshold configuration
- OPEN_API_KEY env var requirement for entropy routing
- 3-level compute pipeline

**Step 2: Update README.md**

- Update architecture diagram to show dual backend
- Add v0.3 features section
- Update tool descriptions

**Step 3: Update docs/architecture.md**

- Add entropy routing section
- Add verifier pipeline diagram
- Add memory decay section
- Update pipeline state machine for 3-level routing

**Step 4: Commit**

```bash
git add CLAUDE.md README.md docs/architecture.md
git commit -m "docs: update documentation for v0.3.0"
```

---

### Task 12: Final Integration Test

**Files:**
- Modify: `tests/test_integration.py`

**Context:** Add integration tests that exercise the full v0.3 pipeline with mocked LLM, verifying entropy routing, verifier gating, and memory integration.

**Step 1: Write integration tests**

```python
class TestV03Integration:
    """Integration tests for v0.3 entropy + verifiers + memory pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_entropy_routing(self):
        """Full pipeline: entropy classifies → distill → plan → evaluate → reflect."""
        # Test with mock LLM that returns entropy-bearing responses
        pass  # Implement with full mock chain

    @pytest.mark.asyncio
    async def test_verifier_short_circuits_broken_code(self):
        """Broken syntax should short-circuit to backtrack without LLM call."""
        pass  # Implement with broken code_output

    @pytest.mark.asyncio
    async def test_system15_skips_plan_but_runs_distill(self):
        """System 1.5 compute level should run distill but skip plan."""
        pass  # Implement with medium-entropy mock
```

**Step 2: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS (120+ existing + ~40 new tests)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test(v0.3): add integration tests for entropy, verifiers, memory"
```

---

### Task 13: Live Validation

**Files:** No new files — testing with real LLM.

**Context:** Test the complete v0.3 pipeline against real APIs (codex exec and OpenAI API).

**Step 1: Hot-reload the server**

Use `metascaffold_restart` MCP tool to reload all modules.

**Step 2: Test entropy classification**

Call `metascaffold_classify` with an ambiguous task. Verify:
- Response includes `entropy` in signals
- `compute_level` is set (1, 1.5, or 2)
- Entropy value is a float

**Step 3: Test evaluator with code_output**

Call `metascaffold_evaluate` with broken Python code. Verify:
- Returns "backtrack" without timeout (verifier catches it fast)
- Response includes verifier findings

**Step 4: Test reflection memory persistence**

Call `metascaffold_reflect`. Verify:
- Rules are stored in `~/.metascaffold/reflection_memory.json`
- Calling reflect again reinforces existing rules

**Step 5: Commit final state**

```bash
git add -A
git commit -m "feat: MetaScaffold v0.3.0 — entropy routing, deterministic verifiers, memory decay"
```
