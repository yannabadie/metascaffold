# MetaScaffold Architecture — v0.3

## System Overview

MetaScaffold is an MCP server that acts as a metacognitive layer for Claude Code.
It intercepts Claude's tool calls and provides cognitive services via 9 MCP tools.

v0.3 introduces a dual LLM backend, entropy-based 3-level routing, deterministic verifiers,
and Ebbinghaus memory decay for learned rules.

### Dual-Model Verification with Dual Backend

```
                    ┌─────────────────────────────────┐
                    │           Claude Code            │
                    │         (Anthropic model)        │
                    │                                  │
                    │   Generates code, edits files,   │
                    │   runs commands, calls tools     │
                    └───────────────┬──────────────────┘
                                    │ MCP (stdio)
                    ┌───────────────▼──────────────────────────────┐
                    │          MetaScaffold MCP Server              │
                    │                                              │
                    │  ┌──────────────────┐ ┌───────────────────┐  │
                    │  │  codex exec      │ │  OpenAI API       │  │
                    │  │  (gpt-5.3)       │ │  (gpt-4.1-nano)   │  │
                    │  │  ALL LLM calls   │ │  Entropy probe    │  │
                    │  │  (default)       │ │  (logprobs only)  │  │
                    │  └────────┬─────────┘ └────────┬──────────┘  │
                    │           │                    │             │
                    │  ┌────────▼────────────────────▼──────────┐  │
                    │  │  Classifier (3-level: S1 / S1.5 / S2) │  │
                    │  └────────┬───────────────────────────────┘  │
                    │           │                                  │
                    │  ┌────────▼─────┐  ┌─────────────┐          │
                    │  │  Distiller   │  │  Verifiers   │          │
                    │  └────────┬─────┘  │ (AST, Ruff,  │          │
                    │           │        │  pytest,mypy) │          │
                    │  ┌────────▼─────┐  └──────┬──────┘          │
                    │  │   Planner    │         │                  │
                    │  └────────┬─────┘  ┌──────▼──────┐          │
                    │           │        │  Evaluator   │          │
                    │  ┌────────▼────────┴──────────────┘          │
                    │  │    Pipeline Orchestrator                  │
                    │  │    (3-level adaptive compute)             │
                    │  └────────────┬──────────────────┘           │
                    │               │                              │
                    │  ┌────────────▼───────────────┐              │
                    │  │   Reflector (MARS loop)    │              │
                    │  │   + ReflectionMemory       │              │
                    │  │   (Ebbinghaus decay)       │              │
                    │  └────────────────────────────┘              │
                    │                                              │
                    │  ┌────────────────────────────┐              │
                    │  │  Telemetry  │  NotebookLM  │              │
                    │  │ (SQLite+JSON) │  (Bridge)  │              │
                    │  └────────────────────────────┘              │
                    └──────────────────────────────────────────────┘
```

The key architectural insight: **Claude (generator) and Codex (evaluator) are different models
from different vendors.** This means their failure modes are uncorrelated, avoiding the
"self-evaluation blind spot" documented in the 2025 AI verification literature.

Between generator and evaluator sits a **deterministic sandbox** that executes real code,
providing ground truth (exit codes, test results, stderr) that anchors the LLM evaluation.
In v0.3, a **deterministic verifier suite** further anchors evaluation with AST, lint, and
test checks before the LLM-as-Judge is invoked.

## Component Details

### LLM Client (`llm_client.py`)

Async abstraction over two LLM backends.

**Backend 1: codex exec (default for ALL LLM calls)**

Used by all components (Classifier, Distiller, Planner, Evaluator, Reflector).

Two execution paths:

1. **Structured output** (`_complete_with_schema`): When `response_format` is provided:
   - Writes JSON Schema to temp file
   - Passes `--output-schema <schema.json> -o <result.json>` to codex exec
   - Reads clean JSON from output file
   - Schema MUST have `additionalProperties: false` at top level (auto-added if missing)

2. **Raw output** (`_complete_raw`): When no schema is provided:
   - Captures stdout from codex exec
   - Parses output by finding content between "codex" marker and "tokens used"

**Backend 2: OpenAI API (entropy probe only)**

Used exclusively by the Classifier for entropy measurement.

- **Method:** `complete_with_logprobs(prompt, model="gpt-4.1-nano")`
- **Auth:** `OPEN_API_KEY` environment variable
- **SSL:** Uses `truststore` for corporate SSL certificate compatibility
- **Returns:** `LLMResponse` with `token_logprobs: list[dict]` containing per-token log probabilities
- **Purpose:** Provides raw logprobs that `entropy.py` converts to Shannon entropy for routing

**`LLMResponse` dataclass (v0.3):**

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | Raw text response |
| `parsed` | `dict \| None` | Parsed JSON (structured output) |
| `error` | `str \| None` | Error message (graceful degradation) |
| `token_logprobs` | `list[dict] \| None` | Per-token logprobs (OpenAI API only) |

**Error handling:** Timeout (120s), subprocess errors, missing output, API errors — all return
`LLMResponse(error=...)` for graceful degradation.

### Entropy (`entropy.py`)

Computes Shannon entropy from token logprobs to measure classifier routing uncertainty.

**Formula:** `H = -sum(p * log2(p))` where `p` is the probability of each token.

**Thresholds (configurable):**

| Entropy Range | Compute Level | Behavior |
|---------------|---------------|----------|
| H < 0.3       | System 1      | High confidence — skip Distill + Plan |
| 0.3 <= H < 0.5 | System 1.5   | Moderate confidence — skip Plan only |
| H >= 0.5      | System 2      | Low confidence — full pipeline |

The entropy module is stateless and purely functional: it takes logprobs in, returns a float out.

### Classifier (`classifier.py`)

Routes tasks to System 1 (fast), System 1.5 (medium), or System 2 (deliberate).

**Degradation chain (3-step fallback):**

```
┌──────────────────┐     fail     ┌──────────────────┐     fail     ┌──────────────────┐
│  1. Entropy probe│ ───────────► │  2. Codex exec   │ ───────────► │  3. Heuristic    │
│  (OpenAI API)    │              │  LLM (gpt-5.3)   │              │  (regex + stats) │
│                  │              │                   │              │                  │
│  gpt-4.1-nano    │              │  Structured JSON  │              │  No LLM needed   │
│  logprobs=True   │              │  classification   │              │                  │
│  → Shannon H     │              │                   │              │                  │
└──────────────────┘              └───────────────────┘              └──────────────────┘
```

**Step 1 — Entropy probe:** Calls `complete_with_logprobs()` via OpenAI API, computes Shannon entropy,
maps to compute level (System 1 / 1.5 / 2). Requires `OPEN_API_KEY`.

**Step 2 — Codex exec LLM:** Falls back to semantic classification via `codex exec` with structured output.
Returns `routing` (system1/system2), `confidence`, `reasoning`.

**Step 3 — Heuristic:** Regex patterns for complexity, destructive commands, simple commands,
historical success rate from telemetry.

**Fast paths (no LLM needed):**
- **Fast-path System 1**: Read-only tools (`Read`, `Grep`, `Glob`, etc.) → skip LLM
- **Fast-path System 2**: Tools in `always_system2_tools` config → skip LLM

**JSON Schema** enforces: `routing` (enum: system1/system2), `confidence` (number), `reasoning` (string), `compute_level` (enum: system1/system1_5/system2).

### Distiller (`distiller.py`)

Implements **Self-Thought Task Distillation** — transforms raw task text into structured
`TaskTemplate` before the planner sees it.

**Output:** `objective`, `constraints[]`, `target_files[]`, `variables{}`

Variables use array-of-objects format (`[{key, value}]`) in the JSON Schema for
`additionalProperties: false` compliance, converted to dict in Python.

### Planner (`planner.py`)

Generates 1-3 execution strategies with steps, confidence, risks, and rollback plans.

**Heuristic templates** (fallback): Pattern-matched to refactor/bugfix/feature/generic.

**LLM planning**: Context-aware strategies. Optionally enriched with NotebookLM domain insights.

### Verifiers (`verifiers.py`)

`VerificationSuite` — deterministic checks that run BEFORE LLM-as-Judge evaluation.

**Architecture:**

```
┌──────────────┐
│  code_output │ (stdout/stderr/exit_code from sandbox)
└──────┬───────┘
       │
       ▼
┌──────────────┐   pass    ┌──────────────┐   pass    ┌──────────────┐   pass    ┌──────────────┐
│  AST parse   │ ────────► │  Ruff lint   │ ────────► │   pytest     │ ────────► │    mypy      │
│  (if Python) │           │  (if enabled)│           │  (if enabled)│           │  (if enabled)│
└──────┬───────┘           └──────┬───────┘           └──────┬───────┘           └──────┬───────┘
       │ fail                     │ fail                     │ fail                     │ fail
       ▼                          ▼                          ▼                          ▼
   BACKTRACK               BACKTRACK                  BACKTRACK                  BACKTRACK
   (no LLM call)           (no LLM call)              (no LLM call)              (no LLM call)
```

**Key design principle:** Critical failures short-circuit to "backtrack" verdict WITHOUT invoking
the LLM-as-Judge, saving tokens and latency. Only code that passes all enabled verifiers
proceeds to LLM evaluation.

**Configuration** (`[verifier]` section):

| Setting | Default | Description |
|---------|---------|-------------|
| `run_ast` | `true` | AST parse check on Python files |
| `run_ruff` | `true` | Ruff lint check |
| `run_mypy` | `false` | mypy type check (disabled — can be slow) |
| `run_pytest` | `true` | pytest execution |

### Evaluator (`evaluator.py`)

**Verifiers + LLM-as-Judge** with four verification layers:

1. **Deterministic verifiers** (v0.3): AST, Ruff, pytest, mypy. Critical failures → backtrack without LLM call.
2. **SOFAI feedback**: `failing_tests`, `error_lines`, `root_cause`, `suggested_fix`
3. **Adversarial check**: Scans for security/logic issues. If LLM says "pass" but finds
   adversarial issues → automatically downgrades to "retry"
4. **PAG gate**: `revision_allowed` boolean. When false, the pipeline must not auto-fix
   (prevents model collapse from indiscriminate revision)

The evaluator accepts an optional `code_output` parameter containing the sandbox execution
results, which is passed to the verifier suite before LLM evaluation.

**Verdicts:** `pass` | `retry` | `backtrack` | `escalate`

**Max attempts**: After `max_retry_attempts`, "retry" is automatically promoted to "escalate".

### Reflection Memory (`reflection_memory.py`)

Implements Ebbinghaus forgetting curve for learned rules:

```
retention = e^(-t / (stability * reinforcement_factor))
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stability` | 168 hours (1 week) | Base half-life for rule retention |
| `reinforcement_factor` | Starts at 1.0, grows | Multiplier increased when a rule is reinforced |
| `prune_threshold` | 0.1 | Rules below this retention are pruned |

**Storage:** JSON file at `~/.metascaffold/reflection_memory.json` containing hot rules
with timestamps, reinforcement counts, and content.

**Operations:**
- `store(rule)` — Add a new rule with current timestamp
- `reinforce(rule_id)` — Increase reinforcement factor (strengthens retention)
- `get_active_rules()` — Return rules with retention > prune_threshold
- `prune()` — Remove expired rules (retention < prune_threshold)

### Reflector (`reflector.py`)

Implements **MARS (Metacognitive Agent Reflective Self-improvement)** single-cycle reflection.

Analyzes batches of telemetry events (evaluations, backtracks, escalations) and extracts:
- **Rules**: Normative constraints ("Always run tests after modifying shared code")
- **Procedures**: Reusable strategies that worked or should replace failed approaches

**v0.3 change:** The Reflector stores extracted rules in `ReflectionMemory` and reinforces
existing rules when they are re-discovered. This replaces the v0.2 behavior of unbounded
rule accumulation.

### Pipeline (`pipeline.py`)

6-stage orchestrator managing the cognitive flow with 3-level adaptive compute.

**`PipelineState` (v0.3):**

New field: `compute_level` (enum: `system1` | `system1_5` | `system2`)

New properties:
- `should_bypass_distill` — True when `compute_level == system1`
- `should_bypass_plan` — True when `compute_level in (system1, system1_5)`

**State machine (3-level):**

```
                    ┌──────────────┐
                    │   Classify   │
                    │  (entropy +  │
                    │   fallback)  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         System 1    System 1.5    System 2
         (H < 0.3)  (0.3≤H<0.5)  (H ≥ 0.5)
              │            │            │
              │     ┌──────▼───────┐    │
              │     │   Distill    │    │
              │     └──────┬───────┘    │
              │            │     ┌──────▼───────┐
              │            │     │   Distill    │
              │            │     └──────┬───────┘
              │            │     ┌──────▼───────┐
              │            │     │     Plan     │◄── backtrack ──┐
              │            │     └──────┬───────┘               │
              │            │            │                       │
       ┌──────▼────────────▼────────────▼───────┐               │
       │              Execute                    │◄── retry ──┐ │
       └──────────────────┬─────────────────────┘             │ │
       ┌──────────────────▼─────────────────────┐             │ │
       │              Verify                     │             │ │
       │     (AST, Ruff, pytest, mypy)           │             │ │
       │     critical fail → backtrack           │─────────────┘ │
       └──────────────────┬─────────────────────┘    (no LLM)  │
       ┌──────────────────▼─────────────────────┐               │
       │              Evaluate                   ├──────────────┘
       │         (LLM-as-Judge)                  │
       │              escalate → stop            │
       └──────────────────┬─────────────────────┘
                    pass  │
       ┌──────────────────▼─────────────────────┐
       │              Reflect                    │
       │     (store/reinforce rules in           │
       │      ReflectionMemory)                  │
       └─────────────────────────────────────────┘
```

### Sandbox (`sandbox.py`)

Executes commands in subprocess with:
- Configurable timeout (default 30s)
- stdout/stderr capture
- Duration measurement
- Exit code and timeout detection

### Telemetry (`telemetry.py`)

Dual-sink event logging:
- **JSON files**: One file per event in `~/.metascaffold/telemetry/`
- **SQLite**: Structured storage in `~/.metascaffold/cognitive.db`
- **Queries**: `get_success_rate(task_type)` for classifier feedback, `get_recent_events(n)` for reflector

### NotebookLM Bridge (`notebooklm_bridge.py`)

Queries Google NotebookLM for domain-specific knowledge.
Used by the Planner to enrich strategies with research insights.
Async client with graceful degradation.

## Structured Output

All LLM-calling components use JSON Schema enforcement via `--output-schema`:

| Component  | Schema constant               | Key fields                                              |
|------------|-------------------------------|---------------------------------------------------------|
| Classifier | `_CLASSIFIER_RESPONSE_SCHEMA` | routing (enum), confidence, reasoning, compute_level    |
| Evaluator  | `_EVALUATOR_RESPONSE_SCHEMA`  | verdict (enum), feedback{}, adversarial_findings[], revision_allowed |
| Planner    | `_PLANNER_RESPONSE_SCHEMA`    | strategies[{id, steps[], risks[], rollback_plan}], recommended |
| Distiller  | `_DISTILLER_RESPONSE_SCHEMA`  | objective, constraints[], target_files[], variables[]    |
| Reflector  | `_REFLECTOR_RESPONSE_SCHEMA`  | rules[], procedures[]                                   |

All schemas include `"additionalProperties": false` at the top level and on all nested objects
(required by the OpenAI structured output API).

## Graceful Degradation

Every component follows the same pattern:

```python
async def method_async(self, ...):
    """Try LLM, fall back to heuristic."""
    if self._llm and self._llm.enabled:
        result = await self._llm_method(...)
        if result is not None:
            return result
    return self._heuristic_method(...)
```

### Classifier Degradation Chain (v0.3)

The classifier has a 3-step fallback that is more granular than other components:

```python
async def classify_async(self, ...):
    # Step 1: Entropy probe (OpenAI API)
    if self._llm and self._llm.api_available:
        entropy = await self._entropy_probe(...)
        if entropy is not None:
            return self._route_by_entropy(entropy)

    # Step 2: Codex exec LLM classification
    if self._llm and self._llm.enabled:
        result = await self._llm_classify(...)
        if result is not None:
            return result

    # Step 3: Heuristic fallback
    return self._heuristic_classify(...)
```

### General Failure Modes

- Codex CLI not installed → `LLMClient.enabled = False`
- Subprocess error → `LLMResponse(error=...)`
- Timeout (120s) → `LLMResponse(error="timed out")`
- JSON parse failure → return `None` → trigger fallback
- NotebookLM unavailable → empty insights
- `OPEN_API_KEY` not set → skip entropy probe → codex LLM → heuristic
- OpenAI API error → skip entropy probe → codex LLM → heuristic
- Verifier tool not installed (e.g., ruff, mypy) → skip that verifier, continue

## Configuration

Priority: `config_path` argument > `~/.metascaffold/config.toml` > `config/default_config.toml`

Deep-merged via `_merge_dicts()`. Paths expanded via `~` resolution.

### v0.3 Config Sections

```toml
[classifier]
entropy_threshold = 0.5           # H >= this → System 2
medium_entropy_threshold = 0.3    # H >= this → System 1.5

[verifier]
run_ast = true                    # AST parse check
run_ruff = true                   # Ruff lint check
run_mypy = false                  # mypy type check (disabled by default)
run_pytest = true                 # pytest execution

[memory]
prune_threshold = 0.1             # Prune rules below this retention
stability_hours = 168             # Base half-life (1 week)
storage_path = "~/.metascaffold/reflection_memory.json"
```

## Hot Reload

`metascaffold_restart` reloads all modules in dependency order:

```
config → telemetry → notebooklm_bridge → llm_client → entropy →
verifiers → reflection_memory → classifier → planner → sandbox →
evaluator → distiller → reflector → pipeline
```

Then re-instantiates all component instances with fresh config.
Only reloads modules already in `sys.modules` (new tools require full server restart).

v0.3 additions to the reload chain: `entropy`, `verifiers`, `reflection_memory` modules
are inserted after `llm_client` and before `classifier` in the dependency order.
