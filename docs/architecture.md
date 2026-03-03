# MetaScaffold Architecture — v0.2

## System Overview

MetaScaffold is an MCP server that acts as a metacognitive layer for Claude Code.
It intercepts Claude's tool calls and provides cognitive services via 9 MCP tools.

### Dual-Model Verification

```
                    ┌─────────────────────────────────┐
                    │           Claude Code            │
                    │         (Anthropic model)        │
                    │                                  │
                    │   Generates code, edits files,   │
                    │   runs commands, calls tools     │
                    └───────────────┬──────────────────┘
                                    │ MCP (stdio)
                    ┌───────────────▼──────────────────┐
                    │       MetaScaffold MCP Server     │
                    │        (Codex gpt-5.3 model)     │
                    │                                  │
                    │  ┌──────────┐  ┌─────────────┐   │
                    │  │Classifier│  │  Distiller   │   │
                    │  └────┬─────┘  └──────┬──────┘   │
                    │       │               │          │
                    │  ┌────▼─────┐  ┌──────▼──────┐   │
                    │  │ Planner  │  │  Evaluator   │   │
                    │  └────┬─────┘  └──────┬──────┘   │
                    │       │               │          │
                    │  ┌────▼───────────────▼──────┐   │
                    │  │    Pipeline Orchestrator   │   │
                    │  └────────────┬───────────────┘   │
                    │               │                  │
                    │  ┌────────────▼───────────────┐   │
                    │  │   Reflector (MARS loop)    │   │
                    │  └────────────────────────────┘   │
                    │                                  │
                    │  ┌────────────────────────────┐   │
                    │  │  Telemetry  │  NotebookLM  │   │
                    │  │ (SQLite+JSON) │  (Bridge)  │   │
                    │  └────────────────────────────┘   │
                    └──────────────────────────────────┘
```

The key architectural insight: **Claude (generator) and Codex (evaluator) are different models
from different vendors.** This means their failure modes are uncorrelated, avoiding the
"self-evaluation blind spot" documented in the 2025 AI verification literature.

Between generator and evaluator sits a **deterministic sandbox** that executes real code,
providing ground truth (exit codes, test results, stderr) that anchors the LLM evaluation.

## Component Details

### LLM Client (`llm_client.py`)

Async abstraction over `codex exec` subprocess.

**Two execution paths:**

1. **Structured output** (`_complete_with_schema`): When `response_format` is provided:
   - Writes JSON Schema to temp file
   - Passes `--output-schema <schema.json> -o <result.json>` to codex exec
   - Reads clean JSON from output file
   - Schema MUST have `additionalProperties: false` at top level (auto-added if missing)

2. **Raw output** (`_complete_raw`): When no schema is provided:
   - Captures stdout from codex exec
   - Parses output by finding content between "codex" marker and "tokens used"

**Error handling:** Timeout (120s), subprocess errors, missing output — all return
`LLMResponse(error=...)` for graceful degradation.

### Classifier (`classifier.py`)

Routes tasks to System 1 (fast, automatic) or System 2 (slow, deliberate).

**Three routing paths:**
- **Fast-path System 1**: Read-only tools (`Read`, `Grep`, `Glob`, etc.) → skip LLM
- **Fast-path System 2**: Tools in `always_system2_tools` config → skip LLM
- **LLM classification**: Everything else → semantic analysis via Codex

**Heuristic signals** (fallback): regex patterns for complexity, destructive commands,
simple commands, historical success rate from telemetry.

**JSON Schema** enforces: `routing` (enum: system1/system2), `confidence` (number), `reasoning` (string).

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

### Evaluator (`evaluator.py`)

**LLM-as-Judge** with three verification layers:

1. **SOFAI feedback**: `failing_tests`, `error_lines`, `root_cause`, `suggested_fix`
2. **Adversarial check**: Scans for security/logic issues. If LLM says "pass" but finds
   adversarial issues → automatically downgrades to "retry"
3. **PAG gate**: `revision_allowed` boolean. When false, the pipeline must not auto-fix
   (prevents model collapse from indiscriminate revision)

**Verdicts:** `pass` | `retry` | `backtrack` | `escalate`

**Max attempts**: After `max_retry_attempts`, "retry" is automatically promoted to "escalate".

### Reflector (`reflector.py`)

Implements **MARS (Metacognitive Agent Reflective Self-improvement)** single-cycle reflection.

Analyzes batches of telemetry events (evaluations, backtracks, escalations) and extracts:
- **Rules**: Normative constraints ("Always run tests after modifying shared code")
- **Procedures**: Reusable strategies that worked or should replace failed approaches

### Pipeline (`pipeline.py`)

6-stage orchestrator managing the cognitive flow.

**State machine:**

```
                    ┌──────────────┐
                    │   Classify   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐    System 1
                    │   Bypass?    ├────────────────────────┐
                    └──────┬───────┘                        │
                     No    │                                │
                    ┌──────▼───────┐                        │
                    │   Distill    │                        │
                    └──────┬───────┘                        │
                    ┌──────▼───────┐                        │
                    │     Plan     │◄──── backtrack ────┐   │
                    └──────┬───────┘                    │   │
                    ┌──────▼───────┐                    │   │
                    │   Execute    │◄──── retry ────┐   │   │
                    └──────┬───────┘                │   │   │
                    ┌──────▼───────┐                │   │   │
                    │   Evaluate   ├────────────────┘   │   │
                    │              ├────────────────────┘   │
                    │              ├── escalate → stop      │
                    └──────┬───────┘                        │
                     pass  │                                │
                    ┌──────▼───────┐◄───────────────────────┘
                    │   Reflect    │
                    └──────────────┘
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
| Classifier | `_CLASSIFIER_RESPONSE_SCHEMA` | routing (enum), confidence, reasoning                   |
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

Failure modes handled:
- Codex CLI not installed → `LLMClient.enabled = False`
- Subprocess error → `LLMResponse(error=...)`
- Timeout (120s) → `LLMResponse(error="timed out")`
- JSON parse failure → return `None` → trigger fallback
- NotebookLM unavailable → empty insights

## Configuration

Priority: `config_path` argument > `~/.metascaffold/config.toml` > `config/default_config.toml`

Deep-merged via `_merge_dicts()`. Paths expanded via `~` resolution.

## Hot Reload

`metascaffold_restart` reloads all modules in dependency order:

```
config → telemetry → notebooklm_bridge → llm_client →
classifier → planner → sandbox → evaluator → distiller → reflector → pipeline
```

Then re-instantiates all component instances with fresh config.
Only reloads modules already in `sys.modules` (new tools require full server restart).
