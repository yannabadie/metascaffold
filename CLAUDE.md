# MetaScaffold — Claude Code Instructions

## Overview

MetaScaffold is a **metacognition MCP server plugin** for Claude Code.
It implements a dual-process (System 1 / System 1.5 / System 2) cognitive architecture where:

- **Claude** (Anthropic) is the **generator** — writes code, makes edits, calls tools
- **Codex gpt-5.3** (OpenAI) is the **metacognitive layer** — classifies, evaluates, plans, reflects
- **OpenAI API gpt-4.1-nano** provides **entropy probes** — token logprobs for routing uncertainty measurement

This cross-model architecture means generator and evaluator have uncorrelated failure modes,
providing robust verification without the "self-evaluation blind spot" of single-model systems.

**Version:** 0.3.0
**Status:** Dual backend (codex exec + OpenAI API), entropy-based routing, deterministic verifiers, memory decay.
**Tests:** 175 passing (unit + integration)

## Project Structure

```
MetaScaffold/
├── src/metascaffold/           # Main package
│   ├── __init__.py             # Package version
│   ├── server.py               # FastMCP server — 9 tools, stdio transport
│   ├── llm_client.py           # Dual LLM backend: codex exec + OpenAI API (logprobs)
│   ├── classifier.py           # 3-level routing: entropy probe → codex LLM → heuristic
│   ├── distiller.py            # Self-Thought Task Distillation
│   ├── planner.py              # Context-aware strategy generation
│   ├── evaluator.py            # Verifiers + LLM-as-Judge + adversarial checks + PAG gate
│   ├── reflector.py            # MARS reflection loop (stores rules in ReflectionMemory)
│   ├── pipeline.py             # 6-stage cognitive orchestrator with 3-level compute
│   ├── entropy.py              # Shannon entropy computation from token logprobs
│   ├── verifiers.py            # Deterministic verification suite (AST, Ruff, pytest, mypy)
│   ├── reflection_memory.py    # Ebbinghaus decay model for learned rules
│   ├── sandbox.py              # Sandboxed subprocess execution
│   ├── telemetry.py            # Event logging (JSON files + SQLite)
│   ├── notebooklm_bridge.py    # NotebookLM knowledge base integration
│   └── config.py               # TOML configuration management
├── hooks/                      # Claude Code hooks
│   ├── pre_tool_gate.py        # PreToolUse — advisory classification trigger
│   └── post_tool_evaluate.py   # PostToolUse — advisory evaluation trigger
├── tests/                      # pytest suite (175 tests)
│   ├── test_llm_client.py      # LLM client + structured output tests
│   ├── test_classifier.py      # Heuristic + LLM + entropy classification tests
│   ├── test_evaluator.py       # Heuristic + LLM-as-Judge + verifier tests
│   ├── test_planner.py         # Heuristic + LLM planning tests
│   ├── test_distiller.py       # Task distillation tests
│   ├── test_reflector.py       # MARS reflection tests
│   ├── test_pipeline.py        # Pipeline orchestration tests
│   ├── test_integration.py     # End-to-end integration tests
│   ├── test_server.py          # MCP tool registration tests
│   ├── test_sandbox.py         # Sandbox execution tests
│   ├── test_telemetry.py       # Telemetry logging tests
│   ├── test_config.py          # Configuration loading tests
│   ├── test_hooks.py           # Hook interception tests
│   ├── test_entropy.py         # Shannon entropy computation tests
│   ├── test_verifiers.py       # Verification suite tests
│   ├── test_reflection_memory.py # Ebbinghaus decay model tests
│   ├── test_notebooklm_bridge.py
│   └── test_restart.py         # Hot-reload tests
├── config/
│   └── default_config.toml     # Default configuration
├── docs/
│   ├── architecture.md         # Architecture documentation
│   └── plans/                  # Design & implementation documents
│       ├── 2026-03-03-metascaffold-design.md       # v0.1 original design
│       ├── 2026-03-03-metascaffold-implementation.md
│       └── 2026-03-03-metascaffold-v02-llm-pipeline.md  # v0.2 LLM plan
├── scripts/
│   └── query_nlm.py            # NotebookLM query utility
├── pyproject.toml              # Project metadata & dependencies
└── uv.lock                     # Dependency lockfile
```

## Running

```bash
# Install dependencies
uv sync

# Run MCP server (stdio transport)
uv run python src/metascaffold/server.py

# Register with Claude Code
claude mcp add metascaffold -- uv --directory /path/to/MetaScaffold run python src/metascaffold/server.py

# Run tests
uv run pytest -v

# Run a single test file
uv run pytest tests/test_classifier.py -v
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPEN_API_KEY` | For entropy routing | OpenAI API key used by `complete_with_logprobs()` for the classifier's entropy probe |

When `OPEN_API_KEY` is not set, the classifier skips the entropy probe and falls back to codex exec LLM classification, then heuristic.

## Dual Backend Architecture

MetaScaffold v0.3 uses two LLM backends:

```
┌──────────────────────────────────────────────────────────┐
│                   MetaScaffold LLM Client                │
│                                                          │
│  ┌─────────────────────┐    ┌─────────────────────────┐  │
│  │   codex exec (CLI)  │    │   OpenAI API (REST)     │  │
│  │                     │    │                         │  │
│  │  Model: gpt-5.3     │    │  Model: gpt-4.1-nano    │  │
│  │  Auth: ChatGPT Plus │    │  Auth: OPEN_API_KEY     │  │
│  │  Cost: $0 (included)│    │  Feature: logprobs=True │  │
│  │  SSL: system certs  │    │  SSL: truststore        │  │
│  │                     │    │                         │  │
│  │  Used by:           │    │  Used by:               │  │
│  │  - ALL LLM calls    │    │  - Classifier entropy   │  │
│  │  - Classify (codex) │    │    probe ONLY           │  │
│  │  - Distill          │    │                         │  │
│  │  - Plan             │    │  Returns: token_logprobs│  │
│  │  - Evaluate         │    │  in LLMResponse         │  │
│  │  - Reflect          │    │                         │  │
│  └─────────────────────┘    └─────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

**Key constraint:** `codex exec` (gpt-5.3) is the **default backend for ALL LLM calls**. The OpenAI API (`complete_with_logprobs()`) is **ONLY** used for the Classifier's entropy probe via gpt-4.1-nano.

## Cognitive Pipeline (3-Level Compute)

```
Classify → Distill → Plan → Execute → Evaluate → Reflect
   │          │                           │
   │ System 1 │───────────────────────────┘ (skip Distill + Plan)
   │          │
   │ System 1.5 ─────────────────────────── (skip Plan only)
   │
   └── retry loop (Evaluate → re-Execute) ──→ escalate after max_attempts
       backtrack loop (Evaluate → re-Plan → Execute)
```

| Compute Level | Entropy Range | Stages Executed | Description |
|---------------|---------------|-----------------|-------------|
| System 1      | H < 0.3       | Classify → Execute → Evaluate → Reflect | Fast path: skip Distill + Plan |
| System 1.5    | 0.3 <= H < 0.5 | Classify → Distill → Execute → Evaluate → Reflect | Medium path: skip Plan only |
| System 2      | H >= 0.5      | Classify → Distill → Plan → Execute → Evaluate → Reflect | Full pipeline |

### Classifier Degradation Chain

The classifier uses a 3-step fallback chain:

1. **Entropy probe** (OpenAI API, gpt-4.1-nano) — Shannon entropy from token logprobs: `H = -sum(p * log2(p))`
2. **Codex exec LLM** (gpt-5.3) — Semantic classification with structured output
3. **Heuristic** — Regex patterns, tool type, historical success rate

If step 1 fails (no API key, network error), falls back to step 2. If step 2 fails, falls back to step 3.

| Stage     | Component   | LLM Model        | Purpose                                      |
|-----------|-------------|------------------|----------------------------------------------|
| Classify  | Classifier  | gpt-4.1-nano (API) + gpt-5.3 (codex) | Entropy probe → LLM routing → heuristic |
| Distill   | Distiller   | gpt-4.1-nano (codex) | Structure raw task into template          |
| Plan      | Planner     | gpt-4.1-mini (codex) | Generate execution strategies with rollbacks |
| Execute   | Sandbox     | —                | Sandboxed subprocess (deterministic)         |
| Verify    | Verifiers   | —                | AST parse, Ruff lint, pytest, mypy           |
| Evaluate  | Evaluator   | o3-mini (codex)  | LLM-as-Judge with adversarial checks         |
| Reflect   | Reflector   | o3-mini (codex)  | Extract rules/procedures into ReflectionMemory |

## MCP Tools (9)

| Tool                       | Async | Description                                              |
|----------------------------|-------|----------------------------------------------------------|
| `metascaffold_classify`    | yes   | 3-level routing with entropy probe + LLM + heuristic     |
| `metascaffold_distill`     | yes   | Structure raw task into template (objective, constraints) |
| `metascaffold_plan`        | yes   | Generate strategies with risks and rollback plans         |
| `metascaffold_sandbox_exec`| no    | Execute command in sandboxed subprocess                   |
| `metascaffold_evaluate`    | yes   | Verifiers + LLM-as-Judge verdict: pass/retry/backtrack/escalate |
| `metascaffold_reflect`     | yes   | Extract learning patterns into ReflectionMemory           |
| `metascaffold_nlm_query`   | yes   | Query NotebookLM knowledge base                           |
| `metascaffold_telemetry_query` | no | Query historical success rates from SQLite              |
| `metascaffold_restart`     | no    | Hot-reload all modules (no server restart needed)         |

## LLM Backend

### Primary: codex exec (all LLM calls)

**Engine:** `codex exec` subprocess (Codex CLI)
**Auth:** ChatGPT Plus subscription (no API key needed)
**Cost:** Zero additional cost (included in subscription)
**Structured output:** `--output-schema <schema.json> -o <output.json>` for guaranteed valid JSON

Each component defines a `_*_RESPONSE_SCHEMA` (JSON Schema with `additionalProperties: false`)
that enforces the output structure at the model level. The LLM client auto-adds
`additionalProperties: false` to schemas that don't include it.

### Secondary: OpenAI API (entropy probe only)

**Engine:** `complete_with_logprobs()` via `openai` Python SDK
**Auth:** `OPEN_API_KEY` environment variable
**Model:** gpt-4.1-nano
**Feature:** `logprobs=True` returns `token_logprobs` in `LLMResponse`
**SSL:** Uses `truststore` for corporate SSL certificate compatibility
**Purpose:** Classifier entropy probe ONLY — no other component uses this backend

**Fallback:** When codex is unavailable or the LLM call fails, every component falls back
silently to heuristic behavior. When the OpenAI API is unavailable, the classifier skips
the entropy probe and falls through to codex exec classification.

## v0.3 Components

### Entropy (`entropy.py`)

Computes Shannon entropy from token logprobs to measure routing uncertainty:

```
H = -sum(p * log2(p))
```

- Low entropy (H < 0.3) → high confidence → System 1 (fast path)
- Medium entropy (0.3 <= H < 0.5) → moderate confidence → System 1.5 (skip Plan)
- High entropy (H >= 0.5) → low confidence → System 2 (full pipeline)

### Verifiers (`verifiers.py`)

`VerificationSuite` runs deterministic checks BEFORE LLM-as-Judge evaluation:

| Verifier | Check | Short-circuit |
|----------|-------|---------------|
| AST parse | `ast.parse()` on Python files | Yes — syntax error → backtrack |
| Ruff lint | `ruff check` subprocess | Yes — critical lint → backtrack |
| pytest   | `pytest` subprocess | Yes — test failures → backtrack |
| mypy     | `mypy` type check (optional) | Yes — type errors → backtrack |

Critical failures short-circuit to "backtrack" verdict **without** an LLM call, saving tokens and latency.

### Reflection Memory (`reflection_memory.py`)

Ebbinghaus forgetting curve for learned rules:

```
retention = e^(-t / (stability * reinforcement_factor))
```

- **stability**: Base half-life (default: 168 hours = 1 week)
- **reinforcement_factor**: Increases when a rule is reinforced (repeated learning)
- **Pruning**: Rules with retention < 0.1 are pruned on load
- **Storage**: JSON file at `~/.metascaffold/reflection_memory.json`

The Reflector stores and reinforces rules in ReflectionMemory instead of accumulating them unboundedly.

## Configuration

Default config: `config/default_config.toml`
User override: `~/.metascaffold/config.toml` (optional, merged over defaults)

```toml
[classifier]
system2_threshold = 0.8           # Below this confidence → System 2
always_system2_tools = ["Write"]  # Always force System 2
entropy_threshold = 0.5           # Above this entropy → System 2
medium_entropy_threshold = 0.3    # Above this entropy → System 1.5

[sandbox]
default_timeout_seconds = 30
max_retry_attempts = 3

[telemetry]
json_dir = "~/.metascaffold/telemetry/"
sqlite_path = "~/.metascaffold/cognitive.db"

[notebooklm]
enabled = true
default_notebook = "MetaScaffold_Core"
fallback_on_error = true

[llm]
enabled = true
fallback_to_heuristics = true
classifier_model = "gpt-4.1-nano"
evaluator_model = "o3-mini"
planner_model = "gpt-4.1-mini"
distiller_model = "gpt-4.1-nano"
reflector_model = "o3-mini"

[verifier]
run_ast = true                    # AST parse check on Python files
run_ruff = true                   # Ruff lint check
run_mypy = false                  # mypy type check (disabled by default)
run_pytest = true                 # pytest execution

[memory]
prune_threshold = 0.1             # Prune rules below this retention
stability_hours = 168             # Base half-life in hours (1 week)
storage_path = "~/.metascaffold/reflection_memory.json"
```

## Key Architecture Decisions

1. **Cross-model verification**: Claude generates → Codex evaluates. Different models = uncorrelated biases.
2. **Dual backend**: `codex exec` for all LLM calls (free via ChatGPT Plus) + OpenAI API only for entropy logprobs.
3. **Deterministic intermediate layer**: Sandbox executes real code; verifiers check before LLM judges.
4. **Structured output via `--output-schema`**: JSON Schemas enforced at model level (not post-hoc parsing).
5. **Graceful degradation**: Entropy probe → codex LLM → heuristic fallback chain. Never hard-fails.
6. **Hot-reload**: `metascaffold_restart` reloads all modules (including entropy, verifiers, memory) without killing the MCP server process.
7. **Entropy-based routing**: Shannon entropy from token logprobs provides objective uncertainty measurement, replacing prompt-based self-reported confidence.
8. **Ebbinghaus memory decay**: Learned rules fade naturally over time unless reinforced, preventing unbounded accumulation.

## Development

- **Python 3.11+** required
- **uv** for dependency management
- **pytest** + **pytest-asyncio** for testing
- All async tests use `asyncio_mode = "auto"` (see `pyproject.toml`)
- Mock `asyncio.create_subprocess_exec` for LLM client tests
- Mock `client.complete` return value for component tests
- Set `OPEN_API_KEY` in environment for entropy probe integration tests

## Known Limitations

- **Hot-reload** only reloads modules already in `sys.modules`; new tools require full server restart
- **Model parameter is ignored**: Codex exec uses whatever model is configured in Codex CLI
- **Hooks are advisory**: PreToolUse/PostToolUse hooks suggest but don't enforce classification/evaluation
- **Entropy probe requires API key**: `OPEN_API_KEY` must be set for entropy-based routing; without it, falls back to codex LLM classification
- **mypy verifier disabled by default**: Can be slow on large codebases; enable via `[verifier] run_mypy = true`

## Research Foundations

Built on 2025-2026 dual-process AI research:
- **SOFAI-LM**: Slow/Fast AI metacognitive governance
- **SELF-THOUGHT**: Task distillation before self-correction
- **MARS**: Metacognitive Agent Reflective Self-improvement
- **PAG**: Plan-and-Gate selective revision (prevents model collapse)
- **Adversarial post-exec**: Counter confirmation bias in LLM evaluation
- **Ebbinghaus forgetting curve**: Spaced repetition and natural memory decay for learned rules

See `docs/research_sources.json` for the full bibliography.
