# MetaScaffold — Claude Code Instructions

## Overview

MetaScaffold is a **metacognition MCP server plugin** for Claude Code.
It implements a dual-process (System 1 / System 2) cognitive architecture where:

- **Claude** (Anthropic) is the **generator** — writes code, makes edits, calls tools
- **Codex gpt-5.3** (OpenAI) is the **metacognitive layer** — classifies, evaluates, plans, reflects

This cross-model architecture means generator and evaluator have uncorrelated failure modes,
providing robust verification without the "self-evaluation blind spot" of single-model systems.

**Version:** 0.2.0
**Status:** All 7 cognitive components use real LLM calls via `codex exec` with heuristic fallback.
**Tests:** 120 passing (unit + integration)

## Project Structure

```
MetaScaffold/
├── src/metascaffold/           # Main package
│   ├── __init__.py             # Package version
│   ├── server.py               # FastMCP server — 9 tools, stdio transport
│   ├── llm_client.py           # LLM abstraction over codex exec subprocess
│   ├── classifier.py           # System 1/2 routing (LLM + heuristic)
│   ├── distiller.py            # Self-Thought Task Distillation
│   ├── planner.py              # Context-aware strategy generation
│   ├── evaluator.py            # LLM-as-Judge + adversarial checks + PAG gate
│   ├── reflector.py            # MARS reflection loop (telemetry learning)
│   ├── pipeline.py             # 6-stage cognitive orchestrator
│   ├── sandbox.py              # Sandboxed subprocess execution
│   ├── telemetry.py            # Event logging (JSON files + SQLite)
│   ├── notebooklm_bridge.py    # NotebookLM knowledge base integration
│   └── config.py               # TOML configuration management
├── hooks/                      # Claude Code hooks
│   ├── pre_tool_gate.py        # PreToolUse — advisory classification trigger
│   └── post_tool_evaluate.py   # PostToolUse — advisory evaluation trigger
├── tests/                      # pytest suite (120 tests)
│   ├── test_llm_client.py      # LLM client + structured output tests
│   ├── test_classifier.py      # Heuristic + LLM classification tests
│   ├── test_evaluator.py       # Heuristic + LLM-as-Judge tests
│   ├── test_planner.py         # Heuristic + LLM planning tests
│   ├── test_distiller.py       # Task distillation tests
│   ├── test_reflector.py       # MARS reflection tests
│   ├── test_pipeline.py        # Pipeline orchestration tests (16 tests)
│   ├── test_integration.py     # End-to-end integration tests
│   ├── test_server.py          # MCP tool registration tests
│   ├── test_sandbox.py         # Sandbox execution tests
│   ├── test_telemetry.py       # Telemetry logging tests
│   ├── test_config.py          # Configuration loading tests
│   ├── test_hooks.py           # Hook interception tests
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

## Cognitive Pipeline

```
Classify → Distill → Plan → Execute → Evaluate → Reflect
   │                                      │
   │ System 1 bypass ─────────────────────┘ (skip Distill+Plan)
   │
   └── retry loop (Evaluate → re-Execute) ──→ escalate after max_attempts
       backtrack loop (Evaluate → re-Plan → Execute)
```

| Stage     | Component   | LLM Model        | Purpose                                     |
|-----------|-------------|------------------|---------------------------------------------|
| Classify  | Classifier  | gpt-4.1-nano     | Route to System 1 (fast) or System 2 (slow) |
| Distill   | Distiller   | gpt-4.1-nano     | Structure raw task into template             |
| Plan      | Planner     | gpt-4.1-mini     | Generate execution strategies with rollbacks |
| Execute   | Sandbox     | —                | Sandboxed subprocess (deterministic)         |
| Evaluate  | Evaluator   | o3-mini          | LLM-as-Judge with adversarial checks         |
| Reflect   | Reflector   | o3-mini          | Extract rules/procedures from telemetry      |

## MCP Tools (9)

| Tool                       | Async | Description                                              |
|----------------------------|-------|----------------------------------------------------------|
| `metascaffold_classify`    | yes   | System 1/2 routing with LLM semantic analysis            |
| `metascaffold_distill`     | yes   | Structure raw task into template (objective, constraints) |
| `metascaffold_plan`        | yes   | Generate strategies with risks and rollback plans         |
| `metascaffold_sandbox_exec`| no    | Execute command in sandboxed subprocess                   |
| `metascaffold_evaluate`    | yes   | LLM-as-Judge verdict: pass/retry/backtrack/escalate       |
| `metascaffold_reflect`     | yes   | Extract learning patterns from telemetry events           |
| `metascaffold_nlm_query`   | yes   | Query NotebookLM knowledge base                           |
| `metascaffold_telemetry_query` | no | Query historical success rates from SQLite              |
| `metascaffold_restart`     | no    | Hot-reload all modules (no server restart needed)         |

## LLM Backend

**Engine:** `codex exec` subprocess (Codex CLI)
**Auth:** ChatGPT Plus subscription (no API key needed)
**Cost:** Zero additional cost (included in subscription)
**Structured output:** `--output-schema <schema.json> -o <output.json>` for guaranteed valid JSON

Each component defines a `_*_RESPONSE_SCHEMA` (JSON Schema with `additionalProperties: false`)
that enforces the output structure at the model level. The LLM client auto-adds
`additionalProperties: false` to schemas that don't include it.

**Fallback:** When codex is unavailable or the LLM call fails, every component falls back
silently to v0.1 heuristic behavior. The system never hard-fails on LLM unavailability.

## Configuration

Default config: `config/default_config.toml`
User override: `~/.metascaffold/config.toml` (optional, merged over defaults)

```toml
[classifier]
system2_threshold = 0.8           # Below this confidence → System 2
always_system2_tools = ["Write"]  # Always force System 2

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
```

## Key Architecture Decisions

1. **Cross-model verification**: Claude generates → Codex evaluates. Different models = uncorrelated biases.
2. **Deterministic intermediate layer**: Sandbox executes real code between generator and evaluator.
3. **Structured output via `--output-schema`**: JSON Schemas enforced at model level (not post-hoc parsing).
4. **Graceful degradation**: Every async LLM method has a sync heuristic fallback.
5. **Hot-reload**: `metascaffold_restart` reloads all modules without killing the MCP server process.

## Development

- **Python 3.11+** required
- **uv** for dependency management
- **pytest** + **pytest-asyncio** for testing
- All async tests use `asyncio_mode = "auto"` (see `pyproject.toml`)
- Mock `asyncio.create_subprocess_exec` for LLM client tests
- Mock `client.complete` return value for component tests

## Known Limitations

- **LLM confidence** is prompt-based (self-reported), not entropy/logprob-based
- **Reflector** accumulates rules without memory management (no forgetting curve)
- **Hot-reload** only reloads modules already in `sys.modules`; new tools require full server restart
- **Model parameter is ignored**: Codex exec uses whatever model is configured in Codex CLI
- **Hooks are advisory**: PreToolUse/PostToolUse hooks suggest but don't enforce classification/evaluation

## Research Foundations

Built on 2025-2026 dual-process AI research:
- **SOFAI-LM**: Slow/Fast AI metacognitive governance
- **SELF-THOUGHT**: Task distillation before self-correction
- **MARS**: Metacognitive Agent Reflective Self-improvement
- **PAG**: Plan-and-Gate selective revision (prevents model collapse)
- **Adversarial post-exec**: Counter confirmation bias in LLM evaluation

See `docs/research_sources.json` for the full bibliography.
