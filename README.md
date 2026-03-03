# MetaScaffold

> Metacognition MCP server plugin for Claude Code — adds a dual-process cognitive layer powered by LLM inference with entropy-based adaptive compute.

## What is MetaScaffold?

MetaScaffold is a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that gives Claude Code a **metacognitive layer**. Instead of executing every task the same way, Claude gains the ability to:

- **Classify** tasks using entropy-based 3-level routing (System 1 / 1.5 / 2)
- **Distill** raw task descriptions into structured templates
- **Plan** execution strategies with risks and rollback plans
- **Verify** results with deterministic checks (AST, Ruff, pytest, mypy) before LLM judgment
- **Evaluate** results using an independent LLM-as-Judge
- **Reflect** on past performance with Ebbinghaus memory decay

### Cross-Model Architecture with Dual Backend

```
┌──────────────────────┐         ┌──────────────────────────────────────────┐
│     Claude Code      │         │            MetaScaffold MCP              │
│   (Anthropic model)  │◄───────►│                                          │
│                      │  stdio  │  ┌────────────────┐  ┌────────────────┐  │
│   Writes code        │         │  │  codex exec    │  │  OpenAI API    │  │
│   Makes edits        │         │  │  (gpt-5.3)     │  │  (gpt-4.1-nano)│  │
│   Calls tools        │         │  │                │  │                │  │
│   ► GENERATOR        │         │  │  ALL LLM calls │  │  Entropy probe │  │
│                      │         │  │  (default)     │  │  (logprobs)    │  │
│                      │         │  └────────┬───────┘  └───────┬────────┘  │
│                      │         │           │                  │           │
│                      │         │     ┌─────▼──────────────────▼─────┐     │
│                      │         │     │   Classify → Distill → Plan  │     │
│                      │         │     │   Execute → Verify → Evaluate│     │
│                      │         │     │   Reflect (with memory decay)│     │
│                      │         │     │   ► METACOGNITIVE LAYER      │     │
│                      │         │     └──────────────────────────────┘     │
└──────────────────────┘         └──────────────────────────────────────────┘
                                          │
                                   ┌──────┴──────┐
                                   │   Sandbox   │
                                   │  (real code │
                                   │  execution) │
                                   └─────────────┘
```

Generator (Claude) and evaluator (Codex) are **different models from different vendors**, so their failure modes are uncorrelated — a structural advantage over single-model self-evaluation.

## v0.3 Features

- **Dual Backend LLM Client** — `codex exec` (gpt-5.3, free via ChatGPT Plus) for all LLM calls + OpenAI API (gpt-4.1-nano) for entropy logprobs only
- **Entropy-Based Routing** — Shannon entropy from token logprobs measures routing uncertainty: `H = -sum(p * log2(p))`
- **3-Level Adaptive Compute** — System 1 (skip Distill+Plan), System 1.5 (skip Plan), System 2 (full pipeline)
- **Deterministic Verifiers** — AST parse, Ruff lint, pytest, mypy run BEFORE LLM-as-Judge; critical failures short-circuit without LLM call
- **Ebbinghaus Memory Decay** — Learned rules fade naturally via forgetting curve; pruned when retention < 0.1

## Installation

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), [Codex CLI](https://github.com/openai/codex) with ChatGPT Plus subscription.

**Optional:** `OPEN_API_KEY` environment variable for entropy-based routing (falls back to codex LLM without it).

```bash
# Clone and install
git clone https://github.com/yannabadie/metascaffold.git
cd metascaffold
uv sync

# (Optional) Set OpenAI API key for entropy routing
export OPEN_API_KEY="sk-..."

# Register with Claude Code
claude mcp add metascaffold -- uv --directory $(pwd) run python src/metascaffold/server.py
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `metascaffold_classify` | 3-level routing with entropy probe + codex LLM + heuristic fallback |
| `metascaffold_distill` | Structure raw task into template with objectives and constraints |
| `metascaffold_plan` | Generate execution strategies with risks and rollback plans |
| `metascaffold_sandbox_exec` | Execute command in sandboxed subprocess |
| `metascaffold_evaluate` | Deterministic verifiers + LLM-as-Judge verdict with adversarial checks |
| `metascaffold_reflect` | Extract learning rules into ReflectionMemory with decay |
| `metascaffold_nlm_query` | Query NotebookLM knowledge base |
| `metascaffold_telemetry_query` | Query historical task success rates |
| `metascaffold_restart` | Hot-reload all modules without server restart |

## Cognitive Pipeline (3-Level Compute)

```
 ┌─────────┐    ┌─────────┐    ┌──────┐    ┌─────────┐    ┌────────┐    ┌──────────┐    ┌─────────┐
 │Classify │───►│ Distill │───►│ Plan │───►│ Execute │───►│ Verify │───►│ Evaluate │───►│ Reflect │
 └────┬────┘    └────┬────┘    └──────┘    └─────────┘    └────────┘    └────┬─────┘    └─────────┘
      │              │                                                       │
      │ System 1     │───────────────────────────────────────────────────────┘
      │ (H < 0.3: skip Distill + Plan)
      │              │
      │ System 1.5   └──────────────────► Plan skipped, Distill kept
      │ (0.3 ≤ H < 0.5)
      │
      └── Evaluate: retry ──► re-Execute
          Evaluate: backtrack ──► re-Plan ──► Execute
          Evaluate: escalate ──► stop (human needed)
          Verify: critical fail ──► backtrack (no LLM call)
```

## Configuration

Copy `config/default_config.toml` to `~/.metascaffold/config.toml` to customize:

```toml
[classifier]
system2_threshold = 0.8
always_system2_tools = ["Write"]
entropy_threshold = 0.5           # H >= 0.5 → System 2
medium_entropy_threshold = 0.3    # H >= 0.3 → System 1.5

[verifier]
run_ast = true
run_ruff = true
run_mypy = false
run_pytest = true

[memory]
prune_threshold = 0.1
stability_hours = 168
storage_path = "~/.metascaffold/reflection_memory.json"

[llm]
enabled = true
fallback_to_heuristics = true
```

## Development

```bash
# Run tests (175 tests)
uv run pytest -v

# Run MCP server standalone
uv run python src/metascaffold/server.py
```

## Research Foundations

Built on 2025-2026 dual-process AI architecture research:

- **SOFAI-LM** — Slow/Fast AI metacognitive governance
- **SELF-THOUGHT** — Task distillation before self-correction
- **MARS** — Metacognitive Agent Reflective Self-improvement
- **PAG** — Plan-and-Gate selective revision control
- **Adversarial post-execution** — Counter confirmation bias in evaluation
- **Ebbinghaus forgetting curve** — Natural memory decay for learned rules

## License

MIT
