# MetaScaffold

> Metacognition MCP server plugin for Claude Code — adds a dual-process cognitive layer powered by LLM inference.

## What is MetaScaffold?

MetaScaffold is a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that gives Claude Code a **metacognitive layer**. Instead of executing every task the same way, Claude gains the ability to:

- **Classify** tasks as simple (System 1) or complex (System 2)
- **Distill** raw task descriptions into structured templates
- **Plan** execution strategies with risks and rollback plans
- **Evaluate** results using an independent LLM-as-Judge
- **Reflect** on past performance to learn from mistakes

### Cross-Model Architecture

```
┌──────────────────────┐         ┌──────────────────────────────────┐
│     Claude Code      │         │         MetaScaffold MCP         │
│   (Anthropic model)  │◄───────►│       (Codex gpt-5.3 model)     │
│                      │  stdio  │                                  │
│   Writes code        │         │   Classifies tasks               │
│   Makes edits        │         │   Plans strategies               │
│   Calls tools        │         │   Evaluates results              │
│   ► GENERATOR        │         │   Reflects on telemetry          │
│                      │         │   ► METACOGNITIVE LAYER          │
└──────────────────────┘         └──────────────────────────────────┘
                                          │
                                   ┌──────┴──────┐
                                   │   Sandbox   │
                                   │  (real code │
                                   │  execution) │
                                   └─────────────┘
```

Generator (Claude) and evaluator (Codex) are **different models from different vendors**, so their failure modes are uncorrelated — a structural advantage over single-model self-evaluation.

## Installation

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/), [Codex CLI](https://github.com/openai/codex) with ChatGPT Plus subscription.

```bash
# Clone and install
git clone https://github.com/yannabadie/metascaffold.git
cd metascaffold
uv sync

# Register with Claude Code
claude mcp add metascaffold -- uv --directory $(pwd) run python src/metascaffold/server.py
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `metascaffold_classify` | Route task to System 1 (fast) or System 2 (deliberate) |
| `metascaffold_distill` | Structure raw task into template with objectives and constraints |
| `metascaffold_plan` | Generate execution strategies with risks and rollback plans |
| `metascaffold_sandbox_exec` | Execute command in sandboxed subprocess |
| `metascaffold_evaluate` | LLM-as-Judge verdict with adversarial checks |
| `metascaffold_reflect` | Extract learning rules from cognitive telemetry |
| `metascaffold_nlm_query` | Query NotebookLM knowledge base |
| `metascaffold_telemetry_query` | Query historical task success rates |
| `metascaffold_restart` | Hot-reload all modules without server restart |

## Cognitive Pipeline

```
 ┌─────────┐    ┌─────────┐    ┌──────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐
 │Classify │───►│ Distill │───►│ Plan │───►│ Execute │───►│ Evaluate │───►│ Reflect │
 └────┬────┘    └─────────┘    └──────┘    └─────────┘    └────┬─────┘    └─────────┘
      │                                                        │
      │ System 1 ──────────────────────────────────────────────┘
      │ (bypass Distill + Plan)
      │
      └── Evaluate: retry ──► re-Execute
          Evaluate: backtrack ──► re-Plan ──► Execute
          Evaluate: escalate ──► stop (human needed)
```

## Configuration

Copy `config/default_config.toml` to `~/.metascaffold/config.toml` to customize:

```toml
[classifier]
system2_threshold = 0.8
always_system2_tools = ["Write"]

[llm]
enabled = true
fallback_to_heuristics = true
```

## Development

```bash
# Run tests (120 tests)
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

## License

MIT
