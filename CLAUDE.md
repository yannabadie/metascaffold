# MetaScaffold — Claude Code Instructions

## Overview
MetaScaffold is a metacognition MCP server plugin for Claude Code.
It implements a **6-stage LLM-powered cognitive pipeline**: Classify → Distill → Plan → Execute → Evaluate → Reflect.

**Version:** 0.2.0 — All components use real LLM calls (via `codex exec`) with heuristic fallback.

## Project Structure
- `src/metascaffold/` — Main package (MCP server + 7 cognitive components)
  - `server.py` — FastMCP server exposing 9 tools
  - `llm_client.py` — LLM abstraction over Codex CLI subprocess
  - `classifier.py` — System 1/2 routing (LLM + heuristic)
  - `distiller.py` — Task structuring (Self-Thought Distillation)
  - `planner.py` — Strategy generation (LLM + heuristic templates)
  - `evaluator.py` — LLM-as-Judge with SOFAI feedback, adversarial check, PAG gate
  - `reflector.py` — MARS reflection loop for telemetry learning
  - `pipeline.py` — 6-stage cognitive orchestrator
  - `sandbox.py` — Sandboxed command execution
  - `telemetry.py` — Event logging (JSON + SQLite)
  - `notebooklm_bridge.py` — NotebookLM knowledge base integration
  - `config.py` — Configuration management
- `hooks/` — Claude Code hooks (PreToolUse, PostToolUse)
- `tests/` — pytest test suite (116+ tests)
- `config/` — Default TOML configuration
- `docs/plans/` — Design and implementation documents

## Running
```bash
# Install dependencies
uv sync

# Run MCP server
uv run python src/metascaffold/server.py

# Run tests
uv run pytest -v
```

## MCP Tools (9)
1. `metascaffold_classify` — LLM-powered System 1/2 task routing
2. `metascaffold_distill` — Structure raw tasks into templates
3. `metascaffold_plan` — Generate context-aware execution strategies
4. `metascaffold_sandbox_exec` — Execute commands in sandbox
5. `metascaffold_evaluate` — LLM-as-Judge evaluation with SOFAI feedback
6. `metascaffold_reflect` — Extract rules/procedures from telemetry
7. `metascaffold_nlm_query` — Query NotebookLM knowledge base
8. `metascaffold_telemetry_query` — Query historical success rates
9. `metascaffold_restart` — Hot-reload all modules

## LLM Backend
Uses `codex exec` subprocess (ChatGPT Plus subscription, no API key needed).
All components gracefully degrade to v0.1 heuristics when LLM is unavailable.

## Architecture
- `docs/plans/2026-03-03-metascaffold-design.md` — Original v0.1 design
- `docs/plans/2026-03-03-metascaffold-v02-llm-pipeline.md` — v0.2 LLM pipeline plan
