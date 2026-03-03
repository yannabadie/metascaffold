# MetaScaffold — Claude Code Instructions

## Overview
MetaScaffold is a metacognition MCP server plugin for Claude Code.
It implements dual-process (System 1/System 2) cognitive architecture.

## Project Structure
- `src/metascaffold/` — Main package (MCP server + components)
- `hooks/` — Claude Code hooks (PreToolUse, PostToolUse)
- `tests/` — pytest test suite
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

## Architecture
See `docs/plans/2026-03-03-metascaffold-design.md` for full design.
