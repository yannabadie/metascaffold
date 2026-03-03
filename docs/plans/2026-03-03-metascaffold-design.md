# MetaScaffold — Design Document

**Date**: 2026-03-03
**Author**: Yann Abadie (Solutions Architect) + Claude Code
**Status**: Approved

---

## 1. Purpose

MetaScaffold is a **metacognition plugin for Claude Code** that implements dual-process cognitive architecture (System 1 / System 2). It forces self-evaluation, planning, and self-correction before and during task execution.

**Goals:**
- Automatically detect when a task requires deep deliberation vs. fast execution
- Enforce structured planning before high-risk actions
- Provide sandboxed execution for experimental code
- Log cognitive telemetry (confidence levels, strategy changes, backtracking)
- Leverage NotebookLM as an external knowledge base for research-heavy tasks

---

## 2. Architecture Overview

**Pattern: Middleware Cognitif** — MCP Server + Claude Code Hooks

```
User Request → Hooks (PreToolUse) → MCP Server "MetaScaffold"
                                        ├── Classifier (System 1/2 routing)
                                        ├── Planner (decomposition)
                                        ├── Executor (worktree + subprocess)
                                        ├── Evaluator (auto-critique)
                                        └── Telemetry (JSON + SQLite)
                                    ↕
                              NotebookLM (external brain)
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Integration | Hybrid Hooks + MCP | Hooks intercept automatically; MCP provides cognitive tools |
| Sandboxing | Git Worktrees + Subprocess | Double isolation without Docker overhead |
| Telemetry | JSON + SQLite | JSON for real-time logs; SQLite for inter-session learning |
| NotebookLM | notebooklm-py (v0.3.2) | Most mature unofficial API; graceful degradation if unavailable |
| Language | Python 3.11+ | Matches notebooklm-py requirements; rich MCP ecosystem |

---

## 3. Components

### 3.1 Classifier (System 1/2 Router)

**MCP Tool**: `metascaffold_classify`

Determines whether a task requires deep deliberation (System 2) or can proceed directly (System 1).

**Routing Signals:**

| Signal | System 1 (fast) | System 2 (deliberate) |
|--------|-----------------|----------------------|
| Complexity | Atomic task, 1-2 files | Multi-file, architecture |
| Uncertainty | High confidence (known pattern) | Low confidence, ambiguity |
| Reversibility | Easily reversible | Destructive or hard to undo |
| History | Similar tasks succeeded | Error pattern detected |
| Criticality | Dev/test | Production, security, data |

**Output Schema:**
```json
{
    "routing": "system2",
    "confidence": 0.72,
    "reasoning": "Multi-file refactor with unclear scope",
    "signals": {
        "complexity": "high",
        "reversibility": "medium",
        "uncertainty": "high",
        "historical_success_rate": 0.6
    }
}
```

**Configurable threshold:** Default confidence < 0.8 → System 2.

### 3.2 Planner (Decomposition & Strategy)

**MCP Tool**: `metascaffold_plan`

For System 2 tasks, produces a structured plan before any action.

**Process:**
1. Decompose task into atomic sub-tasks
2. Identify risks for each sub-task
3. Propose 2-3 alternative strategies
4. Estimate confidence for each strategy
5. Consult NotebookLM if domain is covered by knowledge base

**Output Schema:**
```json
{
    "task": "Refactor authentication system",
    "strategies": [
        {
            "id": "A",
            "description": "Progressive migration with adapter pattern",
            "steps": ["..."],
            "confidence": 0.85,
            "risks": ["Adapter overhead", "Dual-path bugs"],
            "rollback_plan": "Revert adapter, restore original"
        }
    ],
    "recommended": "A",
    "notebooklm_insights": "..."
}
```

### 3.3 Sandbox (Execution Isolation)

**MCP Tool**: `metascaffold_sandbox_exec`

Executes experimental code without risk to the main environment.

**Double layer:**
1. **Git Worktree** — Source code isolation (temporary branch, clean merge or delete)
2. **Restricted Subprocess** — Execution isolation (timeout, memory limit, optional network block)

**Input/Output:**
```json
// Input
{
    "code": "python test_experiment.py",
    "timeout_seconds": 30,
    "memory_limit_mb": 512,
    "network_access": false,
    "worktree": true
}
// Output
{
    "exit_code": 0,
    "stdout": "...",
    "stderr": "...",
    "duration_ms": 1234,
    "memory_peak_mb": 45,
    "worktree_path": "/tmp/metascaffold-wt-abc123",
    "worktree_branch": "metascaffold/experiment-001"
}
```

### 3.4 Evaluator (Auto-Critique & Correction)

**MCP Tool**: `metascaffold_evaluate`

After execution, evaluates the result and decides: accept, correct, or backtrack.

**Verdicts:**
- `pass` — Result accepted, merge worktree
- `retry` — Same strategy, minor corrections (max 3 attempts)
- `backtrack` — Change strategy, return to Planner
- `escalate` — Request human intervention

**Output Schema:**
```json
{
    "verdict": "retry",
    "confidence": 0.55,
    "issues": [
        {"type": "test_failure", "detail": "3/12 tests failing", "severity": "medium"}
    ],
    "corrections": [
        {"file": "auth.py", "line": 42, "suggestion": "Missing null check"}
    ],
    "attempt": 2,
    "max_attempts": 3,
    "strategy_alternatives_remaining": 1
}
```

### 3.5 Telemetry (Cognitive Journal)

**Dual storage:**

| Layer | Format | Content | Usage |
|-------|--------|---------|-------|
| Real-time | JSON (1 file/session) | Each timestamped event | Readable, debug, git-trackable |
| Historical | SQLite | Aggregates by task type, error patterns, confidence trends | Inter-session learning, feeds Classifier |

**Tracked events:** `classification`, `plan_created`, `strategy_selected`, `execution_result`, `evaluation`, `backtrack`, `escalation`

**Paths:** `~/.metascaffold/telemetry/` (JSON), `~/.metascaffold/cognitive.db` (SQLite)

### 3.6 NotebookLM Bridge

**MCP Tools**: `metascaffold_nlm_query`, `metascaffold_nlm_upload`, `metascaffold_nlm_create`

Interface with `notebooklm-py` for knowledge-enriched reflection.

**Graceful degradation:** If NotebookLM is unavailable (expired cookies, broken API), Bridge returns empty result and Planner continues without it. No blocking.

---

## 4. Integration

### 4.1 Claude Code Hooks

Declared in `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|Edit|Write",
        "command": "python ~/.metascaffold/hooks/pre_tool_gate.py"
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash|Edit|Write",
        "command": "python ~/.metascaffold/hooks/post_tool_evaluate.py"
      }
    ]
  }
}
```

**Key points:**
- Only match `Bash|Edit|Write` (environment-modifying actions)
- Read tools (Read, Grep, Glob) pass without interception
- Hook scripts are lightweight (~50 lines) — delegate to MCP server

### 4.2 Configuration

File: `~/.metascaffold/config.toml`

```toml
[classifier]
system2_threshold = 0.8
always_system2_tools = ["Write"]

[sandbox]
default_timeout_seconds = 30
default_memory_limit_mb = 512
network_access = false
max_retry_attempts = 3

[telemetry]
json_dir = "~/.metascaffold/telemetry/"
sqlite_path = "~/.metascaffold/cognitive.db"
log_level = "info"

[notebooklm]
enabled = true
default_notebook = "MetaScaffold_Core"
fallback_on_error = true

[mcp_server]
host = "127.0.0.1"
port = 8787
```

### 4.3 Data Flow

```
1. User asks "Refactor the auth module"
2. Claude prepares Edit call on auth.py
3. PreToolUse hook fires → calls Classifier
4. Classifier: confidence=0.65 → SYSTEM 2
5. Hook returns message: "System 2 activated. Use metascaffold_plan first."
6. Claude calls metascaffold_plan → Planner proposes 2 strategies
7. Claude selects strategy A, begins execution
8. For risky code → metascaffold_sandbox_exec in worktree
9. PostToolUse hook → calls Evaluator
10. Evaluator: "3 tests failing" → verdict "retry" + corrections
11. Claude applies corrections
12. PostToolUse hook → Evaluator: "all tests pass" → verdict "pass"
13. Worktree merged, telemetry logs entire cycle
```

---

## 5. Project Structure

```
MetaScaffold/
├── pyproject.toml
├── README.md
├── CLAUDE.md
│
├── src/
│   └── metascaffold/
│       ├── __init__.py
│       ├── server.py                 # MCP Server entry point
│       ├── classifier.py             # System 1/2 routing
│       ├── planner.py                # Decomposition & strategy
│       ├── sandbox.py                # Worktree + subprocess isolation
│       ├── evaluator.py              # Auto-critique & correction
│       ├── telemetry.py              # JSON + SQLite logging
│       ├── notebooklm_bridge.py      # notebooklm-py interface
│       └── config.py                 # TOML configuration
│
├── hooks/
│   ├── pre_tool_gate.py              # PreToolUse hook
│   └── post_tool_evaluate.py         # PostToolUse hook
│
├── tests/
│   ├── test_classifier.py
│   ├── test_planner.py
│   ├── test_sandbox.py
│   ├── test_evaluator.py
│   ├── test_telemetry.py
│   └── test_integration.py
│
├── docs/
│   └── plans/
│
└── config/
    └── default_config.toml
```

---

## 6. Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11+ |
| MCP SDK | mcp[cli] | latest |
| NotebookLM | notebooklm-py | 0.3.2 |
| Config | tomli / tomllib | stdlib (3.11+) |
| Database | SQLite | stdlib |
| Testing | pytest | latest |
| Package | pyproject.toml + uv | latest |

---

## 7. Phased Implementation

### Phase 1: Infrastructure & NotebookLM Connection
- Install notebooklm-py
- Authenticate user
- Verify API connectivity

### Phase 2: Knowledge Base Ingestion
- Source 15 research papers (ArXiv) on LLM metacognition
- Upload to NotebookLM "MetaScaffold_Core" notebook
- Extract key concepts

### Phase 3: Blueprint & Architecture
- Generate detailed architecture from knowledge base
- Validate security (sandboxing, isolation)
- Validate telemetry design

### Phase 4: Repository Initialization
- Scaffold project structure
- Implement MCP server with all 6 components
- Implement hooks
- Write unit tests
- Validate process isolation
