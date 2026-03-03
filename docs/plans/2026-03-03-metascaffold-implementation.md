# MetaScaffold Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a metacognition MCP server plugin for Claude Code with dual-process (System 1/2) cognitive architecture, sandboxed execution, telemetry, and NotebookLM integration.

**Architecture:** Middleware Cognitif pattern — a Python MCP server exposes 6 cognitive tools (classify, plan, sandbox, evaluate, telemetry, notebooklm). Claude Code hooks (PreToolUse/PostToolUse) intercept modifying actions and route them through the server. Git worktrees + restricted subprocesses provide execution isolation.

**Tech Stack:** Python 3.11+, mcp[cli] (FastMCP), notebooklm-py, pytest, SQLite, tomllib, pydantic

---

## Phase 1: Infrastructure & NotebookLM Connection

### Task 1: Scaffold the Python Project

**Files:**
- Create: `pyproject.toml`
- Create: `src/metascaffold/__init__.py`
- Create: `config/default_config.toml`
- Create: `.gitignore`
- Create: `CLAUDE.md`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "metascaffold"
version = "0.1.0"
description = "Metacognition plugin for Claude Code — dual-process System 1/2 architecture"
requires-python = ">=3.11"
dependencies = [
    "mcp[cli]>=1.2.0",
    "pydantic>=2.0",
    "notebooklm-py>=0.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/metascaffold"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**Step 2: Create package init**

```python
# src/metascaffold/__init__.py
"""MetaScaffold — Metacognition plugin for Claude Code."""

__version__ = "0.1.0"
```

**Step 3: Create default config**

```toml
# config/default_config.toml

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

**Step 4: Create .gitignore**

```gitignore
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.venv/
venv/
.env
*.db
.pytest_cache/
```

**Step 5: Create CLAUDE.md**

```markdown
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
```

**Step 6: Install dependencies and verify**

Run: `cd C:/Code/MetaScaffold && uv sync`
Expected: Dependencies installed, `.venv` created

**Step 7: Commit**

```bash
git add pyproject.toml src/metascaffold/__init__.py config/default_config.toml .gitignore CLAUDE.md
git commit -m "feat: scaffold MetaScaffold project structure"
```

---

### Task 2: Config Module

**Files:**
- Create: `tests/test_config.py`
- Create: `src/metascaffold/config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
"""Tests for the configuration module."""

import os
import tempfile
from pathlib import Path

from metascaffold.config import MetaScaffoldConfig, load_config


class TestMetaScaffoldConfig:
    def test_default_config_loads(self):
        """Default config file should parse without errors."""
        config = load_config()
        assert config.classifier.system2_threshold == 0.8
        assert config.sandbox.default_timeout_seconds == 30
        assert config.telemetry.log_level == "info"
        assert config.notebooklm.enabled is True

    def test_custom_config_overrides_defaults(self, tmp_path):
        """User config should override default values."""
        custom = tmp_path / "config.toml"
        custom.write_text('[classifier]\nsystem2_threshold = 0.6\n')
        config = load_config(custom)
        assert config.classifier.system2_threshold == 0.6
        # Non-overridden values keep defaults
        assert config.sandbox.default_timeout_seconds == 30

    def test_config_expands_home_paths(self):
        """Paths with ~ should expand to user home directory."""
        config = load_config()
        assert "~" not in config.telemetry.json_dir
        assert Path(config.telemetry.json_dir).is_absolute()

    def test_config_has_correct_types(self):
        """Config fields should have correct Python types."""
        config = load_config()
        assert isinstance(config.classifier.system2_threshold, float)
        assert isinstance(config.classifier.always_system2_tools, list)
        assert isinstance(config.sandbox.network_access, bool)
        assert isinstance(config.notebooklm.fallback_on_error, bool)
```

**Step 2: Run test to verify it fails**

Run: `cd C:/Code/MetaScaffold && uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'metascaffold.config'`

**Step 3: Write minimal implementation**

```python
# src/metascaffold/config.py
"""Configuration management for MetaScaffold.

Loads TOML configuration with defaults from config/default_config.toml,
optionally overridden by user config at ~/.metascaffold/config.toml.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "config" / "default_config.toml"
_USER_CONFIG = Path.home() / ".metascaffold" / "config.toml"


@dataclass
class ClassifierConfig:
    system2_threshold: float = 0.8
    always_system2_tools: list[str] = field(default_factory=lambda: ["Write"])


@dataclass
class SandboxConfig:
    default_timeout_seconds: int = 30
    default_memory_limit_mb: int = 512
    network_access: bool = False
    max_retry_attempts: int = 3


@dataclass
class TelemetryConfig:
    json_dir: str = ""
    sqlite_path: str = ""
    log_level: str = "info"


@dataclass
class NotebookLMConfig:
    enabled: bool = True
    default_notebook: str = "MetaScaffold_Core"
    fallback_on_error: bool = True


@dataclass
class McpServerConfig:
    host: str = "127.0.0.1"
    port: int = 8787


@dataclass
class MetaScaffoldConfig:
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    notebooklm: NotebookLMConfig = field(default_factory=NotebookLMConfig)
    mcp_server: McpServerConfig = field(default_factory=McpServerConfig)


def _expand_path(p: str) -> str:
    """Expand ~ and make absolute."""
    return str(Path(p).expanduser().resolve())


def _merge_dicts(base: dict, override: dict) -> dict:
    """Deep merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_config(data: dict) -> MetaScaffoldConfig:
    """Convert a raw dict to a typed MetaScaffoldConfig."""
    cfg = MetaScaffoldConfig()

    if "classifier" in data:
        cfg.classifier = ClassifierConfig(**data["classifier"])

    if "sandbox" in data:
        cfg.sandbox = SandboxConfig(**data["sandbox"])

    if "telemetry" in data:
        t = data["telemetry"]
        cfg.telemetry = TelemetryConfig(
            json_dir=_expand_path(t.get("json_dir", "~/.metascaffold/telemetry/")),
            sqlite_path=_expand_path(t.get("sqlite_path", "~/.metascaffold/cognitive.db")),
            log_level=t.get("log_level", "info"),
        )
    else:
        cfg.telemetry = TelemetryConfig(
            json_dir=_expand_path("~/.metascaffold/telemetry/"),
            sqlite_path=_expand_path("~/.metascaffold/cognitive.db"),
        )

    if "notebooklm" in data:
        cfg.notebooklm = NotebookLMConfig(**data["notebooklm"])

    if "mcp_server" in data:
        cfg.mcp_server = McpServerConfig(**data["mcp_server"])

    return cfg


def load_config(config_path: Path | None = None) -> MetaScaffoldConfig:
    """Load configuration from TOML files.

    Priority: config_path > ~/.metascaffold/config.toml > default_config.toml
    """
    # Load defaults
    base_data: dict = {}
    if _DEFAULT_CONFIG.exists():
        with open(_DEFAULT_CONFIG, "rb") as f:
            base_data = tomllib.load(f)

    # Load user/custom override
    override_path = config_path or _USER_CONFIG
    override_data: dict = {}
    if override_path.exists():
        with open(override_path, "rb") as f:
            override_data = tomllib.load(f)

    merged = _merge_dicts(base_data, override_data)
    return _dict_to_config(merged)
```

**Step 4: Run test to verify it passes**

Run: `cd C:/Code/MetaScaffold && uv run pytest tests/test_config.py -v`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add src/metascaffold/config.py tests/test_config.py
git commit -m "feat: add config module with TOML loading and defaults"
```

---

### Task 3: Telemetry Module

**Files:**
- Create: `tests/test_telemetry.py`
- Create: `src/metascaffold/telemetry.py`

**Step 1: Write the failing test**

```python
# tests/test_telemetry.py
"""Tests for the telemetry (cognitive journal) module."""

import json
import sqlite3
from pathlib import Path

from metascaffold.telemetry import TelemetryLogger, CognitiveEvent


class TestTelemetryLogger:
    def test_log_event_creates_json_file(self, tmp_path):
        """Logging an event should create a session JSON file."""
        logger = TelemetryLogger(
            json_dir=str(tmp_path / "telemetry"),
            sqlite_path=str(tmp_path / "cognitive.db"),
        )
        logger.log(CognitiveEvent(
            event_type="classification",
            data={"routing": "system2", "confidence": 0.72},
        ))
        logger.flush()

        json_files = list((tmp_path / "telemetry").glob("*.json"))
        assert len(json_files) == 1

        with open(json_files[0]) as f:
            events = json.load(f)
        assert len(events) == 1
        assert events[0]["event_type"] == "classification"
        assert events[0]["data"]["confidence"] == 0.72
        assert "timestamp" in events[0]

    def test_log_event_writes_to_sqlite(self, tmp_path):
        """Events should also be persisted to SQLite."""
        logger = TelemetryLogger(
            json_dir=str(tmp_path / "telemetry"),
            sqlite_path=str(tmp_path / "cognitive.db"),
        )
        logger.log(CognitiveEvent(
            event_type="evaluation",
            data={"verdict": "pass", "confidence": 0.9},
        ))
        logger.flush()

        conn = sqlite3.connect(tmp_path / "cognitive.db")
        rows = conn.execute("SELECT event_type, confidence FROM events").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0] == ("evaluation", 0.9)

    def test_multiple_events_same_session(self, tmp_path):
        """Multiple events in one session go to the same JSON file."""
        logger = TelemetryLogger(
            json_dir=str(tmp_path / "telemetry"),
            sqlite_path=str(tmp_path / "cognitive.db"),
        )
        logger.log(CognitiveEvent(event_type="classification", data={"routing": "system1"}))
        logger.log(CognitiveEvent(event_type="plan_created", data={"strategies": 2}))
        logger.flush()

        json_files = list((tmp_path / "telemetry").glob("*.json"))
        assert len(json_files) == 1
        with open(json_files[0]) as f:
            events = json.load(f)
        assert len(events) == 2

    def test_query_historical_success_rate(self, tmp_path):
        """Should query SQLite for historical success rate by task type."""
        logger = TelemetryLogger(
            json_dir=str(tmp_path / "telemetry"),
            sqlite_path=str(tmp_path / "cognitive.db"),
        )
        # Simulate history: 3 passes, 1 fail for "refactor" tasks
        for verdict in ["pass", "pass", "pass", "retry"]:
            logger.log(CognitiveEvent(
                event_type="evaluation",
                data={"verdict": verdict, "task_type": "refactor", "confidence": 0.8},
            ))
        logger.flush()

        rate = logger.get_success_rate("refactor")
        assert rate == 0.75  # 3 out of 4

    def test_success_rate_returns_none_for_unknown(self, tmp_path):
        """Unknown task types should return None (no data)."""
        logger = TelemetryLogger(
            json_dir=str(tmp_path / "telemetry"),
            sqlite_path=str(tmp_path / "cognitive.db"),
        )
        assert logger.get_success_rate("unknown_type") is None
```

**Step 2: Run test to verify it fails**

Run: `cd C:/Code/MetaScaffold && uv run pytest tests/test_telemetry.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/metascaffold/telemetry.py
"""Cognitive telemetry — JSON session logs + SQLite historical database.

Tracks classification decisions, plans, evaluations, backtracks, and escalations.
The Classifier uses historical data to improve routing accuracy over time.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class CognitiveEvent:
    """A single cognitive telemetry event."""
    event_type: str  # classification, plan_created, strategy_selected, execution_result, evaluation, backtrack, escalation
    data: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class TelemetryLogger:
    """Dual-layer telemetry: JSON for real-time, SQLite for history."""

    def __init__(self, json_dir: str, sqlite_path: str):
        self._json_dir = Path(json_dir)
        self._json_dir.mkdir(parents=True, exist_ok=True)
        self._sqlite_path = Path(sqlite_path)
        self._session_id = uuid.uuid4().hex[:12]
        self._events: list[dict] = []
        self._init_db()

    def _init_db(self) -> None:
        """Create SQLite tables if they don't exist."""
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._sqlite_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                confidence REAL,
                task_type TEXT,
                data_json TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def log(self, event: CognitiveEvent) -> None:
        """Log a cognitive event to both JSON buffer and SQLite."""
        record = {
            "session_id": self._session_id,
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "data": event.data,
        }
        self._events.append(record)

        # Write to SQLite immediately
        conn = sqlite3.connect(self._sqlite_path)
        conn.execute(
            "INSERT INTO events (session_id, timestamp, event_type, confidence, task_type, data_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                self._session_id,
                event.timestamp,
                event.event_type,
                event.data.get("confidence"),
                event.data.get("task_type"),
                json.dumps(event.data),
            ),
        )
        conn.commit()
        conn.close()

    def flush(self) -> None:
        """Write buffered events to the session JSON file."""
        if not self._events:
            return
        json_path = self._json_dir / f"session_{self._session_id}.json"
        with open(json_path, "w") as f:
            json.dump(self._events, f, indent=2)

    def get_success_rate(self, task_type: str) -> float | None:
        """Query historical success rate for a given task type.

        Returns float (0.0-1.0) or None if no data exists.
        """
        conn = sqlite3.connect(self._sqlite_path)
        row = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN json_extract(data_json, '$.verdict') = 'pass' THEN 1 ELSE 0 END) as passes
            FROM events
            WHERE event_type = 'evaluation' AND task_type = ?
            """,
            (task_type,),
        ).fetchone()
        conn.close()

        if row is None or row[0] == 0:
            return None
        return row[1] / row[0]
```

**Step 4: Run test to verify it passes**

Run: `cd C:/Code/MetaScaffold && uv run pytest tests/test_telemetry.py -v`
Expected: 5 tests PASS

**Step 5: Commit**

```bash
git add src/metascaffold/telemetry.py tests/test_telemetry.py
git commit -m "feat: add telemetry module with JSON + SQLite dual logging"
```

---

### Task 4: NotebookLM Bridge Module

**Files:**
- Create: `tests/test_notebooklm_bridge.py`
- Create: `src/metascaffold/notebooklm_bridge.py`

**Step 1: Write the failing test**

```python
# tests/test_notebooklm_bridge.py
"""Tests for the NotebookLM bridge module.

These tests mock notebooklm-py to avoid requiring auth.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metascaffold.notebooklm_bridge import NotebookLMBridge, BridgeResult


class TestNotebookLMBridge:
    def test_bridge_creation_with_defaults(self):
        """Bridge should initialize with default config."""
        bridge = NotebookLMBridge(enabled=True, default_notebook="Test")
        assert bridge.enabled is True
        assert bridge.default_notebook == "Test"

    def test_bridge_disabled_returns_empty(self):
        """When disabled, all operations return empty BridgeResult."""
        bridge = NotebookLMBridge(enabled=False, default_notebook="Test")
        result = bridge.query_sync("What is metacognition?")
        assert result.success is False
        assert result.content == ""
        assert "disabled" in result.reason.lower()

    @patch("metascaffold.notebooklm_bridge._get_client")
    def test_query_returns_content_on_success(self, mock_get_client):
        """Successful query should return content from NotebookLM."""
        mock_client = MagicMock()
        mock_client.chat.return_value = MagicMock(text="Metacognition is thinking about thinking.")
        mock_get_client.return_value = mock_client

        bridge = NotebookLMBridge(enabled=True, default_notebook="Test")
        result = bridge.query_sync("What is metacognition?")
        assert result.success is True
        assert "metacognition" in result.content.lower()

    @patch("metascaffold.notebooklm_bridge._get_client")
    def test_query_graceful_degradation_on_error(self, mock_get_client):
        """Errors should return empty result, not raise exceptions."""
        mock_get_client.side_effect = Exception("Auth expired")

        bridge = NotebookLMBridge(
            enabled=True,
            default_notebook="Test",
            fallback_on_error=True,
        )
        result = bridge.query_sync("test")
        assert result.success is False
        assert "Auth expired" in result.reason

    @patch("metascaffold.notebooklm_bridge._get_client")
    def test_query_raises_when_fallback_disabled(self, mock_get_client):
        """With fallback_on_error=False, errors should propagate."""
        mock_get_client.side_effect = Exception("Auth expired")

        bridge = NotebookLMBridge(
            enabled=True,
            default_notebook="Test",
            fallback_on_error=False,
        )
        with pytest.raises(Exception, match="Auth expired"):
            bridge.query_sync("test")
```

**Step 2: Run test to verify it fails**

Run: `cd C:/Code/MetaScaffold && uv run pytest tests/test_notebooklm_bridge.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/metascaffold/notebooklm_bridge.py
"""NotebookLM Bridge — interface to notebooklm-py for knowledge-enriched reflection.

Provides graceful degradation: if NotebookLM is unavailable, returns empty
results and the system continues without external knowledge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BridgeResult:
    """Result from a NotebookLM operation."""
    success: bool
    content: str = ""
    reason: str = ""


def _get_client():
    """Lazy-import and return a NotebookLM client instance.

    This avoids import errors if notebooklm-py is not installed or not authenticated.
    """
    from notebooklm import NotebookLM
    return NotebookLM()


class NotebookLMBridge:
    """Bridge between MetaScaffold and NotebookLM via notebooklm-py."""

    def __init__(
        self,
        enabled: bool = True,
        default_notebook: str = "MetaScaffold_Core",
        fallback_on_error: bool = True,
    ):
        self.enabled = enabled
        self.default_notebook = default_notebook
        self.fallback_on_error = fallback_on_error

    def query_sync(self, question: str, notebook: str | None = None) -> BridgeResult:
        """Query a NotebookLM notebook synchronously.

        Args:
            question: The question to ask
            notebook: Notebook name (uses default if not specified)

        Returns:
            BridgeResult with success status and content
        """
        if not self.enabled:
            return BridgeResult(success=False, reason="NotebookLM bridge is disabled")

        target = notebook or self.default_notebook
        try:
            client = _get_client()
            # Find the notebook by name
            notebooks = client.list_notebooks()
            matching = [nb for nb in notebooks if nb.title == target]
            if not matching:
                return BridgeResult(
                    success=False,
                    reason=f"Notebook '{target}' not found",
                )
            response = client.chat(notebook_id=matching[0].id, message=question)
            return BridgeResult(success=True, content=response.text)

        except Exception as e:
            logger.warning("NotebookLM query failed: %s", e)
            if self.fallback_on_error:
                return BridgeResult(success=False, reason=str(e))
            raise

    def upload_source(self, url: str, notebook: str | None = None) -> BridgeResult:
        """Upload a URL source to a NotebookLM notebook."""
        if not self.enabled:
            return BridgeResult(success=False, reason="NotebookLM bridge is disabled")

        target = notebook or self.default_notebook
        try:
            client = _get_client()
            notebooks = client.list_notebooks()
            matching = [nb for nb in notebooks if nb.title == target]
            if not matching:
                return BridgeResult(success=False, reason=f"Notebook '{target}' not found")
            client.add_source(notebook_id=matching[0].id, url=url)
            return BridgeResult(success=True, content=f"Source uploaded: {url}")
        except Exception as e:
            logger.warning("NotebookLM upload failed: %s", e)
            if self.fallback_on_error:
                return BridgeResult(success=False, reason=str(e))
            raise

    def create_notebook(self, title: str) -> BridgeResult:
        """Create a new NotebookLM notebook."""
        if not self.enabled:
            return BridgeResult(success=False, reason="NotebookLM bridge is disabled")

        try:
            client = _get_client()
            nb = client.create_notebook(title=title)
            return BridgeResult(success=True, content=f"Notebook created: {nb.id}")
        except Exception as e:
            logger.warning("NotebookLM create failed: %s", e)
            if self.fallback_on_error:
                return BridgeResult(success=False, reason=str(e))
            raise
```

**Step 4: Run test to verify it passes**

Run: `cd C:/Code/MetaScaffold && uv run pytest tests/test_notebooklm_bridge.py -v`
Expected: 5 tests PASS

**Step 5: Commit**

```bash
git add src/metascaffold/notebooklm_bridge.py tests/test_notebooklm_bridge.py
git commit -m "feat: add NotebookLM bridge with graceful degradation"
```

---

### Task 5: JALON 1 — NotebookLM Authentication

**This is a manual user checkpoint. No code to write.**

**Step 1: Install notebooklm-py browser dependencies**

Run: `cd C:/Code/MetaScaffold && uv run pip install "notebooklm-py[browser]" && uv run playwright install chromium`

**Step 2: Present authentication command to user**

The user must run this in a separate terminal:

```bash
notebooklm login
```

This opens a Chromium browser window. The user signs into their Google account. Cookies are stored in `~/.notebooklm/storage_state.json`.

**Step 3: Verify authentication works**

Run: `cd C:/Code/MetaScaffold && uv run python -c "from notebooklm import NotebookLM; client = NotebookLM(); print('Notebooks:', [nb.title for nb in client.list_notebooks()])"`
Expected: Prints list of existing notebooks (may be empty)

**STOP HERE. Wait for user confirmation before proceeding to Phase 2.**

---

## Phase 2: Knowledge Base Ingestion

### Task 6: Research Sourcing Script

**Files:**
- Create: `scripts/source_research.py`

**Step 1: Create the research sourcing script**

```python
# scripts/source_research.py
"""Source research papers and repos for MetaScaffold knowledge base.

Downloads PDFs from ArXiv and collects GitHub README URLs
for upload to NotebookLM.
"""

import json
from pathlib import Path

# Curated list of high-impact papers and repos on LLM metacognition
SOURCES = {
    "papers": [
        {
            "title": "Reflexion: Language Agents with Verbal Reinforcement Learning",
            "url": "https://arxiv.org/abs/2303.11366",
            "topics": ["reflexion", "self-correction", "verbal reinforcement"],
        },
        {
            "title": "Self-Refine: Iterative Refinement with Self-Feedback",
            "url": "https://arxiv.org/abs/2303.17651",
            "topics": ["self-refinement", "iterative feedback"],
        },
        {
            "title": "Tree of Thoughts: Deliberate Problem Solving with LLMs",
            "url": "https://arxiv.org/abs/2305.10601",
            "topics": ["system2 thinking", "deliberation", "backtracking"],
        },
        {
            "title": "Chain-of-Thought Prompting Elicits Reasoning in LLMs",
            "url": "https://arxiv.org/abs/2201.11903",
            "topics": ["chain-of-thought", "reasoning"],
        },
        {
            "title": "Constitutional AI: Harmlessness from AI Feedback",
            "url": "https://arxiv.org/abs/2212.08073",
            "topics": ["self-critique", "auto-evaluation"],
        },
        {
            "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
            "url": "https://arxiv.org/abs/2210.03629",
            "topics": ["reasoning-action loop", "agentic systems"],
        },
        {
            "title": "Toolformer: Language Models Can Teach Themselves to Use Tools",
            "url": "https://arxiv.org/abs/2302.04761",
            "topics": ["tool use", "self-teaching"],
        },
        {
            "title": "Language Agent Tree Search (LATS)",
            "url": "https://arxiv.org/abs/2310.04406",
            "topics": ["tree search", "planning", "self-evaluation"],
        },
        {
            "title": "Cognitive Architectures for Language Agents (CoALA)",
            "url": "https://arxiv.org/abs/2309.02427",
            "topics": ["cognitive architecture", "memory", "metacognition"],
        },
        {
            "title": "Metacognitive Prompting Improves Understanding in LLMs",
            "url": "https://arxiv.org/abs/2308.05342",
            "topics": ["metacognitive prompting", "self-awareness"],
        },
        {
            "title": "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
            "url": "https://arxiv.org/abs/2310.11511",
            "topics": ["self-reflection", "RAG", "critique"],
        },
        {
            "title": "Think Before You Act: Unified Policy for Interleaving Language Reasoning with Actions",
            "url": "https://arxiv.org/abs/2304.11477",
            "topics": ["planning before action", "policy"],
        },
        {
            "title": "LLM Self-Correction Is Possible When Done Right",
            "url": "https://arxiv.org/abs/2406.01297",
            "topics": ["self-correction", "verification"],
        },
        {
            "title": "Confidence Estimation in LLMs for Reliable Decision-Making",
            "url": "https://arxiv.org/abs/2307.16040",
            "topics": ["confidence estimation", "uncertainty"],
        },
        {
            "title": "Dual Process Theory and LLMs: System 1 and System 2 Thinking",
            "url": "https://arxiv.org/abs/2407.06023",
            "topics": ["dual process", "system1", "system2"],
        },
    ],
    "github_repos": [
        {
            "name": "Reflexion",
            "url": "https://github.com/noahshinn/reflexion",
            "description": "Reference implementation of the Reflexion framework",
        },
        {
            "name": "Tree of Thoughts",
            "url": "https://github.com/princeton-nlp/tree-of-thought-llm",
            "description": "Official Tree of Thoughts implementation",
        },
        {
            "name": "Self-RAG",
            "url": "https://github.com/AkariAsai/self-rag",
            "description": "Self-Reflective RAG implementation",
        },
        {
            "name": "LATS",
            "url": "https://github.com/andyz245/LanguageAgentTreeSearch",
            "description": "Language Agent Tree Search implementation",
        },
        {
            "name": "LangGraph",
            "url": "https://github.com/langchain-ai/langgraph",
            "description": "Framework for building agentic workflows with cycles",
        },
    ],
}


def save_source_list(output_path: Path = Path("docs/research_sources.json")) -> None:
    """Save the curated source list as JSON for reference."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(SOURCES, f, indent=2)
    print(f"Source list saved to {output_path}")
    print(f"  Papers: {len(SOURCES['papers'])}")
    print(f"  GitHub repos: {len(SOURCES['github_repos'])}")


if __name__ == "__main__":
    save_source_list()
```

**Step 2: Run script to generate source list**

Run: `cd C:/Code/MetaScaffold && uv run python scripts/source_research.py`
Expected: `Source list saved to docs/research_sources.json` with 15 papers and 5 repos

**Step 3: Commit**

```bash
git add scripts/source_research.py docs/research_sources.json
git commit -m "feat: add curated research source list (15 papers, 5 repos)"
```

---

### Task 7: NotebookLM Ingestion Script

**Files:**
- Create: `scripts/ingest_to_notebooklm.py`

**Step 1: Create the ingestion script**

```python
# scripts/ingest_to_notebooklm.py
"""Ingest research sources into NotebookLM.

Creates the MetaScaffold_Core notebook and uploads all curated sources.
Requires prior authentication via `notebooklm login`.
"""

import json
import sys
import time
from pathlib import Path

from metascaffold.notebooklm_bridge import NotebookLMBridge


def main():
    sources_path = Path("docs/research_sources.json")
    if not sources_path.exists():
        print("ERROR: Run scripts/source_research.py first to generate source list.")
        sys.exit(1)

    with open(sources_path) as f:
        sources = json.load(f)

    bridge = NotebookLMBridge(enabled=True, default_notebook="MetaScaffold_Core", fallback_on_error=False)

    # Create notebook
    print("Creating notebook 'MetaScaffold_Core'...")
    result = bridge.create_notebook("MetaScaffold_Core")
    if result.success:
        print(f"  Created: {result.content}")
    else:
        print(f"  Note: {result.reason} (may already exist)")

    # Upload papers
    print(f"\nUploading {len(sources['papers'])} papers...")
    for i, paper in enumerate(sources["papers"], 1):
        print(f"  [{i}/{len(sources['papers'])}] {paper['title']}")
        result = bridge.upload_source(paper["url"])
        if result.success:
            print(f"    OK")
        else:
            print(f"    WARN: {result.reason}")
        time.sleep(2)  # Rate limiting

    # Upload GitHub repos
    print(f"\nUploading {len(sources['github_repos'])} GitHub repos...")
    for i, repo in enumerate(sources["github_repos"], 1):
        print(f"  [{i}/{len(sources['github_repos'])}] {repo['name']}")
        result = bridge.upload_source(repo["url"])
        if result.success:
            print(f"    OK")
        else:
            print(f"    WARN: {result.reason}")
        time.sleep(2)

    print("\nIngestion complete. Verify in NotebookLM web UI.")


if __name__ == "__main__":
    main()
```

**Step 2: Run ingestion (requires authenticated NotebookLM)**

Run: `cd C:/Code/MetaScaffold && uv run python scripts/ingest_to_notebooklm.py`
Expected: Each source uploads with "OK" status

**Step 3: Commit**

```bash
git add scripts/ingest_to_notebooklm.py
git commit -m "feat: add NotebookLM ingestion script for research sources"
```

**JALON 2: Confirm sources are ingested. List key concepts identified. Wait for user feu vert.**

---

## Phase 3: Core Components

### Task 8: Classifier Module

**Files:**
- Create: `tests/test_classifier.py`
- Create: `src/metascaffold/classifier.py`

**Step 1: Write the failing test**

```python
# tests/test_classifier.py
"""Tests for the System 1/2 classifier."""

from metascaffold.classifier import Classifier, ClassificationResult


class TestClassifier:
    def test_simple_read_is_system1(self):
        """Simple read operations should route to System 1."""
        c = Classifier(system2_threshold=0.8, always_system2_tools=[])
        result = c.classify(
            tool_name="Read",
            tool_input={"file_path": "/tmp/test.py"},
            context="Read a file to understand the code",
        )
        assert result.routing == "system1"
        assert result.confidence >= 0.8

    def test_multi_file_edit_is_system2(self):
        """Complex multi-file edits should route to System 2."""
        c = Classifier(system2_threshold=0.8, always_system2_tools=[])
        result = c.classify(
            tool_name="Edit",
            tool_input={"file_path": "/src/auth.py"},
            context="Refactor the entire authentication system across 5 modules",
        )
        assert result.routing == "system2"
        assert result.confidence < 0.8

    def test_always_system2_tools(self):
        """Tools in always_system2_tools should always route to System 2."""
        c = Classifier(system2_threshold=0.8, always_system2_tools=["Write"])
        result = c.classify(
            tool_name="Write",
            tool_input={"file_path": "/tmp/new_file.py"},
            context="Create a small utility function",
        )
        assert result.routing == "system2"

    def test_destructive_bash_is_system2(self):
        """Destructive bash commands should route to System 2."""
        c = Classifier(system2_threshold=0.8, always_system2_tools=[])
        result = c.classify(
            tool_name="Bash",
            tool_input={"command": "rm -rf /tmp/project"},
            context="Clean up temporary files",
        )
        assert result.routing == "system2"
        assert result.confidence < 0.8

    def test_classification_returns_all_fields(self):
        """ClassificationResult should have all required fields."""
        c = Classifier(system2_threshold=0.8, always_system2_tools=[])
        result = c.classify(
            tool_name="Bash",
            tool_input={"command": "ls"},
            context="List files",
        )
        assert isinstance(result, ClassificationResult)
        assert result.routing in ("system1", "system2")
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.reasoning, str)
        assert isinstance(result.signals, dict)

    def test_historical_success_rate_lowers_confidence(self):
        """Low historical success rate should lower confidence."""
        c = Classifier(system2_threshold=0.8, always_system2_tools=[])
        result = c.classify(
            tool_name="Edit",
            tool_input={"file_path": "/src/complex.py"},
            context="Fix the flaky test",
            historical_success_rate=0.3,
        )
        # Low historical success → lower confidence → System 2
        assert result.routing == "system2"
```

**Step 2: Run test to verify it fails**

Run: `cd C:/Code/MetaScaffold && uv run pytest tests/test_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/metascaffold/classifier.py
"""System 1/2 Classifier — routes tasks to fast or deliberate processing.

Uses heuristic signals (complexity, reversibility, uncertainty, history)
to determine whether a task needs deep reflection (System 2) or can
proceed directly (System 1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Heuristic patterns for complexity detection
_COMPLEX_KEYWORDS = re.compile(
    r"refactor|architect|redesign|migrate|across\s+\d+|entire|all\s+modules|system-wide",
    re.IGNORECASE,
)

_DESTRUCTIVE_COMMANDS = re.compile(
    r"rm\s+-rf|drop\s+table|delete|git\s+reset\s+--hard|git\s+push\s+--force|truncate|format",
    re.IGNORECASE,
)

_SIMPLE_COMMANDS = re.compile(
    r"^(ls|pwd|echo|cat|head|tail|wc|date|whoami|which|type|git\s+status|git\s+log|git\s+diff)(\s|$)",
    re.IGNORECASE,
)


@dataclass
class ClassificationResult:
    """Result of the System 1/2 classification."""
    routing: str  # "system1" or "system2"
    confidence: float  # 0.0 - 1.0
    reasoning: str
    signals: dict = field(default_factory=dict)


class Classifier:
    """Heuristic classifier for System 1/2 routing."""

    def __init__(
        self,
        system2_threshold: float = 0.8,
        always_system2_tools: list[str] | None = None,
    ):
        self.system2_threshold = system2_threshold
        self.always_system2_tools = always_system2_tools or []

    def classify(
        self,
        tool_name: str,
        tool_input: dict,
        context: str,
        historical_success_rate: float | None = None,
    ) -> ClassificationResult:
        """Classify a tool call as System 1 or System 2.

        Args:
            tool_name: Name of the tool being called (Bash, Edit, Write, etc.)
            tool_input: The tool's input parameters
            context: Natural language context/description of what's being done
            historical_success_rate: Optional success rate from telemetry (0.0-1.0)

        Returns:
            ClassificationResult with routing decision and confidence
        """
        signals = {
            "complexity": "low",
            "reversibility": "high",
            "uncertainty": "low",
            "historical_success_rate": historical_success_rate,
        }
        confidence = 0.9  # Start optimistic
        reasons: list[str] = []

        # Force System 2 for configured tools
        if tool_name in self.always_system2_tools:
            return ClassificationResult(
                routing="system2",
                confidence=0.5,
                reasoning=f"Tool '{tool_name}' is configured for mandatory System 2",
                signals=signals,
            )

        # Read-only tools are always System 1
        if tool_name in ("Read", "Grep", "Glob", "WebSearch", "WebFetch"):
            return ClassificationResult(
                routing="system1",
                confidence=0.95,
                reasoning=f"Read-only tool '{tool_name}'",
                signals=signals,
            )

        # Check for destructive commands
        command = tool_input.get("command", "")
        if _DESTRUCTIVE_COMMANDS.search(command):
            confidence -= 0.35
            signals["reversibility"] = "low"
            reasons.append("Destructive command detected")

        # Check for simple commands
        if _SIMPLE_COMMANDS.match(command):
            confidence += 0.05
            reasons.append("Simple read-only command")

        # Check complexity from context
        if _COMPLEX_KEYWORDS.search(context):
            confidence -= 0.25
            signals["complexity"] = "high"
            reasons.append("Complex task keywords detected")

        # Factor in historical success rate
        if historical_success_rate is not None:
            if historical_success_rate < 0.5:
                confidence -= 0.2
                signals["uncertainty"] = "high"
                reasons.append(f"Low historical success rate ({historical_success_rate:.0%})")
            elif historical_success_rate < 0.7:
                confidence -= 0.1
                reasons.append(f"Moderate historical success rate ({historical_success_rate:.0%})")

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        routing = "system1" if confidence >= self.system2_threshold else "system2"
        reasoning = "; ".join(reasons) if reasons else f"Standard {tool_name} operation"

        return ClassificationResult(
            routing=routing,
            confidence=confidence,
            reasoning=reasoning,
            signals=signals,
        )
```

**Step 4: Run test to verify it passes**

Run: `cd C:/Code/MetaScaffold && uv run pytest tests/test_classifier.py -v`
Expected: 6 tests PASS

**Step 5: Commit**

```bash
git add src/metascaffold/classifier.py tests/test_classifier.py
git commit -m "feat: add System 1/2 classifier with heuristic routing"
```

---

### Task 9: Planner Module

**Files:**
- Create: `tests/test_planner.py`
- Create: `src/metascaffold/planner.py`

**Step 1: Write the failing test**

```python
# tests/test_planner.py
"""Tests for the planner (decomposition & strategy) module."""

from metascaffold.planner import Planner, Plan, Strategy


class TestPlanner:
    def test_plan_has_at_least_one_strategy(self):
        """Every plan should propose at least one strategy."""
        planner = Planner()
        plan = planner.create_plan(
            task="Add input validation to the login form",
            context="Simple single-file change in auth.py",
        )
        assert isinstance(plan, Plan)
        assert len(plan.strategies) >= 1

    def test_plan_has_recommended_strategy(self):
        """Plan should indicate which strategy is recommended."""
        planner = Planner()
        plan = planner.create_plan(
            task="Refactor database module",
            context="Move from raw SQL to ORM pattern across 3 files",
        )
        assert plan.recommended in [s.id for s in plan.strategies]

    def test_each_strategy_has_steps(self):
        """Each strategy should have concrete steps."""
        planner = Planner()
        plan = planner.create_plan(
            task="Add caching layer",
            context="Implement in-memory caching for API responses",
        )
        for strategy in plan.strategies:
            assert isinstance(strategy, Strategy)
            assert len(strategy.steps) >= 1
            assert strategy.confidence > 0.0
            assert isinstance(strategy.risks, list)
            assert isinstance(strategy.rollback_plan, str)

    def test_plan_includes_task_description(self):
        """Plan should echo back the original task."""
        planner = Planner()
        plan = planner.create_plan(task="Fix bug in parser", context="Off-by-one error")
        assert plan.task == "Fix bug in parser"

    def test_plan_serializes_to_dict(self):
        """Plan should be serializable to a dict for MCP transport."""
        planner = Planner()
        plan = planner.create_plan(task="Test task", context="Test context")
        d = plan.to_dict()
        assert isinstance(d, dict)
        assert "task" in d
        assert "strategies" in d
        assert "recommended" in d
```

**Step 2: Run test to verify it fails**

Run: `cd C:/Code/MetaScaffold && uv run pytest tests/test_planner.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/metascaffold/planner.py
"""Planner — decomposes System 2 tasks into strategies with steps, risks, and rollback plans.

The Planner produces structured plans that the Evaluator can later assess.
It uses heuristic decomposition (not LLM calls) to suggest strategies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Strategy:
    """A single execution strategy with steps, risks, and rollback."""
    id: str
    description: str
    steps: list[str]
    confidence: float
    risks: list[str]
    rollback_plan: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "steps": self.steps,
            "confidence": self.confidence,
            "risks": self.risks,
            "rollback_plan": self.rollback_plan,
        }


@dataclass
class Plan:
    """A structured execution plan with multiple strategies."""
    task: str
    strategies: list[Strategy]
    recommended: str
    notebooklm_insights: str = ""

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "strategies": [s.to_dict() for s in self.strategies],
            "recommended": self.recommended,
            "notebooklm_insights": self.notebooklm_insights,
        }


_REFACTOR_PATTERN = re.compile(r"refactor|restructure|reorganize|rewrite", re.IGNORECASE)
_BUG_PATTERN = re.compile(r"fix|bug|error|issue|broken|crash", re.IGNORECASE)
_FEATURE_PATTERN = re.compile(r"add|create|implement|build|new", re.IGNORECASE)


class Planner:
    """Heuristic planner that decomposes tasks into strategies."""

    def create_plan(
        self,
        task: str,
        context: str,
        notebooklm_insights: str = "",
    ) -> Plan:
        """Create a structured plan for a System 2 task.

        Args:
            task: Description of the task
            context: Additional context (files involved, scope, etc.)
            notebooklm_insights: Optional insights from NotebookLM

        Returns:
            Plan with 1-3 strategies
        """
        strategies: list[Strategy] = []

        if _REFACTOR_PATTERN.search(task):
            strategies = self._plan_refactor(task, context)
        elif _BUG_PATTERN.search(task):
            strategies = self._plan_bugfix(task, context)
        elif _FEATURE_PATTERN.search(task):
            strategies = self._plan_feature(task, context)
        else:
            strategies = self._plan_generic(task, context)

        recommended = strategies[0].id if strategies else "A"

        return Plan(
            task=task,
            strategies=strategies,
            recommended=recommended,
            notebooklm_insights=notebooklm_insights,
        )

    def _plan_refactor(self, task: str, context: str) -> list[Strategy]:
        return [
            Strategy(
                id="A",
                description="Incremental refactor with adapter pattern",
                steps=[
                    "Identify all call sites for the target code",
                    "Create adapter interface that wraps the current implementation",
                    "Implement new version behind the adapter",
                    "Migrate call sites one at a time to the new interface",
                    "Remove adapter and old implementation",
                    "Run full test suite",
                ],
                confidence=0.8,
                risks=["Adapter adds temporary complexity", "Partial migration state"],
                rollback_plan="Revert adapter changes, restore original implementation",
            ),
            Strategy(
                id="B",
                description="Direct replacement in a feature branch",
                steps=[
                    "Create feature branch",
                    "Rewrite the target module from scratch",
                    "Update all call sites",
                    "Run full test suite",
                    "Merge or discard branch",
                ],
                confidence=0.6,
                risks=["All-or-nothing: partial completion is not usable", "High blast radius"],
                rollback_plan="Delete feature branch entirely",
            ),
        ]

    def _plan_bugfix(self, task: str, context: str) -> list[Strategy]:
        return [
            Strategy(
                id="A",
                description="Reproduce, isolate, fix with regression test",
                steps=[
                    "Write a failing test that reproduces the bug",
                    "Run test to confirm it fails",
                    "Identify root cause via debugger or logging",
                    "Apply minimal fix",
                    "Run test to confirm it passes",
                    "Run full test suite for regressions",
                ],
                confidence=0.85,
                risks=["Root cause may be deeper than symptoms"],
                rollback_plan="Revert the fix commit",
            ),
            Strategy(
                id="B",
                description="Defensive fix with extra validation",
                steps=[
                    "Add input validation at the entry point",
                    "Add error handling around the failing code path",
                    "Write tests for both valid and invalid inputs",
                    "Run full test suite",
                ],
                confidence=0.7,
                risks=["May mask the real bug instead of fixing it"],
                rollback_plan="Revert validation changes",
            ),
        ]

    def _plan_feature(self, task: str, context: str) -> list[Strategy]:
        return [
            Strategy(
                id="A",
                description="TDD: test-first implementation",
                steps=[
                    "Define the public API / interface for the feature",
                    "Write failing tests for the happy path",
                    "Implement minimal code to pass tests",
                    "Write edge case tests",
                    "Implement edge case handling",
                    "Integration test",
                ],
                confidence=0.8,
                risks=["API design may need revision after implementation"],
                rollback_plan="Revert feature branch",
            ),
        ]

    def _plan_generic(self, task: str, context: str) -> list[Strategy]:
        return [
            Strategy(
                id="A",
                description="Standard step-by-step approach",
                steps=[
                    "Analyze the current state and requirements",
                    "Identify files to modify",
                    "Make changes incrementally with tests",
                    "Verify all tests pass",
                ],
                confidence=0.75,
                risks=["Requirements may be unclear"],
                rollback_plan="Revert all changes via git",
            ),
        ]
```

**Step 4: Run test to verify it passes**

Run: `cd C:/Code/MetaScaffold && uv run pytest tests/test_planner.py -v`
Expected: 5 tests PASS

**Step 5: Commit**

```bash
git add src/metascaffold/planner.py tests/test_planner.py
git commit -m "feat: add planner module with heuristic strategy decomposition"
```

---

### Task 10: Sandbox Module

**Files:**
- Create: `tests/test_sandbox.py`
- Create: `src/metascaffold/sandbox.py`

**Step 1: Write the failing test**

```python
# tests/test_sandbox.py
"""Tests for the sandbox (worktree + subprocess isolation) module."""

import os
import sys

import pytest

from metascaffold.sandbox import Sandbox, SandboxResult


class TestSandbox:
    def test_execute_simple_command(self, tmp_path):
        """Should execute a simple command and capture output."""
        sandbox = Sandbox(work_dir=str(tmp_path))
        result = sandbox.execute("echo hello world")
        assert isinstance(result, SandboxResult)
        assert result.exit_code == 0
        assert "hello world" in result.stdout

    def test_capture_stderr(self, tmp_path):
        """Should capture stderr output."""
        sandbox = Sandbox(work_dir=str(tmp_path))
        result = sandbox.execute(f"{sys.executable} -c \"import sys; sys.stderr.write('err msg')\"")
        assert "err msg" in result.stderr

    def test_timeout_kills_process(self, tmp_path):
        """Commands exceeding timeout should be killed."""
        sandbox = Sandbox(work_dir=str(tmp_path), default_timeout_seconds=2)
        result = sandbox.execute(f"{sys.executable} -c \"import time; time.sleep(30)\"")
        assert result.exit_code != 0
        assert result.timed_out is True

    def test_failed_command_returns_nonzero(self, tmp_path):
        """Failed commands should return non-zero exit code."""
        sandbox = Sandbox(work_dir=str(tmp_path))
        result = sandbox.execute(f"{sys.executable} -c \"raise ValueError('boom')\"")
        assert result.exit_code != 0
        assert "boom" in result.stderr

    def test_result_includes_duration(self, tmp_path):
        """Result should include execution duration in ms."""
        sandbox = Sandbox(work_dir=str(tmp_path))
        result = sandbox.execute("echo fast")
        assert result.duration_ms >= 0

    def test_result_serializes_to_dict(self, tmp_path):
        """SandboxResult should be serializable to dict."""
        sandbox = Sandbox(work_dir=str(tmp_path))
        result = sandbox.execute("echo test")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "exit_code" in d
        assert "stdout" in d
        assert "stderr" in d
        assert "duration_ms" in d
```

**Step 2: Run test to verify it fails**

Run: `cd C:/Code/MetaScaffold && uv run pytest tests/test_sandbox.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/metascaffold/sandbox.py
"""Sandbox — isolated execution via restricted subprocesses.

Provides timeout, stderr/stdout capture, and duration tracking.
Git worktree integration is handled at the MCP server level.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass


@dataclass
class SandboxResult:
    """Result of a sandboxed command execution."""
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False
    worktree_path: str | None = None
    worktree_branch: str | None = None

    def to_dict(self) -> dict:
        d = {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
        }
        if self.worktree_path:
            d["worktree_path"] = self.worktree_path
            d["worktree_branch"] = self.worktree_branch
        return d


class Sandbox:
    """Restricted subprocess executor with timeout and output capture."""

    def __init__(
        self,
        work_dir: str = ".",
        default_timeout_seconds: int = 30,
    ):
        self.work_dir = work_dir
        self.default_timeout_seconds = default_timeout_seconds

    def execute(
        self,
        command: str,
        timeout_seconds: int | None = None,
    ) -> SandboxResult:
        """Execute a command in a restricted subprocess.

        Args:
            command: Shell command to execute
            timeout_seconds: Override default timeout (None = use default)

        Returns:
            SandboxResult with exit code, stdout, stderr, and timing
        """
        timeout = timeout_seconds or self.default_timeout_seconds
        start = time.monotonic()
        timed_out = False

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.work_dir,
            )
            exit_code = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = -1
            stdout = ""
            stderr = f"Command timed out after {timeout}s"

        duration_ms = int((time.monotonic() - start) * 1000)

        return SandboxResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            timed_out=timed_out,
        )
```

**Step 4: Run test to verify it passes**

Run: `cd C:/Code/MetaScaffold && uv run pytest tests/test_sandbox.py -v`
Expected: 6 tests PASS

**Step 5: Commit**

```bash
git add src/metascaffold/sandbox.py tests/test_sandbox.py
git commit -m "feat: add sandbox module with subprocess isolation and timeout"
```

---

### Task 11: Evaluator Module

**Files:**
- Create: `tests/test_evaluator.py`
- Create: `src/metascaffold/evaluator.py`

**Step 1: Write the failing test**

```python
# tests/test_evaluator.py
"""Tests for the evaluator (auto-critique & correction) module."""

from metascaffold.evaluator import Evaluator, EvaluationResult
from metascaffold.sandbox import SandboxResult


class TestEvaluator:
    def test_pass_on_zero_exit_code(self):
        """Exit code 0 with clean output should evaluate as 'pass'."""
        evaluator = Evaluator(max_retry_attempts=3)
        result = evaluator.evaluate(
            sandbox_result=SandboxResult(
                exit_code=0, stdout="All 12 tests passed", stderr="", duration_ms=500
            ),
            attempt=1,
        )
        assert isinstance(result, EvaluationResult)
        assert result.verdict == "pass"
        assert result.confidence >= 0.8

    def test_retry_on_test_failure(self):
        """Test failures should trigger 'retry' verdict."""
        evaluator = Evaluator(max_retry_attempts=3)
        result = evaluator.evaluate(
            sandbox_result=SandboxResult(
                exit_code=1,
                stdout="3 passed, 2 failed",
                stderr="FAILED test_auth - AssertionError",
                duration_ms=800,
            ),
            attempt=1,
        )
        assert result.verdict == "retry"
        assert len(result.issues) > 0

    def test_escalate_after_max_attempts(self):
        """Exceeding max attempts should trigger 'escalate'."""
        evaluator = Evaluator(max_retry_attempts=3)
        result = evaluator.evaluate(
            sandbox_result=SandboxResult(
                exit_code=1, stdout="", stderr="Error", duration_ms=100
            ),
            attempt=3,
        )
        assert result.verdict == "escalate"

    def test_backtrack_on_severe_error(self):
        """Severe errors (crash, import error) should trigger 'backtrack'."""
        evaluator = Evaluator(max_retry_attempts=3)
        result = evaluator.evaluate(
            sandbox_result=SandboxResult(
                exit_code=1,
                stdout="",
                stderr="ModuleNotFoundError: No module named 'nonexistent'",
                duration_ms=100,
            ),
            attempt=1,
        )
        assert result.verdict == "backtrack"

    def test_timeout_is_retry(self):
        """Timed out commands should trigger retry (may need more time)."""
        evaluator = Evaluator(max_retry_attempts=3)
        result = evaluator.evaluate(
            sandbox_result=SandboxResult(
                exit_code=-1, stdout="", stderr="Timed out", duration_ms=30000, timed_out=True
            ),
            attempt=1,
        )
        assert result.verdict == "retry"

    def test_result_serializes_to_dict(self):
        """EvaluationResult should serialize to dict for MCP transport."""
        evaluator = Evaluator(max_retry_attempts=3)
        result = evaluator.evaluate(
            sandbox_result=SandboxResult(exit_code=0, stdout="ok", stderr="", duration_ms=100),
            attempt=1,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "verdict" in d
        assert "confidence" in d
        assert "issues" in d
```

**Step 2: Run test to verify it fails**

Run: `cd C:/Code/MetaScaffold && uv run pytest tests/test_evaluator.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/metascaffold/evaluator.py
"""Evaluator — auto-critique and correction engine.

Analyzes sandbox execution results and produces a verdict:
- pass: result is acceptable
- retry: try again with corrections (same strategy)
- backtrack: change strategy (return to Planner)
- escalate: request human intervention
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from metascaffold.sandbox import SandboxResult


_SEVERE_ERRORS = re.compile(
    r"ModuleNotFoundError|ImportError|SyntaxError|IndentationError|RecursionError|MemoryError|PermissionError",
    re.IGNORECASE,
)

_TEST_FAILURE = re.compile(
    r"FAIL|failed|error|assert|AssertionError|AssertionError",
    re.IGNORECASE,
)


@dataclass
class Issue:
    """A single issue found during evaluation."""
    type: str  # "test_failure", "crash", "timeout", "severe_error"
    detail: str
    severity: str  # "low", "medium", "high", "critical"

    def to_dict(self) -> dict:
        return {"type": self.type, "detail": self.detail, "severity": self.severity}


@dataclass
class EvaluationResult:
    """Result of the auto-evaluation."""
    verdict: str  # "pass", "retry", "backtrack", "escalate"
    confidence: float
    issues: list[Issue] = field(default_factory=list)
    corrections: list[dict] = field(default_factory=list)
    attempt: int = 1
    max_attempts: int = 3

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "issues": [i.to_dict() for i in self.issues],
            "corrections": self.corrections,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
        }


class Evaluator:
    """Evaluates sandbox results and produces verdicts."""

    def __init__(self, max_retry_attempts: int = 3):
        self.max_retry_attempts = max_retry_attempts

    def evaluate(
        self,
        sandbox_result: SandboxResult,
        attempt: int = 1,
    ) -> EvaluationResult:
        """Evaluate a sandbox execution result.

        Args:
            sandbox_result: The result from sandbox execution
            attempt: Current attempt number (1-based)

        Returns:
            EvaluationResult with verdict and details
        """
        issues: list[Issue] = []
        combined_output = sandbox_result.stdout + "\n" + sandbox_result.stderr

        # Check for timeout
        if sandbox_result.timed_out:
            issues.append(Issue(
                type="timeout",
                detail=f"Command timed out after {sandbox_result.duration_ms}ms",
                severity="medium",
            ))

        # Check for severe errors (backtrack-worthy)
        if _SEVERE_ERRORS.search(sandbox_result.stderr):
            issues.append(Issue(
                type="severe_error",
                detail=sandbox_result.stderr.strip()[:200],
                severity="critical",
            ))

        # Check for test failures
        if sandbox_result.exit_code != 0 and _TEST_FAILURE.search(combined_output):
            issues.append(Issue(
                type="test_failure",
                detail=sandbox_result.stderr.strip()[:200],
                severity="medium",
            ))

        # Determine verdict
        if sandbox_result.exit_code == 0 and not issues:
            return EvaluationResult(
                verdict="pass",
                confidence=0.9,
                issues=issues,
                attempt=attempt,
                max_attempts=self.max_retry_attempts,
            )

        # Check for severe errors → backtrack
        if any(i.severity == "critical" for i in issues):
            return EvaluationResult(
                verdict="backtrack",
                confidence=0.3,
                issues=issues,
                attempt=attempt,
                max_attempts=self.max_retry_attempts,
            )

        # Check if max attempts exceeded → escalate
        if attempt >= self.max_retry_attempts:
            return EvaluationResult(
                verdict="escalate",
                confidence=0.2,
                issues=issues,
                attempt=attempt,
                max_attempts=self.max_retry_attempts,
            )

        # Default: retry
        return EvaluationResult(
            verdict="retry",
            confidence=0.5,
            issues=issues,
            attempt=attempt,
            max_attempts=self.max_retry_attempts,
        )
```

**Step 4: Run test to verify it passes**

Run: `cd C:/Code/MetaScaffold && uv run pytest tests/test_evaluator.py -v`
Expected: 6 tests PASS

**Step 5: Commit**

```bash
git add src/metascaffold/evaluator.py tests/test_evaluator.py
git commit -m "feat: add evaluator module with auto-critique and verdict engine"
```

---

## Phase 4: Integration

### Task 12: MCP Server

**Files:**
- Create: `tests/test_server.py`
- Create: `src/metascaffold/server.py`
- Create: `.mcp.json`

**Step 1: Write the failing test**

```python
# tests/test_server.py
"""Tests for the MCP server tool registration.

Verifies that all MetaScaffold tools are correctly registered on the FastMCP instance.
"""

from metascaffold.server import mcp


class TestMCPServerRegistration:
    def test_server_has_classify_tool(self):
        """metascaffold_classify tool should be registered."""
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "metascaffold_classify" in tool_names

    def test_server_has_plan_tool(self):
        """metascaffold_plan tool should be registered."""
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "metascaffold_plan" in tool_names

    def test_server_has_sandbox_exec_tool(self):
        """metascaffold_sandbox_exec tool should be registered."""
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "metascaffold_sandbox_exec" in tool_names

    def test_server_has_evaluate_tool(self):
        """metascaffold_evaluate tool should be registered."""
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "metascaffold_evaluate" in tool_names

    def test_server_has_nlm_query_tool(self):
        """metascaffold_nlm_query tool should be registered."""
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "metascaffold_nlm_query" in tool_names

    def test_server_has_telemetry_query_tool(self):
        """metascaffold_telemetry_query tool should be registered."""
        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "metascaffold_telemetry_query" in tool_names
```

**Step 2: Run test to verify it fails**

Run: `cd C:/Code/MetaScaffold && uv run pytest tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/metascaffold/server.py
"""MetaScaffold MCP Server — cognitive middleware for Claude Code.

Exposes 6 tools: classify, plan, sandbox_exec, evaluate, nlm_query, telemetry_query.
Run with: python src/metascaffold/server.py
Register with: claude mcp add metascaffold -- python src/metascaffold/server.py
"""

from __future__ import annotations

import logging
import sys
from typing import Annotated

from pydantic import Field
from mcp.server.fastmcp import FastMCP

from metascaffold.classifier import Classifier
from metascaffold.config import load_config
from metascaffold.evaluator import Evaluator
from metascaffold.notebooklm_bridge import NotebookLMBridge
from metascaffold.planner import Planner
from metascaffold.sandbox import Sandbox
from metascaffold.telemetry import CognitiveEvent, TelemetryLogger

# Configure logging to stderr (NEVER print to stdout in stdio MCP servers)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("metascaffold")

# Load configuration
config = load_config()

# Initialize components
classifier = Classifier(
    system2_threshold=config.classifier.system2_threshold,
    always_system2_tools=config.classifier.always_system2_tools,
)
planner = Planner()
sandbox = Sandbox(default_timeout_seconds=config.sandbox.default_timeout_seconds)
evaluator = Evaluator(max_retry_attempts=config.sandbox.max_retry_attempts)
telemetry = TelemetryLogger(
    json_dir=config.telemetry.json_dir,
    sqlite_path=config.telemetry.sqlite_path,
)
nlm_bridge = NotebookLMBridge(
    enabled=config.notebooklm.enabled,
    default_notebook=config.notebooklm.default_notebook,
    fallback_on_error=config.notebooklm.fallback_on_error,
)

# Create MCP server
mcp = FastMCP("metascaffold")


@mcp.tool()
def metascaffold_classify(
    tool_name: Annotated[str, Field(description="Name of the tool being called (Bash, Edit, Write, etc.)")],
    tool_input: Annotated[str, Field(description="JSON string of the tool's input parameters")],
    context: Annotated[str, Field(description="Natural language description of what is being done")],
) -> dict:
    """Classify a task as System 1 (fast) or System 2 (deliberate).

    Analyzes complexity, reversibility, uncertainty, and historical success rate
    to determine whether the task needs deep reflection before execution.
    """
    import json as _json
    parsed_input = _json.loads(tool_input) if isinstance(tool_input, str) else tool_input

    # Check historical success rate from telemetry
    historical_rate = telemetry.get_success_rate(context[:50])

    result = classifier.classify(
        tool_name=tool_name,
        tool_input=parsed_input,
        context=context,
        historical_success_rate=historical_rate,
    )

    # Log classification event
    telemetry.log(CognitiveEvent(
        event_type="classification",
        data={
            "routing": result.routing,
            "confidence": result.confidence,
            "tool_name": tool_name,
            "reasoning": result.reasoning,
        },
    ))
    telemetry.flush()

    return {
        "routing": result.routing,
        "confidence": result.confidence,
        "reasoning": result.reasoning,
        "signals": result.signals,
    }


@mcp.tool()
def metascaffold_plan(
    task: Annotated[str, Field(description="Description of the task to plan")],
    context: Annotated[str, Field(description="Additional context: files involved, scope, constraints")],
) -> dict:
    """Create a structured execution plan for a System 2 task.

    Decomposes the task into strategies with steps, risks, confidence scores,
    and rollback plans. Optionally consults NotebookLM for domain knowledge.
    """
    # Optionally consult NotebookLM
    nlm_insights = ""
    if nlm_bridge.enabled:
        nlm_result = nlm_bridge.query_sync(
            f"What are best practices for: {task}? Consider: {context}"
        )
        if nlm_result.success:
            nlm_insights = nlm_result.content

    plan = planner.create_plan(
        task=task,
        context=context,
        notebooklm_insights=nlm_insights,
    )

    # Log plan creation
    telemetry.log(CognitiveEvent(
        event_type="plan_created",
        data={
            "task": task,
            "num_strategies": len(plan.strategies),
            "recommended": plan.recommended,
        },
    ))
    telemetry.flush()

    return plan.to_dict()


@mcp.tool()
def metascaffold_sandbox_exec(
    command: Annotated[str, Field(description="Shell command to execute in the sandbox")],
    timeout_seconds: Annotated[int, Field(description="Timeout in seconds (default: 30)")] = 30,
) -> dict:
    """Execute a command in a sandboxed subprocess with timeout and output capture.

    Provides isolation through restricted subprocess execution.
    Captures stdout, stderr, exit code, and execution duration.
    """
    result = sandbox.execute(command=command, timeout_seconds=timeout_seconds)

    # Log execution
    telemetry.log(CognitiveEvent(
        event_type="execution_result",
        data={
            "command": command[:100],
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "duration_ms": result.duration_ms,
        },
    ))
    telemetry.flush()

    return result.to_dict()


@mcp.tool()
def metascaffold_evaluate(
    exit_code: Annotated[int, Field(description="Exit code from the executed command")],
    stdout: Annotated[str, Field(description="Standard output from the command")],
    stderr: Annotated[str, Field(description="Standard error from the command")],
    duration_ms: Annotated[int, Field(description="Execution duration in milliseconds")],
    attempt: Annotated[int, Field(description="Current attempt number (1-based)")] = 1,
    timed_out: Annotated[bool, Field(description="Whether the command timed out")] = False,
) -> dict:
    """Evaluate the result of a sandboxed execution.

    Produces a verdict: pass, retry, backtrack, or escalate.
    Detects test failures, severe errors, and timeout conditions.
    """
    from metascaffold.sandbox import SandboxResult

    sandbox_result = SandboxResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=duration_ms,
        timed_out=timed_out,
    )
    result = evaluator.evaluate(sandbox_result=sandbox_result, attempt=attempt)

    # Log evaluation
    telemetry.log(CognitiveEvent(
        event_type="evaluation",
        data={
            "verdict": result.verdict,
            "confidence": result.confidence,
            "attempt": result.attempt,
            "num_issues": len(result.issues),
        },
    ))
    telemetry.flush()

    return result.to_dict()


@mcp.tool()
def metascaffold_nlm_query(
    question: Annotated[str, Field(description="Question to ask the NotebookLM knowledge base")],
    notebook: Annotated[str, Field(description="Notebook name (uses default if empty)")] = "",
) -> dict:
    """Query the NotebookLM knowledge base for domain-specific insights.

    Returns sourced answers from the MetaScaffold research corpus.
    Degrades gracefully if NotebookLM is unavailable.
    """
    result = nlm_bridge.query_sync(
        question=question,
        notebook=notebook or None,
    )
    return {"success": result.success, "content": result.content, "reason": result.reason}


@mcp.tool()
def metascaffold_telemetry_query(
    task_type: Annotated[str, Field(description="Task type to query success rate for")],
) -> dict:
    """Query cognitive telemetry for historical success rates.

    Returns the success rate for a given task type based on past evaluations.
    Used by the Classifier to improve routing accuracy over time.
    """
    rate = telemetry.get_success_rate(task_type)
    return {
        "task_type": task_type,
        "success_rate": rate,
        "has_data": rate is not None,
    }


if __name__ == "__main__":
    logger.info("Starting MetaScaffold MCP Server...")
    mcp.run(transport="stdio")
```

**Step 4: Create .mcp.json for Claude Code integration**

```json
{
  "mcpServers": {
    "metascaffold": {
      "type": "stdio",
      "command": "uv",
      "args": [
        "--directory",
        ".",
        "run",
        "python",
        "src/metascaffold/server.py"
      ],
      "env": {}
    }
  }
}
```

**Step 5: Run test to verify it passes**

Run: `cd C:/Code/MetaScaffold && uv run pytest tests/test_server.py -v`
Expected: 6 tests PASS

**Step 6: Commit**

```bash
git add src/metascaffold/server.py .mcp.json tests/test_server.py
git commit -m "feat: add MCP server with all 6 cognitive tools"
```

---

### Task 13: Claude Code Hooks

**Files:**
- Create: `hooks/pre_tool_gate.py`
- Create: `hooks/post_tool_evaluate.py`
- Create: `tests/test_hooks.py`

**Step 1: Write the failing test**

```python
# tests/test_hooks.py
"""Tests for the Claude Code hook scripts.

Tests the hook logic in isolation (without requiring a running MCP server).
"""

import json
import sys
from unittest.mock import patch

from hooks.pre_tool_gate import should_intercept, format_system2_message
from hooks.post_tool_evaluate import parse_tool_result


class TestPreToolGate:
    def test_read_tool_not_intercepted(self):
        """Read-only tools should not be intercepted."""
        assert should_intercept("Read") is False

    def test_edit_tool_intercepted(self):
        """Edit tool should be intercepted."""
        assert should_intercept("Edit") is True

    def test_bash_tool_intercepted(self):
        """Bash tool should be intercepted."""
        assert should_intercept("Bash") is True

    def test_write_tool_intercepted(self):
        """Write tool should be intercepted."""
        assert should_intercept("Write") is True

    def test_system2_message_format(self):
        """System 2 activation message should be properly formatted."""
        msg = format_system2_message(confidence=0.65, reasoning="Complex refactor")
        assert "System 2" in msg
        assert "0.65" in msg


class TestPostToolEvaluate:
    def test_parse_success_result(self):
        """Successful tool results should parse correctly."""
        result = parse_tool_result(exit_code=0, stdout="All tests passed", stderr="")
        assert result["exit_code"] == 0
        assert "passed" in result["stdout"].lower()

    def test_parse_failure_result(self):
        """Failed tool results should parse correctly."""
        result = parse_tool_result(exit_code=1, stdout="", stderr="Error: file not found")
        assert result["exit_code"] == 1
```

**Step 2: Run test to verify it fails**

Run: `cd C:/Code/MetaScaffold && uv run pytest tests/test_hooks.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the hook implementations**

```python
# hooks/__init__.py
```

```python
# hooks/pre_tool_gate.py
"""PreToolUse hook — intercepts modifying actions for System 1/2 classification.

This script is called by Claude Code before each tool use.
It reads hook input from stdin (JSON) and outputs a message to stdout.

Exit codes:
- 0: Allow the tool call to proceed
- 2: Block the tool call and show the stdout message to Claude

The hook communicates with the MetaScaffold MCP server via the tools
that Claude already has access to, by injecting a guidance message.
"""

from __future__ import annotations

import json
import sys

# Tools that should trigger classification
_MODIFYING_TOOLS = {"Bash", "Edit", "Write", "NotebookEdit"}


def should_intercept(tool_name: str) -> bool:
    """Check if this tool should be intercepted for classification."""
    return tool_name in _MODIFYING_TOOLS


def format_system2_message(confidence: float, reasoning: str) -> str:
    """Format a message telling Claude to use System 2 deliberation."""
    return (
        f"[MetaScaffold] System 2 activated (confidence: {confidence:.2f}). "
        f"Reason: {reasoning}. "
        f"IMPORTANT: Before proceeding, call metascaffold_plan to create "
        f"a structured plan for this task. Then follow the recommended strategy."
    )


def main():
    """Entry point for the PreToolUse hook."""
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)  # Can't parse input, allow through

    tool_name = hook_input.get("tool_name", "")

    if not should_intercept(tool_name):
        sys.exit(0)  # Allow through

    # For now, output a guidance message suggesting classification.
    # The actual classification happens via the MCP tool that Claude calls.
    print(
        f"[MetaScaffold] Consider calling metascaffold_classify before "
        f"using {tool_name}. This helps determine if deep planning is needed.",
        file=sys.stderr,
    )
    sys.exit(0)  # Allow through (advisory, not blocking)


if __name__ == "__main__":
    main()
```

```python
# hooks/post_tool_evaluate.py
"""PostToolUse hook — sends execution results to the evaluator.

This script is called by Claude Code after each tool use.
It reads hook input from stdin (JSON) and can output guidance to stderr.
"""

from __future__ import annotations

import json
import sys


def parse_tool_result(exit_code: int, stdout: str, stderr: str) -> dict:
    """Parse a tool execution result into a structured dict."""
    return {
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
    }


def main():
    """Entry point for the PostToolUse hook."""
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")

    # Only evaluate modifying tools
    if tool_name not in {"Bash", "Edit", "Write", "NotebookEdit"}:
        sys.exit(0)

    # Advisory message suggesting evaluation
    tool_result = hook_input.get("tool_result", {})
    if isinstance(tool_result, dict) and tool_result.get("exit_code", 0) != 0:
        print(
            f"[MetaScaffold] {tool_name} returned non-zero exit code. "
            f"Consider calling metascaffold_evaluate to assess the result.",
            file=sys.stderr,
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `cd C:/Code/MetaScaffold && PYTHONPATH=. uv run pytest tests/test_hooks.py -v`
Expected: 7 tests PASS

**Step 5: Commit**

```bash
git add hooks/__init__.py hooks/pre_tool_gate.py hooks/post_tool_evaluate.py tests/test_hooks.py
git commit -m "feat: add Claude Code hooks for PreToolUse and PostToolUse"
```

---

### Task 14: Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write the integration test**

```python
# tests/test_integration.py
"""Integration tests — validate the full cognitive loop end-to-end.

Tests the pipeline: classify → plan → sandbox → evaluate → telemetry
without requiring a running MCP server or NotebookLM auth.
"""

import json
from pathlib import Path

from metascaffold.classifier import Classifier
from metascaffold.evaluator import Evaluator
from metascaffold.planner import Planner
from metascaffold.sandbox import Sandbox
from metascaffold.telemetry import CognitiveEvent, TelemetryLogger


class TestCognitiveLoop:
    """Tests the full System 2 cognitive loop."""

    def test_full_system2_loop_pass(self, tmp_path):
        """A complete System 2 loop that ends in 'pass'."""
        # 1. Classify
        classifier = Classifier(system2_threshold=0.8, always_system2_tools=["Write"])
        classification = classifier.classify(
            tool_name="Write",
            tool_input={"file_path": str(tmp_path / "test.py")},
            context="Create a new test file for the auth module",
        )
        assert classification.routing == "system2"

        # 2. Plan
        planner = Planner()
        plan = planner.create_plan(
            task="Create a new test file for the auth module",
            context=f"Target: {tmp_path / 'test.py'}",
        )
        assert len(plan.strategies) >= 1

        # 3. Execute in sandbox
        sandbox = Sandbox(work_dir=str(tmp_path), default_timeout_seconds=10)
        # Create a simple Python file and run it
        test_file = tmp_path / "test.py"
        test_file.write_text("print('test passed')")
        result = sandbox.execute(f"python {test_file}")
        assert result.exit_code == 0
        assert "test passed" in result.stdout

        # 4. Evaluate
        evaluator = Evaluator(max_retry_attempts=3)
        evaluation = evaluator.evaluate(sandbox_result=result, attempt=1)
        assert evaluation.verdict == "pass"

        # 5. Telemetry logs everything
        telemetry = TelemetryLogger(
            json_dir=str(tmp_path / "telemetry"),
            sqlite_path=str(tmp_path / "cognitive.db"),
        )
        telemetry.log(CognitiveEvent(
            event_type="classification",
            data={"routing": classification.routing, "confidence": classification.confidence},
        ))
        telemetry.log(CognitiveEvent(
            event_type="plan_created",
            data={"strategies": len(plan.strategies)},
        ))
        telemetry.log(CognitiveEvent(
            event_type="evaluation",
            data={"verdict": evaluation.verdict, "confidence": evaluation.confidence},
        ))
        telemetry.flush()

        # Verify JSON log
        json_files = list((tmp_path / "telemetry").glob("*.json"))
        assert len(json_files) == 1
        with open(json_files[0]) as f:
            events = json.load(f)
        assert len(events) == 3
        assert events[0]["event_type"] == "classification"
        assert events[2]["event_type"] == "evaluation"

    def test_full_system2_loop_retry_then_pass(self, tmp_path):
        """A System 2 loop where first attempt fails, retry succeeds."""
        sandbox = Sandbox(work_dir=str(tmp_path), default_timeout_seconds=10)
        evaluator = Evaluator(max_retry_attempts=3)

        # First attempt: failing script
        fail_script = tmp_path / "fail.py"
        fail_script.write_text("raise ValueError('bug')")
        result1 = sandbox.execute(f"python {fail_script}")
        eval1 = evaluator.evaluate(sandbox_result=result1, attempt=1)
        assert eval1.verdict in ("retry", "backtrack")

        # Second attempt: fixed script
        fix_script = tmp_path / "fix.py"
        fix_script.write_text("print('fixed')")
        result2 = sandbox.execute(f"python {fix_script}")
        eval2 = evaluator.evaluate(sandbox_result=result2, attempt=2)
        assert eval2.verdict == "pass"

    def test_system1_bypasses_planning(self):
        """System 1 tasks should not need planning."""
        classifier = Classifier(system2_threshold=0.8, always_system2_tools=[])
        classification = classifier.classify(
            tool_name="Read",
            tool_input={"file_path": "/tmp/readme.md"},
            context="Read the README file",
        )
        assert classification.routing == "system1"
        # System 1: no planning needed, proceed directly
```

**Step 2: Run integration tests**

Run: `cd C:/Code/MetaScaffold && uv run pytest tests/test_integration.py -v`
Expected: 3 tests PASS

**Step 3: Run full test suite**

Run: `cd C:/Code/MetaScaffold && uv run pytest -v`
Expected: All tests PASS (config: 4, telemetry: 5, nlm_bridge: 5, classifier: 6, planner: 5, sandbox: 6, evaluator: 6, server: 6, hooks: 7, integration: 3 = ~53 tests)

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add integration tests for full cognitive loop"
```

---

### Task 15: Final Wiring and Verification

**Step 1: Register the MCP server with Claude Code**

Run: `claude mcp add --transport stdio --scope user metascaffold -- uv --directory C:/Code/MetaScaffold run python src/metascaffold/server.py`

**Step 2: Verify MCP server is accessible**

Run: `claude mcp list`
Expected: `metascaffold` appears in the list

**Step 3: Run full test suite one final time**

Run: `cd C:/Code/MetaScaffold && uv run pytest -v --tb=short`
Expected: All tests PASS

**Step 4: Final commit with all remaining files**

```bash
git add -A
git commit -m "feat: complete MetaScaffold v0.1.0 — metacognition plugin for Claude Code"
```

---

## Summary

| Phase | Tasks | Tests | Key Deliverable |
|-------|-------|-------|-----------------|
| 1: Infrastructure | 1-5 | 14 | Project scaffold + config + NLM auth |
| 2: Knowledge Base | 6-7 | 0 (scripts) | 15 papers + 5 repos ingested |
| 3: Core Components | 8-11 | 28 | Classifier + Planner + Sandbox + Evaluator |
| 4: Integration | 12-15 | 16 | MCP Server + Hooks + Integration tests |
| **Total** | **15 tasks** | **~58 tests** | **MetaScaffold v0.1.0** |
