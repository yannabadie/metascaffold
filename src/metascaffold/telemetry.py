"""Cognitive telemetry -- JSON session logs + SQLite historical database.

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

    def get_recent_events(self, count: int = 50) -> list[dict]:
        """Return the most recent *count* events from SQLite.

        Each dict has keys: session_id, timestamp, event_type, data.
        Returns an empty list if the database is empty or on error.
        """
        try:
            conn = sqlite3.connect(self._sqlite_path)
            rows = conn.execute(
                """
                SELECT session_id, timestamp, event_type, data_json
                FROM events
                ORDER BY id DESC
                LIMIT ?
                """,
                (count,),
            ).fetchall()
            conn.close()
            events = []
            for session_id, timestamp, event_type, data_json in reversed(rows):
                events.append({
                    "session_id": session_id,
                    "timestamp": timestamp,
                    "event_type": event_type,
                    "data": json.loads(data_json) if data_json else {},
                })
            return events
        except Exception:
            return []

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
