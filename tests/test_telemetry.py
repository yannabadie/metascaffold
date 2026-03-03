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
