"""Ebbinghaus decay reflection memory for MetaScaffold.

Stores reflection rules with forgetting-curve retention decay.
Unused rules fade over time; reinforced rules persist proportionally longer.

The Ebbinghaus forgetting curve:
    retention = e^(-t / (stability * reinforcement_factor))

where:
    t = hours since last reinforcement
    stability = base half-life in hours (default 168h = 1 week)
    reinforcement_factor = 1 + log(1 + reinforcement_count)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("metascaffold.reflection_memory")


@dataclass
class ReflectionRule:
    """A reflection rule with Ebbinghaus forgetting-curve decay."""

    content: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_reinforced: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retention_strength: float = 1.0
    reinforcement_count: int = 0
    source_events: list[str] = field(default_factory=list)

    def compute_retention(self, stability_hours: float = 168.0) -> float:
        """Compute current retention using the Ebbinghaus forgetting curve.

        retention = e^(-t / (stability * reinforcement_factor))

        where t = hours since last_reinforced,
              reinforcement_factor = 1 + log(1 + reinforcement_count)
        """
        now = datetime.now(timezone.utc)
        elapsed = (now - self.last_reinforced).total_seconds() / 3600.0
        reinforcement_factor = 1 + math.log(1 + self.reinforcement_count)
        exponent = -elapsed / (stability_hours * reinforcement_factor)
        return math.exp(exponent)

    def reinforce(self) -> None:
        """Reinforce this rule: reset timer, increment count, reset strength."""
        self.last_reinforced = datetime.now(timezone.utc)
        self.reinforcement_count += 1
        self.retention_strength = 1.0

    def to_dict(self) -> dict:
        """Serialize all fields to a dictionary. Datetimes as ISO strings."""
        return {
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "last_reinforced": self.last_reinforced.isoformat(),
            "retention_strength": self.retention_strength,
            "reinforcement_count": self.reinforcement_count,
            "source_events": self.source_events,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ReflectionRule:
        """Deserialize a ReflectionRule from a dictionary."""
        return cls(
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_reinforced=datetime.fromisoformat(data["last_reinforced"]),
            retention_strength=data.get("retention_strength", 1.0),
            reinforcement_count=data.get("reinforcement_count", 0),
            source_events=data.get("source_events", []),
        )


class ReflectionMemory:
    """Manages a collection of reflection rules with Ebbinghaus decay.

    Rules that are not reinforced gradually lose retention.
    Rules below the prune threshold are removed during prune().
    """

    def __init__(
        self,
        storage_path: Path | str | None = None,
        prune_threshold: float = 0.1,
        stability_hours: float = 168.0,
    ):
        if storage_path is None:
            self.storage_path = Path.home() / ".metascaffold" / "reflection_memory.json"
        else:
            self.storage_path = Path(storage_path)
        self.prune_threshold = prune_threshold
        self.stability_hours = stability_hours
        self.rules: list[ReflectionRule] = []

    def add_rule(self, content: str, source_events: list[str] | None = None) -> ReflectionRule:
        """Add a new reflection rule."""
        rule = ReflectionRule(
            content=content,
            source_events=source_events or [],
        )
        self.rules.append(rule)
        return rule

    def reinforce(self, content: str) -> bool:
        """Find a rule by content match and reinforce it.

        Returns True if a matching rule was found and reinforced.
        """
        for rule in self.rules:
            if rule.content == content:
                rule.reinforce()
                return True
        return False

    def prune(self) -> list[ReflectionRule]:
        """Remove rules with retention below prune_threshold.

        Returns the list of pruned (removed) rules.
        """
        pruned = []
        remaining = []
        for rule in self.rules:
            retention = rule.compute_retention(stability_hours=self.stability_hours)
            if retention < self.prune_threshold:
                pruned.append(rule)
            else:
                remaining.append(rule)
        self.rules = remaining
        return pruned

    def get_active_rules(self, min_retention: float = 0.3) -> list[ReflectionRule]:
        """Return rules with retention above the given threshold."""
        active = []
        for rule in self.rules:
            retention = rule.compute_retention(stability_hours=self.stability_hours)
            if retention >= min_retention:
                active.append(rule)
        return active

    def save(self) -> None:
        """Write rules to JSON file."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = [rule.to_dict() for rule in self.rules]
        self.storage_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("Saved %d reflection rules to %s", len(self.rules), self.storage_path)

    def load(self) -> None:
        """Read rules from JSON file. Handles missing file and parse errors."""
        if not self.storage_path.exists():
            logger.debug("No reflection memory file at %s", self.storage_path)
            return
        try:
            text = self.storage_path.read_text(encoding="utf-8")
            data = json.loads(text)
            self.rules = [ReflectionRule.from_dict(item) for item in data]
            logger.debug("Loaded %d reflection rules from %s", len(self.rules), self.storage_path)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse reflection memory: %s", e)
            self.rules = []
