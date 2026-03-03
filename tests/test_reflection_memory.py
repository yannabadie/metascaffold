"""Tests for Ebbinghaus decay reflection memory."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from metascaffold.reflection_memory import ReflectionMemory, ReflectionRule


class TestReflectionRule:
    def test_default_retention_is_one(self):
        """New rule should have retention_strength=1.0."""
        rule = ReflectionRule(content="Always run tests")
        assert rule.retention_strength == 1.0

    def test_compute_retention_decays_over_time(self):
        """A rule from 168h ago should have 0 < retention < 1."""
        rule = ReflectionRule(content="Check imports first")
        rule.last_reinforced = datetime.now(timezone.utc) - timedelta(hours=168)
        retention = rule.compute_retention(stability_hours=168.0)
        assert 0 < retention < 1

    def test_reinforced_rule_decays_slower(self):
        """A rule with reinforcement_count=5 should decay slower than one with 0."""
        hours_ago = 168
        past = datetime.now(timezone.utc) - timedelta(hours=hours_ago)

        rule_weak = ReflectionRule(content="rule A")
        rule_weak.last_reinforced = past
        rule_weak.reinforcement_count = 0

        rule_strong = ReflectionRule(content="rule B")
        rule_strong.last_reinforced = past
        rule_strong.reinforcement_count = 5

        retention_weak = rule_weak.compute_retention(stability_hours=168.0)
        retention_strong = rule_strong.compute_retention(stability_hours=168.0)
        assert retention_strong > retention_weak

    def test_fresh_rule_has_full_retention(self):
        """A just-created rule should have compute_retention() close to 1.0."""
        rule = ReflectionRule(content="New rule")
        retention = rule.compute_retention()
        assert retention == pytest.approx(1.0, abs=0.01)

    def test_to_dict_roundtrip(self):
        """to_dict() -> from_dict() should preserve content and reinforcement_count."""
        rule = ReflectionRule(
            content="Always validate input",
            reinforcement_count=3,
        )
        data = rule.to_dict()
        restored = ReflectionRule.from_dict(data)
        assert restored.content == rule.content
        assert restored.reinforcement_count == rule.reinforcement_count


class TestReflectionMemory:
    def test_add_rule(self):
        """Adding a rule should increase rules count and match content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ReflectionMemory(storage_path=Path(tmpdir) / "mem.json")
            rule = mem.add_rule("Always run tests")
            assert len(mem.rules) == 1
            assert rule.content == "Always run tests"

    def test_reinforce_existing_rule(self):
        """Reinforcing an existing rule should increment its count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ReflectionMemory(storage_path=Path(tmpdir) / "mem.json")
            mem.add_rule("Check imports first")
            found = mem.reinforce("Check imports first")
            assert found is True
            assert mem.rules[0].reinforcement_count == 1

    def test_prune_removes_decayed_rules(self):
        """A rule from 30 days ago with prune_threshold=0.5 should be pruned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ReflectionMemory(
                storage_path=Path(tmpdir) / "mem.json",
                prune_threshold=0.5,
            )
            rule = mem.add_rule("Old rule")
            rule.last_reinforced = datetime.now(timezone.utc) - timedelta(days=30)
            pruned = mem.prune()
            assert len(pruned) == 1
            assert pruned[0].content == "Old rule"
            assert len(mem.rules) == 0

    def test_save_and_load(self):
        """Save then load into a new ReflectionMemory should preserve rules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mem.json"
            mem1 = ReflectionMemory(storage_path=path)
            mem1.add_rule("Rule A")
            mem1.add_rule("Rule B")
            mem1.save()

            mem2 = ReflectionMemory(storage_path=path)
            mem2.load()
            assert len(mem2.rules) == 2
            assert mem2.rules[0].content == "Rule A"
            assert mem2.rules[1].content == "Rule B"

    def test_get_active_rules_filters_by_retention(self):
        """Only fresh rules should appear; very old rules should be filtered out."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ReflectionMemory(storage_path=Path(tmpdir) / "mem.json")
            fresh = mem.add_rule("Fresh rule")
            old = mem.add_rule("Old rule")
            old.last_reinforced = datetime.now(timezone.utc) - timedelta(days=60)

            active = mem.get_active_rules(min_retention=0.3)
            contents = [r.content for r in active]
            assert "Fresh rule" in contents
            assert "Old rule" not in contents
