"""Tests for the configuration module."""

import os
import tempfile
from pathlib import Path

from metascaffold.config import MetaScaffoldConfig, load_config
from metascaffold.config import ClassifierConfig, VerifierConfig, MemoryConfig


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

    def test_llm_config_defaults(self):
        """LLM config should have default models for each component."""
        cfg = load_config()
        assert cfg.llm.classifier_model == "gpt-4.1-nano"
        assert cfg.llm.evaluator_model == "o3-mini"
        assert cfg.llm.planner_model == "gpt-4.1-mini"
        assert cfg.llm.distiller_model == "gpt-4.1-nano"
        assert cfg.llm.reflector_model == "o3-mini"
        assert cfg.llm.enabled is True
        assert cfg.llm.fallback_to_heuristics is True


class TestV03Config:
    def test_entropy_config_has_defaults(self):
        """ClassifierConfig should have entropy thresholds with correct defaults."""
        cfg = ClassifierConfig()
        assert cfg.entropy_threshold == 0.5
        assert cfg.medium_entropy_threshold == 0.3

    def test_verifier_config_has_defaults(self):
        """VerifierConfig should have correct boolean and timeout defaults."""
        cfg = VerifierConfig()
        assert cfg.run_ast is True
        assert cfg.run_ruff is True
        assert cfg.run_mypy is False
        assert cfg.run_pytest is False

    def test_memory_config_has_defaults(self):
        """MemoryConfig should have correct numeric defaults."""
        cfg = MemoryConfig()
        assert cfg.prune_threshold == 0.1
        assert cfg.stability_hours == 168.0
