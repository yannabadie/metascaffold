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
