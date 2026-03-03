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
