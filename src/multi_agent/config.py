"""Configuration loading from TOML files."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    enabled: bool = True
    model: str | None = None
    system_prompt_override: str | None = None


@dataclass
class GeneralConfig:
    file_patterns: list[str] = field(default_factory=lambda: ["*.md", "*.txt"])
    consensus_threshold: int = 2
    timeout_seconds: int = 600
    canon_directories: list[str] = field(default_factory=lambda: ["canon"])
    max_canon_size_kb: int = 500


@dataclass
class MultiAgentConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    agents: dict[str, AgentConfig] = field(default_factory=lambda: {
        "scientific_rigor": AgentConfig(),
        "canon_continuity": AgentConfig(),
        "sociopolitical": AgentConfig(),
    })


def _find_config_file(start: Path) -> Path | None:
    """Walk up from start to find multi_agent.toml."""
    current = start.resolve()
    while True:
        candidate = current / "multi_agent.toml"
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def load_config(path: Path | None = None) -> MultiAgentConfig:
    """Load configuration from a TOML file.

    If no path is given, searches upward from cwd for multi_agent.toml.
    Falls back to defaults if no file is found.
    """
    if path is None:
        path = _find_config_file(Path.cwd())

    if path is None:
        return MultiAgentConfig()

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    config = MultiAgentConfig()

    if "general" in raw:
        g = raw["general"]
        for fld in (
            "file_patterns", "consensus_threshold",
            "timeout_seconds", "canon_directories", "max_canon_size_kb",
        ):
            if fld in g:
                setattr(config.general, fld, g[fld])

    if "agents" in raw:
        for name, agent_raw in raw["agents"].items():
            agent_cfg = AgentConfig(
                enabled=agent_raw.get("enabled", True),
                model=agent_raw.get("model"),
                system_prompt_override=agent_raw.get("system_prompt_override"),
            )
            config.agents[name] = agent_cfg

    enabled_count = sum(1 for a in config.agents.values() if a.enabled)
    if config.general.consensus_threshold > enabled_count:
        raise ValueError(
            f"consensus_threshold ({config.general.consensus_threshold}) "
            f"exceeds enabled agent count ({enabled_count})"
        )
    if enabled_count < 2:
        raise ValueError("At least 2 agents must be enabled")

    return config
