"""Configuration loading from TOML files."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentConfig:
    enabled: bool = True
    propose_model: str | None = None
    review_model: str | None = None
    system_prompt_override: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    propose_max_turns: int | None = None
    review_max_turns: int | None = None


@dataclass
class GeneralConfig:
    file_patterns: list[str] = field(default_factory=lambda: ["*.md", "*.txt"])
    consensus_threshold: int = 2
    timeout_seconds: int = 600
    canon_directories: list[str] = field(default_factory=lambda: ["canon"])
    max_canon_size_kb: int = 500
    max_rounds: int = 3
    min_severity: str = "minor"
    propose_max_turns: int = 3
    review_max_turns: int = 2


@dataclass
class TaskConfig:
    prompt: str = ""


@dataclass
class MultiAgentConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    agents: dict[str, AgentConfig] = field(default_factory=lambda: {
        "scientific_rigor": AgentConfig(),
        "canon_continuity": AgentConfig(),
        "sociopolitical": AgentConfig(),
    })
    tasks: dict[str, TaskConfig] = field(default_factory=dict)


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


def load_config(
    path: Path | None = None,
    search_from: Path | None = None,
) -> MultiAgentConfig:
    """Load configuration from a TOML file.

    If no path is given, searches upward from search_from (or cwd) for
    multi_agent.toml. Falls back to defaults if no file is found.
    """
    if path is None:
        path = _find_config_file(search_from or Path.cwd())

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
            "max_rounds", "min_severity",
            "propose_max_turns", "review_max_turns",
        ):
            if fld in g:
                setattr(config.general, fld, g[fld])

    if "agents" in raw:
        for name, agent_raw in raw["agents"].items():
            agent_cfg = AgentConfig(
                enabled=agent_raw.get("enabled", True),
                propose_model=agent_raw.get("propose_model"),
                review_model=agent_raw.get("review_model"),
                system_prompt_override=agent_raw.get("system_prompt_override"),
                allowed_tools=agent_raw.get("allowed_tools", []),
                propose_max_turns=agent_raw.get("propose_max_turns"),
                review_max_turns=agent_raw.get("review_max_turns"),
            )
            config.agents[name] = agent_cfg

    if "tasks" in raw:
        for name, task_raw in raw["tasks"].items():
            config.tasks[name] = TaskConfig(
                prompt=task_raw.get("prompt", ""),
            )

    enabled_count = sum(1 for a in config.agents.values() if a.enabled)
    if config.general.consensus_threshold > enabled_count:
        raise ValueError(
            f"consensus_threshold ({config.general.consensus_threshold}) "
            f"exceeds enabled agent count ({enabled_count})"
        )
    if enabled_count < 2:
        raise ValueError("At least 2 agents must be enabled")
    if config.general.max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    valid_severities = ("critical", "major", "minor", "suggestion")
    if config.general.min_severity not in valid_severities:
        raise ValueError(
            f"min_severity must be one of {valid_severities}, "
            f"got '{config.general.min_severity}'"
        )

    return config
