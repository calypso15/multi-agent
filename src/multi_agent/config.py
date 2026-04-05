"""Configuration loading from TOML files."""

from __future__ import annotations

import dataclasses
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def _field_names(cls: type) -> set[str]:
    """Return the set of field names for a dataclass."""
    return {f.name for f in dataclasses.fields(cls)}


@dataclass
class AgentConfig:
    system_prompt: str | None = None
    display_name: str | None = None
    enabled: bool = True
    propose_model: str | None = None
    review_model: str | None = None
    allowed_tools: list[str] = field(default_factory=list)
    propose_max_turns: int | None = None
    review_max_turns: int | None = None


def get_display_name(agent_key: str, agent_cfg: AgentConfig) -> str:
    """Return the display name for an agent, defaulting to titlecased key."""
    return agent_cfg.display_name or agent_key.replace("_", " ").title()


KNOWN_BACKENDS = {"claude-cli"}


@dataclass
class GeneralConfig:
    backend: str = "claude-cli"
    file_patterns: list[str] = field(default_factory=lambda: ["*.md", "*.txt"])
    consensus_threshold: int = 2
    timeout_seconds: int = 600
    reference_directories: list[str] = field(default_factory=lambda: ["reference"])
    max_reference_size_kb: int = 500
    max_rounds: int = 3
    min_severity: str = "minor"
    min_blocking_severity: str = "major"
    propose_max_turns: int = 0  # 0 = unlimited (no --max-turns flag)
    review_max_turns: int = 0


@dataclass
class CommandConfig:
    prompt: str = ""
    description: str = ""
    agents: list[str] = field(default_factory=list)
    consensus_threshold: int | None = None
    propose_model: str | None = None
    review_model: str | None = None
    propose_instructions: str | None = None


# Default review command, used when absent from TOML.
DEFAULT_REVIEW_COMMAND = CommandConfig(
    description="Review files and propose changes via consensus",
    prompt=(
        "Review the submitted content from your specialty perspective "
        "and propose CONCRETE edits to improve it. Keep edits minimal "
        "\u2014 change only what is necessary to fix the issue."
    ),
)

DEFAULT_ASK_COMMAND = CommandConfig(
    description="Ask a question and get a consensus answer",
    prompt=(
        "The file contains a QUESTION, not content to review. "
        "You MUST answer it by proposing exactly one edit that replaces the "
        "entire file content with your answer.\n\n"
        "Instructions:\n"
        "1. Read the reference files to gather information relevant to the question.\n"
        "2. Propose exactly ONE edit where original_text is the full file content "
        "(the question) and replacement_text is your complete Markdown answer.\n"
        "3. Set severity to \"major\".\n"
        "4. You MUST always propose this edit. Never return an empty edits array. "
        "The question always needs an answer."
    ),
    propose_instructions=(
        "\n# YOUR TASK\n"
        "The file above contains a QUESTION. You must ANSWER it.\n\n"
        "Read the reference files to find relevant information, then propose "
        "exactly one edit that replaces the entire file content (the question) "
        "with your complete answer.\n\n"
        "You MUST propose an edit. Do NOT return an empty edits array.\n\n"
        "Return your response as JSON.\n"
    ),
)


@dataclass
class MultiAgentConfig:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    agents: dict[str, AgentConfig] = field(default_factory=dict)
    commands: dict[str, CommandConfig] = field(default_factory=dict)


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
        config = MultiAgentConfig()
        if "review" not in config.commands:
            config.commands["review"] = dataclasses.replace(DEFAULT_REVIEW_COMMAND)
        if "ask" not in config.commands:
            config.commands["ask"] = dataclasses.replace(DEFAULT_ASK_COMMAND)
        return config

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    config = MultiAgentConfig()

    if "general" in raw:
        g = raw["general"]
        valid = _field_names(GeneralConfig)
        unknown = set(g.keys()) - valid
        if unknown:
            raise ValueError(
                f"Unknown key(s) in [general]: {', '.join(sorted(unknown))}"
            )
        for fld in valid:
            if fld in g:
                setattr(config.general, fld, g[fld])

    if "agents" in raw:
        valid = _field_names(AgentConfig)
        for name, agent_raw in raw["agents"].items():
            unknown = set(agent_raw.keys()) - valid
            if unknown:
                raise ValueError(
                    f"Unknown key(s) in [agents.{name}]: "
                    f"{', '.join(sorted(unknown))}"
                )
            agent_cfg = AgentConfig(
                **{k: v for k, v in agent_raw.items() if k in valid}
            )
            config.agents[name] = agent_cfg

    if "tasks" in raw and "commands" in raw:
        raise ValueError(
            "Config contains both [tasks] and [commands]. "
            "Migrate [tasks] entries to [commands]."
        )

    commands_raw = raw.get("commands") or raw.get("tasks")
    if commands_raw is not None:
        if "tasks" in raw:
            print(
                "Warning: [tasks] is deprecated, migrate to [commands]. "
                "See multi_agent.example.toml.",
                file=sys.stderr,
            )
        section = "commands" if "commands" in raw else "tasks"
        valid = _field_names(CommandConfig)
        for name, cmd_raw in commands_raw.items():
            unknown = set(cmd_raw.keys()) - valid
            if unknown:
                raise ValueError(
                    f"Unknown key(s) in [{section}.{name}]: "
                    f"{', '.join(sorted(unknown))}"
                )
            config.commands[name] = CommandConfig(
                **{k: v for k, v in cmd_raw.items() if k in valid}
            )

    from multi_agent.claude_runner import KNOWN_TOOLS

    for name, agent_cfg in config.agents.items():
        invalid = set(agent_cfg.allowed_tools) - KNOWN_TOOLS
        if invalid:
            raise ValueError(
                f"Unknown tool(s) in [agents.{name}].allowed_tools: "
                f"{', '.join(sorted(invalid))}"
            )

    for name, agent_cfg in config.agents.items():
        if agent_cfg.enabled and not agent_cfg.system_prompt:
            raise ValueError(
                f"Agent '{name}' is missing required 'system_prompt' field"
            )
        if agent_cfg.display_name is not None and not agent_cfg.display_name:
            raise ValueError(
                f"Agent '{name}' has an empty 'display_name' — "
                "omit the field to use the default, or provide a non-empty value"
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
    if config.general.min_blocking_severity not in valid_severities:
        raise ValueError(
            f"min_blocking_severity must be one of {valid_severities}, "
            f"got '{config.general.min_blocking_severity}'"
        )
    from multi_agent.models import severity_index
    if severity_index(config.general.min_blocking_severity) > severity_index(
        config.general.min_severity,
    ):
        raise ValueError(
            f"min_blocking_severity ('{config.general.min_blocking_severity}') "
            f"cannot be less severe than min_severity "
            f"('{config.general.min_severity}')"
        )
    if config.general.backend not in KNOWN_BACKENDS:
        raise ValueError(
            f"backend must be one of {sorted(KNOWN_BACKENDS)}, "
            f"got '{config.general.backend}'"
        )

    # Insert default commands if missing; merge TOML overrides on top of
    # defaults for built-in commands so users only need to specify the fields
    # they want to change.
    for builtin_name, builtin_default in (
        ("review", DEFAULT_REVIEW_COMMAND),
        ("ask", DEFAULT_ASK_COMMAND),
    ):
        if builtin_name not in config.commands:
            config.commands[builtin_name] = dataclasses.replace(builtin_default)
        else:
            merged = dataclasses.replace(builtin_default)
            for fld in dataclasses.fields(CommandConfig):
                val = getattr(config.commands[builtin_name], fld.name)
                if val != fld.default and val != (
                    fld.default_factory() if fld.default_factory is not dataclasses.MISSING  # type: ignore[comparison-overlap]
                    else fld.default
                ):
                    setattr(merged, fld.name, val)
            config.commands[builtin_name] = merged

    enabled_agents = {k for k, v in config.agents.items() if v.enabled}
    for name, cmd_cfg in config.commands.items():
        if not cmd_cfg.prompt:
            raise ValueError(
                f"Command '{name}' is missing required 'prompt' field"
            )
        if cmd_cfg.agents:
            unknown = set(cmd_cfg.agents) - set(config.agents.keys())
            if unknown:
                raise ValueError(
                    f"Command '{name}' references unknown agent(s): "
                    f"{', '.join(sorted(unknown))}"
                )
            disabled = set(cmd_cfg.agents) - enabled_agents
            if disabled:
                raise ValueError(
                    f"Command '{name}' references disabled agent(s): "
                    f"{', '.join(sorted(disabled))}"
                )
        if cmd_cfg.consensus_threshold is not None:
            agent_count = len(cmd_cfg.agents) if cmd_cfg.agents else enabled_count
            if cmd_cfg.consensus_threshold > agent_count:
                raise ValueError(
                    f"Command '{name}' consensus_threshold "
                    f"({cmd_cfg.consensus_threshold}) exceeds its "
                    f"agent count ({agent_count})"
                )

    return config
