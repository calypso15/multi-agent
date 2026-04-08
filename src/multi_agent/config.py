"""Configuration loading from TOML files."""

from __future__ import annotations

import dataclasses
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _field_names(cls: type) -> set[str]:
    """Return the set of field names for a dataclass."""
    return {f.name for f in dataclasses.fields(cls)}


# ---------------------------------------------------------------------------
# Raw config dataclasses (loaded from TOML)
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    # Agent-only
    system_prompt: str | None = None
    display_name: str | None = None
    enabled: bool = True
    # Per-agent cascading (None = inherit from general)
    weight: int | None = None
    propose_model: str | None = None
    review_model: str | None = None
    propose_max_turns: int | None = None
    review_max_turns: int | None = None
    timeout_seconds: int | None = None
    allowed_tools: list[str] | None = None
    file_patterns: list[str] | None = None
    reference_directories: list[str] | None = None
    max_reference_size_kb: int | None = None


def get_display_name(agent_key: str, agent_cfg: AgentConfig) -> str:
    """Return the display name for an agent, defaulting to titlecased key."""
    return agent_cfg.display_name or agent_key.replace("_", " ").title()


KNOWN_BACKENDS = {"claude-cli"}


@dataclass
class GeneralConfig:
    # Global-only
    backend: str = "claude-cli"
    # Per-agent cascading defaults
    propose_model: str | None = None   # None = backend default
    review_model: str | None = None
    propose_max_turns: int = 0  # 0 = unlimited (no --max-turns flag)
    review_max_turns: int = 0
    timeout_seconds: int = 600
    allowed_tools: list[str] = field(default_factory=list)
    file_patterns: list[str] = field(default_factory=lambda: ["*.md", "*.txt"])
    reference_directories: list[str] = field(default_factory=lambda: ["reference"])
    max_reference_size_kb: int = 500
    # Per-run cascading defaults
    consensus_threshold: int = 2
    max_rounds: int = 3
    min_severity: str = "minor"
    min_blocking_severity: str = "major"


@dataclass
class CommandAgentConfig:
    """Per-agent overrides within a command."""
    weight: int | None = None
    propose_model: str | None = None
    review_model: str | None = None
    propose_max_turns: int | None = None
    review_max_turns: int | None = None
    timeout_seconds: int | None = None
    allowed_tools: list[str] | None = None
    file_patterns: list[str] | None = None
    reference_directories: list[str] | None = None
    max_reference_size_kb: int | None = None


@dataclass
class CommandConfig:
    # Command-only
    prompt: str = ""
    description: str = ""
    agents: dict[str, CommandAgentConfig] = field(default_factory=dict)
    propose_instructions: str | None = None
    # Per-agent cascading (None = inherit)
    propose_model: str | None = None
    review_model: str | None = None
    propose_max_turns: int | None = None
    review_max_turns: int | None = None
    timeout_seconds: int | None = None
    allowed_tools: list[str] | None = None
    file_patterns: list[str] | None = None
    reference_directories: list[str] | None = None
    max_reference_size_kb: int | None = None
    # Per-run cascading (None = inherit)
    consensus_threshold: int | None = None
    max_rounds: int | None = None
    min_severity: str | None = None
    min_blocking_severity: str | None = None


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


# ---------------------------------------------------------------------------
# Resolved config dataclasses (fully resolved, no None for non-model fields)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedAgentSettings:
    """Fully resolved settings for one agent in the context of one command."""
    weight: int
    propose_model: str | None      # None = backend default
    review_model: str | None       # None = backend default
    propose_max_turns: int
    review_max_turns: int
    timeout_seconds: int
    allowed_tools: list[str]
    file_patterns: list[str]
    reference_directories: list[str]
    max_reference_size_kb: int


@dataclass(frozen=True)
class ResolvedRunConfig:
    """Everything needed for one run, fully resolved.

    Built once in cli.py, threaded through consensus.py.
    Replaces the (config, command_*) parameter bundle.
    """
    # Per-agent resolved settings
    agent_settings: dict[str, ResolvedAgentSettings]
    # Per-run resolved settings
    max_rounds: int
    min_severity: str
    min_blocking_severity: str
    consensus_threshold: int
    # Passthrough (level-specific, no cascade)
    agents: dict[str, AgentConfig]
    backend: str
    command_name: str | None
    command_prompt: str | None
    command_propose_instructions: str | None


# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------

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
        _insert_builtin_commands(config)
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
        cmd_agent_valid = _field_names(CommandAgentConfig)
        for name, cmd_raw in commands_raw.items():
            unknown = set(cmd_raw.keys()) - valid
            if unknown:
                raise ValueError(
                    f"Unknown key(s) in [{section}.{name}]: "
                    f"{', '.join(sorted(unknown))}"
                )
            # Convert agents field: list → dict with empty overrides,
            # dict → dict with CommandAgentConfig values.
            fields = {k: v for k, v in cmd_raw.items() if k in valid}
            if "agents" in fields:
                raw_agents = fields["agents"]
                if isinstance(raw_agents, list):
                    fields["agents"] = {
                        n: CommandAgentConfig() for n in raw_agents
                    }
                elif isinstance(raw_agents, dict):
                    parsed_agents = {}
                    for agent_name, overrides in raw_agents.items():
                        if not isinstance(overrides, dict):
                            raise ValueError(
                                f"[{section}.{name}.agents.{agent_name}] "
                                "must be a table, not a scalar"
                            )
                        bad = set(overrides.keys()) - cmd_agent_valid
                        if bad:
                            raise ValueError(
                                f"Unknown key(s) in "
                                f"[{section}.{name}.agents.{agent_name}]: "
                                f"{', '.join(sorted(bad))}"
                            )
                        parsed_agents[agent_name] = CommandAgentConfig(
                            **{k: v for k, v in overrides.items()
                               if k in cmd_agent_valid}
                        )
                    fields["agents"] = parsed_agents
            config.commands[name] = CommandConfig(**fields)

    # --- Structural validation (raw config) ---

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
    if enabled_count < 2:
        raise ValueError("At least 2 agents must be enabled")

    if config.general.backend not in KNOWN_BACKENDS:
        raise ValueError(
            f"backend must be one of {sorted(KNOWN_BACKENDS)}, "
            f"got '{config.general.backend}'"
        )

    _insert_builtin_commands(config)

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

    return config


def _insert_builtin_commands(config: MultiAgentConfig) -> None:
    """Insert or merge built-in command defaults."""
    for builtin_name, builtin_default in (
        ("review", DEFAULT_REVIEW_COMMAND),
        ("ask", DEFAULT_ASK_COMMAND),
    ):
        if builtin_name not in config.commands:
            config.commands[builtin_name] = dataclasses.replace(builtin_default)
        else:
            merged = dataclasses.replace(builtin_default)
            blank = CommandConfig()
            for fld in dataclasses.fields(CommandConfig):
                val = getattr(config.commands[builtin_name], fld.name)
                if val != getattr(blank, fld.name):
                    setattr(merged, fld.name, val)
            config.commands[builtin_name] = merged


# ---------------------------------------------------------------------------
# Cascading resolution
# ---------------------------------------------------------------------------

def _first_set(*values: Any) -> Any:
    """Return the first value that is not None."""
    for v in values:
        if v is not None:
            return v
    return None


def resolve_run_config(
    config: MultiAgentConfig,
    cmd_name: str | None = None,
    cmd_config: CommandConfig | None = None,
) -> ResolvedRunConfig:
    """Build a fully resolved run config from raw config + command.

    Applies cascading: command-agent > command > agent > general for
    per-agent settings, command > general for per-run settings.
    Agent filtering from cmd_config.agents is applied here.
    """
    cmd = cmd_config or CommandConfig()
    general = config.general

    # Determine which agents participate
    if cmd.agents:
        agents = {k: v for k, v in config.agents.items()
                  if k in cmd.agents and v.enabled}
    else:
        agents = {k: v for k, v in config.agents.items() if v.enabled}

    # Resolve per-agent settings (command-agent > command > agent > general)
    agent_settings: dict[str, ResolvedAgentSettings] = {}
    for name, agent_cfg in agents.items():
        ca = cmd.agents.get(name, CommandAgentConfig()) if cmd.agents else CommandAgentConfig()

        propose_model = _first_set(
            ca.propose_model,
            cmd.propose_model, agent_cfg.propose_model, general.propose_model,
        )
        review_model = _first_set(
            ca.review_model,
            cmd.review_model, agent_cfg.review_model, general.review_model,
        )
        # review_model falls back to propose_model if still None
        if review_model is None:
            review_model = propose_model

        agent_settings[name] = ResolvedAgentSettings(
            weight=_first_set(ca.weight, agent_cfg.weight, 1),
            propose_model=propose_model,
            review_model=review_model,
            propose_max_turns=_first_set(
                ca.propose_max_turns,
                cmd.propose_max_turns, agent_cfg.propose_max_turns,
                general.propose_max_turns,
            ),
            review_max_turns=_first_set(
                ca.review_max_turns,
                cmd.review_max_turns, agent_cfg.review_max_turns,
                general.review_max_turns,
            ),
            timeout_seconds=_first_set(
                ca.timeout_seconds,
                cmd.timeout_seconds, agent_cfg.timeout_seconds,
                general.timeout_seconds,
            ),
            allowed_tools=_first_set(
                ca.allowed_tools,
                cmd.allowed_tools, agent_cfg.allowed_tools,
                general.allowed_tools,
            ),
            file_patterns=_first_set(
                ca.file_patterns,
                cmd.file_patterns, agent_cfg.file_patterns,
                general.file_patterns,
            ),
            reference_directories=_first_set(
                ca.reference_directories,
                cmd.reference_directories, agent_cfg.reference_directories,
                general.reference_directories,
            ),
            max_reference_size_kb=_first_set(
                ca.max_reference_size_kb,
                cmd.max_reference_size_kb, agent_cfg.max_reference_size_kb,
                general.max_reference_size_kb,
            ),
        )

    # Resolve per-run settings (command > general)
    max_rounds = _first_set(cmd.max_rounds, general.max_rounds)
    min_severity = _first_set(cmd.min_severity, general.min_severity)
    min_blocking_severity = _first_set(
        cmd.min_blocking_severity, general.min_blocking_severity,
    )
    consensus_threshold = _first_set(
        cmd.consensus_threshold, general.consensus_threshold,
    )
    total_weight = sum(s.weight for s in agent_settings.values())
    consensus_threshold = min(consensus_threshold, total_weight)

    # --- Semantic validation on resolved values ---
    valid_severities = ("critical", "major", "minor", "suggestion")
    if min_severity not in valid_severities:
        raise ValueError(
            f"min_severity must be one of {valid_severities}, "
            f"got '{min_severity}'"
        )
    if min_blocking_severity not in valid_severities:
        raise ValueError(
            f"min_blocking_severity must be one of {valid_severities}, "
            f"got '{min_blocking_severity}'"
        )
    from multi_agent.models import severity_index
    if severity_index(min_blocking_severity) > severity_index(min_severity):
        raise ValueError(
            f"min_blocking_severity ('{min_blocking_severity}') "
            f"cannot be less severe than min_severity "
            f"('{min_severity}')"
        )
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    if consensus_threshold > total_weight:
        raise ValueError(
            f"consensus_threshold ({consensus_threshold}) "
            f"exceeds total agent weight ({total_weight})"
        )

    return ResolvedRunConfig(
        agent_settings=agent_settings,
        max_rounds=max_rounds,
        min_severity=min_severity,
        min_blocking_severity=min_blocking_severity,
        consensus_threshold=consensus_threshold,
        agents=agents,
        backend=general.backend,
        command_name=cmd_name,
        command_prompt=cmd.prompt if cmd.prompt else None,
        command_propose_instructions=cmd.propose_instructions,
    )
