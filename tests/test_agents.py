"""Tests for agent system prompt assembly and CLI argument building."""

from multi_agent.agents import (
    DISSENT_MODE_SUFFIX,
    EDIT_REVIEW_MODE_SUFFIX,
    build_agent_system_prompt,
    build_command_mode_suffix,
    build_name_normalizer,
)
from multi_agent.claude_runner import build_cli_args
from multi_agent.config import AgentConfig


# --- build_name_normalizer ---


class TestBuildNameNormalizer:
    def test_config_key_maps_to_itself(self, two_agent_config):
        normalize = build_name_normalizer(two_agent_config)
        assert normalize("alpha") == "alpha"
        assert normalize("beta") == "beta"

    def test_display_name_maps_to_config_key(self):
        agents = {"scientific_rigor": AgentConfig(system_prompt="Science.")}
        normalize = build_name_normalizer(agents)
        assert normalize("Scientific Rigor") == "scientific_rigor"

    def test_lowercased_display_name_maps_to_config_key(self):
        agents = {"scientific_rigor": AgentConfig(system_prompt="Science.")}
        normalize = build_name_normalizer(agents)
        assert normalize("scientific rigor") == "scientific_rigor"

    def test_underscore_as_space_maps_to_config_key(self):
        agents = {"scientific_rigor": AgentConfig(system_prompt="Science.")}
        normalize = build_name_normalizer(agents)
        assert normalize("scientific rigor") == "scientific_rigor"

    def test_explicit_display_name(self):
        agents = {"sci": AgentConfig(system_prompt="Science.", display_name="Dr. Science")}
        normalize = build_name_normalizer(agents)
        assert normalize("Dr. Science") == "sci"
        assert normalize("dr. science") == "sci"

    def test_unknown_name_passes_through(self, two_agent_config):
        normalize = build_name_normalizer(two_agent_config)
        assert normalize("unknown_agent") == "unknown_agent"

    def test_parenthetical_suffix_stripped(self):
        agents = {"sociopolitical": AgentConfig(system_prompt="Politics.")}
        normalize = build_name_normalizer(agents)
        assert normalize("Sociopolitical (politics)") == "sociopolitical"
        assert normalize("sociopolitical (review)") == "sociopolitical"

    def test_parenthetical_on_display_name(self):
        agents = {"sci": AgentConfig(system_prompt="S.", display_name="Scientific Rigor")}
        normalize = build_name_normalizer(agents)
        assert normalize("Scientific Rigor (accuracy)") == "sci"


# --- build_command_mode_suffix ---


class TestBuildCommandModeSuffix:
    def test_label_uppercased_with_spaces(self):
        result = build_command_mode_suffix("code-review", "Check the code.")
        assert "CODE REVIEW" in result

    def test_contains_prompt_text(self):
        result = build_command_mode_suffix("review", "Check for bugs.")
        assert "Check for bugs." in result

    def test_contains_json_instructions(self):
        result = build_command_mode_suffix("review", "Check.")
        assert "original_text" in result
        assert "replacement_text" in result


# --- build_agent_system_prompt ---


class TestBuildAgentSystemPrompt:
    def test_command_mode_appends_command_suffix(self):
        result = build_agent_system_prompt(
            "alpha", "command", "Base prompt.",
            command_name="lint", command_prompt="Lint the code.",
        )
        assert result.startswith("Base prompt.")
        assert "LINT" in result
        assert "Lint the code." in result

    def test_review_mode_appends_review_suffix(self):
        result = build_agent_system_prompt("alpha", "review", "Base prompt.")
        assert result == "Base prompt." + EDIT_REVIEW_MODE_SUFFIX

    def test_dissent_mode_appends_dissent_suffix(self):
        result = build_agent_system_prompt("alpha", "dissent", "Base prompt.")
        assert result == "Base prompt." + DISSENT_MODE_SUFFIX

    def test_unknown_mode_appends_nothing(self):
        result = build_agent_system_prompt("alpha", "unknown", "Base prompt.")
        assert result == "Base prompt."


# --- build_cli_args ---


class TestBuildCliArgs:
    def test_starts_with_claude_print(self):
        args = build_cli_args("alpha", "prompt", None, "/repo")
        assert args[0] == "claude"
        assert "--print" in args

    def test_no_max_turns_cli_flag(self):
        """Turn limits are enforced at the stream level, not via CLI flag."""
        args = build_cli_args("alpha", "prompt", None, "/repo")
        assert "--max-turns" not in args

    def test_model_adds_flag(self):
        args = build_cli_args("alpha", "prompt", "sonnet", "/repo")
        idx = args.index("--model")
        assert args[idx + 1] == "sonnet"

    def test_no_model_omits_flag(self):
        args = build_cli_args("alpha", "prompt", None, "/repo")
        assert "--model" not in args

    def test_allowed_tools_includes_read_by_default(self):
        args = build_cli_args("alpha", "prompt", None, "/repo")
        idx = args.index("--allowedTools")
        allowed = args[idx + 1]
        assert "Read" in allowed

    def test_allowed_tools_adds_specified_tools(self):
        args = build_cli_args("alpha", "prompt", None, "/repo", allowed_tools=["Bash", "Grep"])
        idx = args.index("--allowedTools")
        allowed = args[idx + 1].split(",")
        assert "Bash" in allowed
        assert "Grep" in allowed
        assert "Read" in allowed

    def test_allowed_tools_excluded_from_disallowed(self):
        args = build_cli_args("alpha", "prompt", None, "/repo", allowed_tools=["Bash"])
        idx = args.index("--disallowedTools")
        disallowed = args[idx + 1].split(",")
        assert "Bash" not in disallowed
        assert "Read" not in disallowed
