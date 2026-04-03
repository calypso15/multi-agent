"""Tests for configuration loading from TOML files."""

import pytest

from multi_agent.config import (
    AgentConfig,
    CommandConfig,
    GeneralConfig,
    MultiAgentConfig,
    _field_names,
    _find_config_file,
    get_display_name,
    load_config,
)


# --- get_display_name ---


class TestGetDisplayName:
    def test_explicit_display_name(self):
        cfg = AgentConfig(system_prompt="x", display_name="Dr. Science")
        assert get_display_name("sci", cfg) == "Dr. Science"

    def test_none_falls_back_to_titlecased_key(self):
        cfg = AgentConfig(system_prompt="x")
        assert get_display_name("scientific_rigor", cfg) == "Scientific Rigor"

    def test_simple_key(self):
        cfg = AgentConfig(system_prompt="x")
        assert get_display_name("alpha", cfg) == "Alpha"


# --- _field_names ---


class TestFieldNames:
    def test_returns_agent_config_fields(self):
        names = _field_names(AgentConfig)
        assert "system_prompt" in names
        assert "display_name" in names
        assert "enabled" in names
        assert "allowed_tools" in names


# --- _find_config_file ---


class TestFindConfigFile:
    def test_finds_in_start_dir(self, tmp_path):
        (tmp_path / "multi_agent.toml").write_text("[general]\n")
        assert _find_config_file(tmp_path) == tmp_path / "multi_agent.toml"

    def test_finds_in_parent(self, tmp_path):
        child = tmp_path / "sub" / "deep"
        child.mkdir(parents=True)
        (tmp_path / "multi_agent.toml").write_text("[general]\n")
        assert _find_config_file(child) == tmp_path / "multi_agent.toml"

    def test_returns_none_when_not_found(self, tmp_path):
        assert _find_config_file(tmp_path) is None


# --- load_config ---


def _write_toml(tmp_path, content):
    """Write TOML content and return the path."""
    p = tmp_path / "multi_agent.toml"
    p.write_text(content)
    return p


def _minimal_toml(extra=""):
    """Return valid minimal TOML with two agents."""
    return (
        '[agents.alpha]\nsystem_prompt = "Alpha."\n'
        '[agents.beta]\nsystem_prompt = "Beta."\n'
        + extra
    )


class TestLoadConfig:
    def test_no_file_returns_defaults(self, tmp_path):
        config = load_config(search_from=tmp_path)
        assert isinstance(config, MultiAgentConfig)
        assert "review" in config.commands

    def test_valid_minimal_toml(self, tmp_path):
        p = _write_toml(tmp_path, _minimal_toml())
        config = load_config(path=p)
        assert "alpha" in config.agents
        assert "beta" in config.agents
        assert config.agents["alpha"].system_prompt == "Alpha."

    def test_unknown_general_key_raises(self, tmp_path):
        p = _write_toml(tmp_path, '[general]\nbogus = true\n' + _minimal_toml())
        with pytest.raises(ValueError, match="Unknown key.*general"):
            load_config(path=p)

    def test_unknown_agent_key_raises(self, tmp_path):
        toml = (
            '[agents.alpha]\nsystem_prompt = "A"\nbogus = true\n'
            '[agents.beta]\nsystem_prompt = "B"\n'
        )
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ValueError, match="Unknown key.*agents.alpha"):
            load_config(path=p)

    def test_missing_system_prompt_raises(self, tmp_path):
        toml = '[agents.alpha]\nsystem_prompt = "A"\n[agents.beta]\nenabled = true\n'
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ValueError, match="missing required.*system_prompt"):
            load_config(path=p)

    def test_empty_display_name_raises(self, tmp_path):
        toml = (
            '[agents.alpha]\nsystem_prompt = "A"\ndisplay_name = ""\n'
            '[agents.beta]\nsystem_prompt = "B"\n'
        )
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ValueError, match="empty.*display_name"):
            load_config(path=p)

    def test_consensus_threshold_exceeds_agents_raises(self, tmp_path):
        toml = '[general]\nconsensus_threshold = 5\n' + _minimal_toml()
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ValueError, match="consensus_threshold.*exceeds"):
            load_config(path=p)

    def test_fewer_than_two_agents_raises(self, tmp_path):
        toml = '[general]\nconsensus_threshold = 1\n[agents.alpha]\nsystem_prompt = "A"\n'
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ValueError, match="At least 2"):
            load_config(path=p)

    def test_max_rounds_less_than_one_raises(self, tmp_path):
        toml = '[general]\nmax_rounds = 0\n' + _minimal_toml()
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ValueError, match="max_rounds"):
            load_config(path=p)

    def test_invalid_severity_raises(self, tmp_path):
        toml = '[general]\nmin_severity = "extreme"\n' + _minimal_toml()
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ValueError, match="min_severity"):
            load_config(path=p)

    def test_command_references_unknown_agent_raises(self, tmp_path):
        toml = _minimal_toml(
            '[commands.lint]\nprompt = "Lint."\nagents = ["alpha", "ghost"]\n'
        )
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ValueError, match="unknown agent"):
            load_config(path=p)

    def test_both_tasks_and_commands_raises(self, tmp_path):
        toml = (
            _minimal_toml()
            + '[tasks.old]\nprompt = "Old."\n'
            + '[commands.new]\nprompt = "New."\n'
        )
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ValueError, match="tasks.*commands"):
            load_config(path=p)

    def test_default_review_command_inserted(self, tmp_path):
        p = _write_toml(tmp_path, _minimal_toml())
        config = load_config(path=p)
        assert "review" in config.commands
        assert config.commands["review"].prompt

    def test_invalid_backend_raises(self, tmp_path):
        toml = '[general]\nbackend = "openai"\n' + _minimal_toml()
        p = _write_toml(tmp_path, toml)
        with pytest.raises(ValueError, match="backend"):
            load_config(path=p)

    def test_valid_backend_accepted(self, tmp_path):
        toml = '[general]\nbackend = "claude-cli"\n' + _minimal_toml()
        p = _write_toml(tmp_path, toml)
        config = load_config(path=p)
        assert config.general.backend == "claude-cli"

    def test_default_backend(self, tmp_path):
        p = _write_toml(tmp_path, _minimal_toml())
        config = load_config(path=p)
        assert config.general.backend == "claude-cli"
