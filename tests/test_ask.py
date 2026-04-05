"""Tests for the ask command (temp-file consensus Q&A)."""

from unittest.mock import AsyncMock, patch

import pytest

from multi_agent.backend import AgentResult
from multi_agent.config import (
    AgentConfig,
    CommandConfig,
    GeneralConfig,
    MultiAgentConfig,
    load_config,
)
from multi_agent.models import TokenUsage


def _make_agent_result(output=None, error=None):
    return AgentResult(
        output=output or {},
        usage=TokenUsage(),
        duration_seconds=1.0,
        error=error,
    )


def _make_config(agents=None):
    if agents is None:
        agents = {
            "alpha": AgentConfig(system_prompt="Alpha."),
            "beta": AgentConfig(system_prompt="Beta."),
        }
    return MultiAgentConfig(
        general=GeneralConfig(),
        agents=agents,
    )


# --- DEFAULT_ASK_COMMAND in load_config ---


class TestDefaultAskCommand:
    def test_default_ask_command_inserted_no_file(self, tmp_path):
        config = load_config(search_from=tmp_path)
        assert "ask" in config.commands
        assert config.commands["ask"].prompt

    def test_default_ask_command_inserted_with_toml(self, tmp_path):
        toml = (
            '[agents.alpha]\nsystem_prompt = "A"\n'
            '[agents.beta]\nsystem_prompt = "B"\n'
        )
        p = tmp_path / "multi_agent.toml"
        p.write_text(toml)
        config = load_config(path=p)
        assert "ask" in config.commands

    def test_toml_ask_command_not_overridden(self, tmp_path):
        toml = (
            '[agents.alpha]\nsystem_prompt = "A"\n'
            '[agents.beta]\nsystem_prompt = "B"\n'
            '[commands.ask]\nprompt = "Custom ask prompt."\n'
        )
        p = tmp_path / "multi_agent.toml"
        p.write_text(toml)
        config = load_config(path=p)
        assert config.commands["ask"].prompt == "Custom ask prompt."


# --- _run_ask ---


class TestRunAsk:
    def test_question_and_answer_files_persist(self, tmp_path):
        """Both question and answer files remain after a successful run."""
        from multi_agent.cli import _run_ask

        config = _make_config()

        mock_result = type("R", (), {
            "merged_texts": {".multi_agent_ask_answer.md": "The answer."},
            "consensus_reached": True,
            "rounds": [type("Round", (), {"approvals": 2, "reviews": [1, 2]})()],
            "total_usage": TokenUsage(),
            "total_duration_seconds": 1.0,
            "best_approvals": 2,
            "best_round": 0,
            "stalled": False,
        })()

        with patch("multi_agent.cli._create_backend"), \
             patch("multi_agent.context.load_reference", return_value={}), \
             patch("multi_agent.context.count_uncommitted_reference", return_value=0), \
             patch("multi_agent.cli.asyncio") as mock_asyncio, \
             patch("multi_agent.cli.print_header"), \
             patch("multi_agent.cli.print_iteration_success"), \
             patch("multi_agent.cli.print_answer"), \
             patch("multi_agent.cli.print_token_usage"):
            mock_asyncio.run.return_value = mock_result

            _run_ask(config=config, repo_root=tmp_path, question="What is X?")

        assert (tmp_path / ".multi_agent_ask_question.md").exists()
        assert (tmp_path / ".multi_agent_ask_question.md").read_text() == "What is X?"
        assert (tmp_path / ".multi_agent_ask_answer.md").exists()
        assert (tmp_path / ".multi_agent_ask_answer.md").read_text() == "The answer."

    def test_answer_file_written_with_merged_text(self, tmp_path):
        """The answer file on disk contains the consensus answer, not the question."""
        from multi_agent.cli import _run_ask

        config = _make_config()

        mock_result = type("R", (), {
            "merged_texts": {".multi_agent_ask_answer.md": "The answer is 42."},
            "consensus_reached": True,
            "rounds": [type("Round", (), {"approvals": 2, "reviews": [1, 2]})()],
            "total_usage": TokenUsage(),
            "total_duration_seconds": 1.0,
            "best_approvals": 2,
            "best_round": 0,
            "stalled": False,
        })()

        with patch("multi_agent.cli._create_backend"), \
             patch("multi_agent.context.load_reference", return_value={}), \
             patch("multi_agent.context.count_uncommitted_reference", return_value=0), \
             patch("multi_agent.cli.asyncio") as mock_asyncio, \
             patch("multi_agent.cli.print_header"), \
             patch("multi_agent.cli.print_iteration_success"), \
             patch("multi_agent.cli.print_answer") as mock_print_answer, \
             patch("multi_agent.cli.print_token_usage"):
            mock_asyncio.run.return_value = mock_result

            exit_code = _run_ask(
                config=config, repo_root=tmp_path, question="What is the answer?",
            )

        assert exit_code == 0
        mock_print_answer.assert_called_once_with("The answer is 42.")

    def test_no_edits_leaves_question_in_both_files(self, tmp_path):
        """When agents propose no edits, question file persists, answer has question."""
        from multi_agent.cli import _run_ask

        config = _make_config()

        with patch("multi_agent.cli._create_backend"), \
             patch("multi_agent.context.load_reference", return_value={}), \
             patch("multi_agent.context.count_uncommitted_reference", return_value=0), \
             patch("multi_agent.cli.asyncio") as mock_asyncio, \
             patch("multi_agent.cli.print_header"), \
             patch("multi_agent.cli.print_no_edits"), \
             patch("multi_agent.cli.print_token_usage"):
            mock_result = type("R", (), {
                "merged_texts": {},
                "total_usage": TokenUsage(),
                "total_duration_seconds": 1.0,
            })()
            mock_asyncio.run.return_value = mock_result

            _run_ask(config=config, repo_root=tmp_path, question="What is X?")

        assert (tmp_path / ".multi_agent_ask_question.md").read_text() == "What is X?"
        # Answer file still has the question (no edits were applied)
        assert (tmp_path / ".multi_agent_ask_answer.md").read_text() == "What is X?"

    def test_question_written_to_answer_file_before_loop(self, tmp_path):
        """The question is in the answer file when the iteration loop reads it."""
        from multi_agent.cli import _run_ask

        config = _make_config()
        captured_contents = []

        def capture_run(coro):
            answer_file = tmp_path / ".multi_agent_ask_answer.md"
            if answer_file.exists():
                captured_contents.append(answer_file.read_text())
            return type("R", (), {
                "merged_texts": {},
                "total_usage": TokenUsage(),
                "total_duration_seconds": 1.0,
            })()

        with patch("multi_agent.cli._create_backend"), \
             patch("multi_agent.context.load_reference", return_value={}), \
             patch("multi_agent.context.count_uncommitted_reference", return_value=0), \
             patch("multi_agent.cli.asyncio") as mock_asyncio, \
             patch("multi_agent.cli.print_header"), \
             patch("multi_agent.cli.print_no_edits"), \
             patch("multi_agent.cli.print_token_usage"):
            mock_asyncio.run.side_effect = capture_run

            _run_ask(config=config, repo_root=tmp_path, question="My question?")

        assert captured_contents == ["My question?"]


# --- ask CLI command ---


class TestAskCli:
    def test_ask_command_exists(self):
        from multi_agent.cli import main

        runner = __import__("click.testing", fromlist=["CliRunner"]).CliRunner()
        result = runner.invoke(main, ["ask", "--help"])
        assert result.exit_code == 0
        assert "consensus answer" in result.output.lower()

    def test_empty_question_rejected(self):
        from multi_agent.cli import main

        runner = __import__("click.testing", fromlist=["CliRunner"]).CliRunner()
        # Invoke with no question argument
        result = runner.invoke(main, ["ask"])
        assert result.exit_code != 0
