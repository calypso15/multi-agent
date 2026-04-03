"""Shared test fixtures."""

import pytest

from multi_agent.config import AgentConfig


@pytest.fixture
def two_agent_config():
    """Minimal AgentConfig dict with two enabled agents."""
    return {
        "alpha": AgentConfig(system_prompt="Alpha specialty."),
        "beta": AgentConfig(system_prompt="Beta specialty."),
    }
