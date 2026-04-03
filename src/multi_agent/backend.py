"""Backend protocol for agent execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from multi_agent.models import TokenUsage


@dataclass
class AgentResult:
    """Raw result from running an agent."""
    output: dict[str, Any] | None
    usage: TokenUsage
    duration_seconds: float
    error: str | None = None


@runtime_checkable
class AgentBackend(Protocol):
    """Interface for running an agent and getting a parsed result.

    Each backend encapsulates how to communicate with a particular LLM
    (subprocess, SDK, HTTP API, etc.).  The orchestration layer calls
    ``run_agent`` with backend-agnostic parameters and receives a uniform
    ``AgentResult``.
    """

    async def run_agent(
        self,
        agent_name: str,
        prompt: str,
        system_prompt: str,
        repo_root: str,
        timeout_seconds: int,
        *,
        model: str | None = None,
        max_turns: int = 0,
        allowed_tools: list[str] | None = None,
        on_progress: Callable[[str, str], None] | None = None,
        progress_label: str = "running",
        report_tool_use: bool = True,
    ) -> AgentResult: ...
