"""Parallel agent execution, vote tallying, and error handling."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from multi_agent.agents import AGENT_PROMPTS, build_cli_args
from multi_agent.config import MultiAgentConfig
from multi_agent.context import (
    build_file_review_prompt,
    build_review_prompt,
    get_staged_content,
    get_staged_diff,
    get_staged_files,
    load_canon,
)


@dataclass
class ReviewIssue:
    severity: str
    issue: str
    suggestion: str
    file: str = ""
    quote: str = ""


@dataclass
class AgentReview:
    agent_name: str
    verdict: str
    summary: str
    issues: list[ReviewIssue] = field(default_factory=list)
    duration_seconds: float = 0.0
    error: str | None = None


@dataclass
class ConsensusResult:
    approved: bool
    reviews: list[AgentReview]
    total_duration_seconds: float
    files_reviewed: list[str]


def _parse_issues(raw_issues: list[dict[str, Any]]) -> list[ReviewIssue]:
    """Parse issue dicts into ReviewIssue objects."""
    issues = []
    for item in raw_issues:
        issues.append(ReviewIssue(
            severity=item.get("severity", "minor"),
            issue=item.get("issue", ""),
            suggestion=item.get("suggestion", ""),
            file=item.get("file", ""),
            quote=item.get("quote", ""),
        ))
    return issues


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from text, handling markdown code fences."""
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    for marker in ("```json", "```"):
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

    # Try finding first { ... } block
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i + 1])
                    except json.JSONDecodeError:
                        break

    return None


async def _run_single_agent(
    agent_name: str,
    review_prompt: str,
    cli_args: list[str],
    repo_root: str,
    timeout_seconds: int,
    on_progress: Callable[[str, str], None] | None = None,
) -> AgentReview:
    """Run a single reviewer agent via the claude CLI."""
    start = time.monotonic()

    if on_progress:
        on_progress(agent_name, "starting")

    try:
        proc = await asyncio.wait_for(
            _spawn_claude(cli_args, review_prompt, repo_root),
            timeout=timeout_seconds,
        )

        if proc.returncode != 0:
            stderr_text = proc.stderr.strip() if proc.stderr else "unknown error"
            return AgentReview(
                agent_name=agent_name,
                verdict="REQUEST_CHANGES",
                summary=f"Agent process failed (exit {proc.returncode}).",
                error=stderr_text[:500],
                duration_seconds=time.monotonic() - start,
            )

        stdout = proc.stdout or ""

        # claude --print --output-format json returns JSON with a "result" field
        output = _extract_json(stdout)

        # If the outer JSON has a "result" field (claude CLI wrapper), unwrap it
        if output and "result" in output and "verdict" not in output:
            inner = output["result"]
            if isinstance(inner, str):
                output = _extract_json(inner)
            elif isinstance(inner, dict):
                output = inner

        if output is None:
            return AgentReview(
                agent_name=agent_name,
                verdict="REQUEST_CHANGES",
                summary="Could not parse agent response.",
                error=f"Unparseable output: {stdout[:300]}",
                duration_seconds=time.monotonic() - start,
            )

        verdict = output.get("verdict", "REQUEST_CHANGES")
        issues = _parse_issues(output.get("issues", []))

        # Safety: critical issues override an APPROVE verdict
        if any(i.severity == "critical" for i in issues) and verdict == "APPROVE":
            verdict = "REQUEST_CHANGES"

        if on_progress:
            on_progress(agent_name, f"done — {verdict}")

        return AgentReview(
            agent_name=agent_name,
            verdict=verdict,
            summary=output.get("summary", ""),
            issues=issues,
            duration_seconds=time.monotonic() - start,
        )

    except asyncio.TimeoutError:
        return AgentReview(
            agent_name=agent_name,
            verdict="REQUEST_CHANGES",
            summary="Agent timed out.",
            error=f"Timed out after {timeout_seconds}s",
            duration_seconds=time.monotonic() - start,
        )
    except Exception as exc:
        return AgentReview(
            agent_name=agent_name,
            verdict="REQUEST_CHANGES",
            summary=f"Agent failed: {exc}",
            error=str(exc),
            duration_seconds=time.monotonic() - start,
        )


async def _spawn_claude(
    cli_args: list[str],
    prompt: str,
    cwd: str,
) -> asyncio.subprocess.Process:
    """Spawn a claude CLI process and pipe the prompt via stdin."""
    proc = await asyncio.create_subprocess_exec(
        *cli_args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout_bytes, stderr_bytes = await proc.communicate(input=prompt.encode())
    proc.stdout = stdout_bytes.decode() if stdout_bytes else ""
    proc.stderr = stderr_bytes.decode() if stderr_bytes else ""
    return proc


async def run_consensus(
    config: MultiAgentConfig,
    repo_root: str,
    on_progress: Callable[[str, str], None] | None = None,
) -> ConsensusResult:
    """Run all enabled agents in parallel and tally votes."""
    root = Path(repo_root)
    start = time.monotonic()

    # Find staged text files
    staged_files = get_staged_files(root, config.general.file_patterns)
    if not staged_files:
        return ConsensusResult(
            approved=True,
            reviews=[],
            total_duration_seconds=0.0,
            files_reviewed=[],
        )

    # Build review context
    staged_diff = get_staged_diff(root, config.general.file_patterns)
    staged_contents = {}
    for f in staged_files:
        staged_contents[str(f)] = get_staged_content(root, f)

    canon = load_canon(
        root,
        config.general.canon_directories,
        config.general.file_patterns,
        config.general.max_canon_size_kb,
    )

    review_prompt = build_review_prompt(staged_diff, staged_contents, canon)

    # Build CLI args for each enabled agent
    enabled_agents: dict[str, list[str]] = {}
    for name, agent_cfg in config.agents.items():
        if not agent_cfg.enabled:
            continue
        system_prompt = agent_cfg.system_prompt_override or AGENT_PROMPTS[name]
        enabled_agents[name] = build_cli_args(
            name, system_prompt, agent_cfg.model, repo_root,
        )

    # Launch all agents in parallel
    reviews: list[AgentReview] = []

    async with asyncio.TaskGroup() as tg:
        tasks = {
            name: tg.create_task(
                _run_single_agent(
                    name, review_prompt, cli_args, repo_root,
                    config.general.timeout_seconds, on_progress,
                )
            )
            for name, cli_args in enabled_agents.items()
        }

    for name, task in tasks.items():
        reviews.append(task.result())

    # Tally votes
    approvals = sum(1 for r in reviews if r.verdict == "APPROVE")
    approved = approvals >= config.general.consensus_threshold

    return ConsensusResult(
        approved=approved,
        reviews=reviews,
        total_duration_seconds=time.monotonic() - start,
        files_reviewed=[str(f) for f in staged_files],
    )


def resolve_file_args(
    file_paths: list[str],
    repo_root: Path,
    config: MultiAgentConfig,
) -> list[str]:
    """Resolve file arguments to relative paths, expanding directories."""
    resolved: list[Path] = []
    for fp in file_paths:
        abs_path = Path(fp) if Path(fp).is_absolute() else repo_root / fp
        if abs_path.is_dir():
            for pattern in config.general.file_patterns:
                resolved.extend(sorted(abs_path.rglob(pattern)))
        elif abs_path.is_file():
            resolved.append(abs_path)
        else:
            raise FileNotFoundError(f"File not found: {fp}")

    if not resolved:
        raise FileNotFoundError(
            f"No files matching {config.general.file_patterns} found in: "
            + ", ".join(file_paths)
        )

    return [str(p.relative_to(repo_root)) for p in resolved]


async def run_file_review(
    config: MultiAgentConfig,
    repo_root: str,
    target_rel: list[str],
    on_progress: Callable[[str, str], None] | None = None,
) -> ConsensusResult:
    """Review existing files (already committed or on disk).

    target_rel should be relative paths (from resolve_file_args).
    """
    root = Path(repo_root)
    start = time.monotonic()

    # List canon files for cross-reference context
    canon = load_canon(
        root,
        config.general.canon_directories,
        config.general.file_patterns,
        config.general.max_canon_size_kb,
    )
    canon_files = list(canon.keys())

    review_prompt = build_file_review_prompt(target_rel, canon_files)

    # Build CLI args for each enabled agent
    enabled_agents: dict[str, list[str]] = {}
    for name, agent_cfg in config.agents.items():
        if not agent_cfg.enabled:
            continue
        system_prompt = agent_cfg.system_prompt_override or AGENT_PROMPTS[name]
        enabled_agents[name] = build_cli_args(
            name, system_prompt, agent_cfg.model, repo_root,
        )

    # Launch all agents in parallel
    reviews: list[AgentReview] = []

    async with asyncio.TaskGroup() as tg:
        tasks = {
            name: tg.create_task(
                _run_single_agent(
                    name, review_prompt, cli_args, repo_root,
                    config.general.timeout_seconds, on_progress,
                )
            )
            for name, cli_args in enabled_agents.items()
        }

    for name, task in tasks.items():
        reviews.append(task.result())

    approvals = sum(1 for r in reviews if r.verdict == "APPROVE")
    approved = approvals >= config.general.consensus_threshold

    return ConsensusResult(
        approved=approved,
        reviews=reviews,
        total_duration_seconds=time.monotonic() - start,
        files_reviewed=target_rel,
    )
