"""Parallel agent execution, vote tallying, and error handling."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from multi_agent.agents import (
    AGENT_PROMPTS,
    ARBITRATOR_PROMPT,
    build_agent_system_prompt,
    build_cli_args,
    normalize_agent_name,
)
from multi_agent.config import MultiAgentConfig
from multi_agent.context import (
    build_file_review_prompt,
    build_propose_prompt,
    build_review_prompt,
    build_review_round_prompt,
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


# --- Dataclasses for the propose-review-iterate loop ---


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float = 0.0

    def __iadd__(self, other: TokenUsage) -> TokenUsage:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cost_usd += other.cost_usd
        return self


def _extract_usage(outer: dict[str, Any]) -> TokenUsage:
    """Extract token usage from the claude CLI JSON envelope."""
    usage = outer.get("usage", {})
    return TokenUsage(
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
        cost_usd=outer.get("total_cost_usd", 0.0),
    )


@dataclass
class FileEdit:
    file: str
    original_text: str
    replacement_text: str
    rationale: str


@dataclass
class AgentProposal:
    agent_name: str
    edits: list[FileEdit]
    summary: str
    duration_seconds: float = 0.0
    error: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class ProposalReview:
    original_agent: str
    edit_index: int
    verdict: str  # "APPROVE" or "MODIFY"
    modified_replacement: str | None = None
    rationale: str = ""


@dataclass
class AgentReviewResponse:
    agent_name: str
    all_approved: bool
    proposal_reviews: list[ProposalReview]
    summary: str
    duration_seconds: float = 0.0
    error: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class IterationRound:
    round_number: int
    reviews: list[AgentReviewResponse]
    consensus_reached: bool


@dataclass
class Dissent:
    agent_name: str
    opinion: str
    duration_seconds: float = 0.0
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class IterationResult:
    consensus_reached: bool
    final_edits: list[FileEdit]
    proposals: list[AgentProposal]
    rounds: list[IterationRound]
    total_duration_seconds: float
    files_reviewed: list[str]
    total_usage: TokenUsage = field(default_factory=TokenUsage)
    dissents: list[Dissent] = field(default_factory=list)
    best_round: int = -1
    best_approvals: int = 0
    stalled: bool = False


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


def _unwrap_result(result_json: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract the agent's JSON response from the claude CLI result envelope.

    The stream-json result event has a "result" field containing the agent's
    text output, which itself should be JSON.
    """
    if result_json is None:
        return None

    # The result_json is the final "type": "result" event
    inner = result_json.get("result")
    if inner is None:
        return None

    if isinstance(inner, dict):
        return inner

    if isinstance(inner, str):
        return _extract_json(inner)

    return None


def _make_tool_callback(
    agent_name: str,
    on_progress: Callable[[str, str], None] | None,
) -> Callable[[str, str], None] | None:
    """Wrap on_progress to format tool-use events for an agent."""
    if not on_progress:
        return None

    def callback(tool_name: str, summary: str):
        detail = f" {summary}" if summary else ""
        on_progress(agent_name, f"  → {tool_name}{detail}")

    return callback


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
        result = await asyncio.wait_for(
            _spawn_claude(
                cli_args, review_prompt, repo_root,
                on_tool_use=_make_tool_callback(agent_name, on_progress),
            ),
            timeout=timeout_seconds,
        )

        if result.returncode != 0:
            stderr_text = result.stderr.strip() if result.stderr else "unknown error"
            return AgentReview(
                agent_name=agent_name,
                verdict="REQUEST_CHANGES",
                summary=f"Agent process failed (exit {result.returncode}).",
                error=stderr_text[:500],
                duration_seconds=time.monotonic() - start,
            )

        output = _unwrap_result(result.result_json)

        if output is None:
            return AgentReview(
                agent_name=agent_name,
                verdict="REQUEST_CHANGES",
                summary="Could not parse agent response.",
                error=f"Unparseable output: {result.stdout[:300]}",
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


@dataclass
class ClaudeResult:
    """Parsed result from a claude CLI invocation."""
    returncode: int
    result_json: dict[str, Any] | None
    stdout: str
    stderr: str


async def _spawn_claude(
    cli_args: list[str],
    prompt: str,
    cwd: str,
    on_tool_use: Callable[[str, str], None] | None = None,
) -> ClaudeResult:
    """Spawn a claude CLI process, stream events, and return the result.

    With --output-format stream-json, reads events line by line.
    Reports tool_use events via on_tool_use(tool_name, summary).
    Returns the final result event's JSON.
    """
    proc = await asyncio.create_subprocess_exec(
        *cli_args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )

    # Send prompt and close stdin
    proc.stdin.write(prompt.encode())
    await proc.stdin.drain()
    proc.stdin.close()

    result_json: dict[str, Any] | None = None
    all_lines: list[str] = []

    # Read stdout line by line for streaming events
    while True:
        line_bytes = await proc.stdout.readline()
        if not line_bytes:
            break
        line = line_bytes.decode().strip()
        if not line:
            continue
        all_lines.append(line)

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")

        if event_type == "assistant" and on_tool_use:
            content = event.get("message", {}).get("content", [])
            for item in content:
                if item.get("type") == "tool_use":
                    tool_name = item.get("name", "unknown")
                    tool_input = item.get("input", {})
                    # Build a short summary of the tool call
                    summary = _summarize_tool_call(tool_name, tool_input)
                    on_tool_use(tool_name, summary)

        elif event_type == "result":
            result_json = event

    stderr_bytes = await proc.stderr.read()
    await proc.wait()

    return ClaudeResult(
        returncode=proc.returncode,
        result_json=result_json,
        stdout="\n".join(all_lines),
        stderr=stderr_bytes.decode() if stderr_bytes else "",
    )


def _summarize_tool_call(tool_name: str, tool_input: dict) -> str:
    """Build a short human-readable summary of a tool call."""
    if tool_name in ("Read", "read_file"):
        path = tool_input.get("file_path", tool_input.get("path", ""))
        return path.split("/")[-1] if "/" in str(path) else str(path)
    if tool_name in ("WebSearch", "web_search"):
        query = tool_input.get("query", tool_input.get("search_query", ""))
        return f'"{query}"' if query else ""
    if tool_name in ("WebFetch", "web_fetch"):
        url = tool_input.get("url", "")
        return url[:60] if url else ""
    if tool_name in ("Grep", "grep"):
        pattern = tool_input.get("pattern", "")
        return f'"{pattern}"' if pattern else ""
    if tool_name in ("Glob", "glob"):
        pattern = tool_input.get("pattern", "")
        return pattern
    # Fallback: show first string value
    for v in tool_input.values():
        if isinstance(v, str) and v:
            return v[:40]
    return ""


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
            name, system_prompt, agent_cfg.propose_model, repo_root,
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
            name, system_prompt, agent_cfg.propose_model, repo_root,
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


# --- Propose-Review-Iterate loop ---


def _parse_edits(raw_edits: list[dict[str, Any]]) -> list[FileEdit]:
    """Parse edit dicts into FileEdit objects."""
    edits = []
    for item in raw_edits:
        edits.append(FileEdit(
            file=item.get("file", ""),
            original_text=item.get("original_text", ""),
            replacement_text=item.get("replacement_text", ""),
            rationale=item.get("rationale", ""),
        ))
    return edits


def _parse_proposal_reviews(raw: list[dict[str, Any]]) -> list[ProposalReview]:
    """Parse proposal review dicts into ProposalReview objects."""
    reviews = []
    for item in raw:
        reviews.append(ProposalReview(
            original_agent=normalize_agent_name(item.get("original_agent", "")),
            edit_index=item.get("edit_index", 0),
            verdict=item.get("verdict", "APPROVE"),
            modified_replacement=item.get("modified_replacement"),
            rationale=item.get("rationale", ""),
        ))
    return reviews


async def _run_single_proposer(
    agent_name: str,
    propose_prompt: str,
    cli_args: list[str],
    repo_root: str,
    timeout_seconds: int,
    on_progress: Callable[[str, str], None] | None = None,
) -> AgentProposal:
    """Run a single agent in propose mode."""
    start = time.monotonic()

    if on_progress:
        on_progress(agent_name, "proposing")

    try:
        result = await asyncio.wait_for(
            _spawn_claude(
                cli_args, propose_prompt, repo_root,
                on_tool_use=_make_tool_callback(agent_name, on_progress),
            ),
            timeout=timeout_seconds,
        )

        usage = _extract_usage(result.result_json) if result.result_json else TokenUsage()

        if result.returncode != 0:
            stderr_text = result.stderr.strip() if result.stderr else "unknown error"
            return AgentProposal(
                agent_name=agent_name,
                edits=[],
                summary=f"Agent process failed (exit {result.returncode}).",
                error=stderr_text[:500],
                duration_seconds=time.monotonic() - start,
                usage=usage,
            )

        output = _unwrap_result(result.result_json)

        if output is None:
            return AgentProposal(
                agent_name=agent_name,
                edits=[],
                summary="Could not parse agent response.",
                error=f"Unparseable output: {result.stdout[:300]}",
                duration_seconds=time.monotonic() - start,
                usage=usage,
            )

        edits = _parse_edits(output.get("edits", []))
        edit_count = len(edits)

        if on_progress:
            on_progress(agent_name, f"done — {edit_count} edit(s)")

        return AgentProposal(
            agent_name=agent_name,
            edits=edits,
            summary=output.get("summary", ""),
            duration_seconds=time.monotonic() - start,
            usage=usage,
        )

    except asyncio.TimeoutError:
        return AgentProposal(
            agent_name=agent_name,
            edits=[],
            summary="Agent timed out.",
            error=f"Timed out after {timeout_seconds}s",
            duration_seconds=time.monotonic() - start,
        )
    except Exception as exc:
        return AgentProposal(
            agent_name=agent_name,
            edits=[],
            summary=f"Agent failed: {exc}",
            error=str(exc),
            duration_seconds=time.monotonic() - start,
        )


async def _run_single_reviewer(
    agent_name: str,
    review_prompt: str,
    cli_args: list[str],
    repo_root: str,
    timeout_seconds: int,
    round_number: int,
    on_progress: Callable[[str, str], None] | None = None,
) -> AgentReviewResponse:
    """Run a single agent in review mode."""
    start = time.monotonic()

    if on_progress:
        on_progress(agent_name, f"reviewing (round {round_number + 1})")

    try:
        result = await asyncio.wait_for(
            _spawn_claude(
                cli_args, review_prompt, repo_root,
                on_tool_use=_make_tool_callback(agent_name, on_progress),
            ),
            timeout=timeout_seconds,
        )

        usage = _extract_usage(result.result_json) if result.result_json else TokenUsage()

        if result.returncode != 0:
            stderr_text = result.stderr.strip() if result.stderr else "unknown error"
            return AgentReviewResponse(
                agent_name=agent_name,
                all_approved=True,  # failed agent doesn't block
                proposal_reviews=[],
                summary=f"Agent process failed (exit {result.returncode}).",
                error=stderr_text[:500],
                duration_seconds=time.monotonic() - start,
            )

        output = _unwrap_result(result.result_json)

        if output is None:
            return AgentReviewResponse(
                agent_name=agent_name,
                all_approved=True,  # can't parse → don't block
                proposal_reviews=[],
                summary="Could not parse agent response.",
                error=f"Unparseable output: {result.stdout[:300]}",
                duration_seconds=time.monotonic() - start,
                usage=usage,
            )

        all_approved = output.get("all_approved", True)
        proposal_reviews = _parse_proposal_reviews(
            output.get("proposal_reviews", [])
        )

        if on_progress:
            if all_approved:
                on_progress(agent_name, f"done — approved all (round {round_number + 1})")
            else:
                mod_count = sum(
                    1 for r in proposal_reviews if r.verdict == "MODIFY"
                )
                on_progress(
                    agent_name,
                    f"done — modified {mod_count} (round {round_number + 1})",
                )

        return AgentReviewResponse(
            agent_name=agent_name,
            all_approved=all_approved,
            proposal_reviews=proposal_reviews,
            summary=output.get("summary", ""),
            duration_seconds=time.monotonic() - start,
            usage=usage,
        )

    except asyncio.TimeoutError:
        return AgentReviewResponse(
            agent_name=agent_name,
            all_approved=True,  # timed out → don't block
            proposal_reviews=[],
            summary="Agent timed out.",
            error=f"Timed out after {timeout_seconds}s",
            duration_seconds=time.monotonic() - start,
        )
    except Exception as exc:
        return AgentReviewResponse(
            agent_name=agent_name,
            all_approved=True,
            proposal_reviews=[],
            summary=f"Agent failed: {exc}",
            error=str(exc),
            duration_seconds=time.monotonic() - start,
        )


def validate_edits(
    edits: list[FileEdit],
    file_contents: dict[str, str],
) -> list[FileEdit]:
    """Drop edits whose original_text is not found in the file."""
    valid = []
    for edit in edits:
        content = file_contents.get(edit.file, "")
        if edit.original_text and edit.original_text in content:
            valid.append(edit)
    return valid


def deduplicate_edits(edits: list[FileEdit]) -> list[FileEdit]:
    """Remove duplicate and conflicting edits.

    When multiple edits target the same (file, original_text) with different
    replacements, only the first one is kept. Exact duplicates are also removed.
    """
    seen: set[tuple[str, str]] = set()
    result = []
    for edit in edits:
        key = (edit.file, edit.original_text)
        if key not in seen:
            seen.add(key)
            result.append(edit)
    return result


def merge_proposals(
    proposals: list[AgentProposal],
    reviews: list[AgentReviewResponse],
) -> list[AgentProposal]:
    """Apply review modifications to proposals.

    For each MODIFY review, update the corresponding edit's replacement_text.
    If multiple reviewers modify the same edit, the last one wins (next review
    round can resolve disagreements).
    """
    # Build a lookup: (agent_name, edit_index) → new replacement_text
    modifications: dict[tuple[str, int], str] = {}
    for review in reviews:
        for pr in review.proposal_reviews:
            if pr.verdict == "MODIFY" and pr.modified_replacement is not None:
                modifications[(pr.original_agent, pr.edit_index)] = (
                    pr.modified_replacement
                )

    if not modifications:
        return proposals

    updated = []
    for proposal in proposals:
        new_edits = []
        for i, edit in enumerate(proposal.edits):
            key = (proposal.agent_name, i)
            if key in modifications:
                new_edits.append(FileEdit(
                    file=edit.file,
                    original_text=edit.original_text,
                    replacement_text=modifications[key],
                    rationale=edit.rationale,
                ))
            else:
                new_edits.append(edit)
        updated.append(AgentProposal(
            agent_name=proposal.agent_name,
            edits=new_edits,
            summary=proposal.summary,
            duration_seconds=proposal.duration_seconds,
            error=proposal.error,
        ))

    return updated


async def run_propose_phase(
    config: MultiAgentConfig,
    file_contents: dict[str, str],
    canon: dict[str, str],
    staged_diff: str | None,
    repo_root: str,
    task: str | None = None,
    custom_task_prompt: str | None = None,
    on_progress: Callable[[str, str], None] | None = None,
) -> list[AgentProposal]:
    """Run all enabled agents in propose mode (parallel)."""
    propose_prompt = build_propose_prompt(
        file_contents, canon, staged_diff,
        min_severity=config.general.min_severity,
        task=task,
    )

    # Determine the agent mode for the system prompt
    mode = task if task in ("expand", "contract", "custom") else "propose"

    proposals: list[AgentProposal] = []

    async with asyncio.TaskGroup() as tg:
        tasks = {}
        for name, agent_cfg in config.agents.items():
            if not agent_cfg.enabled:
                continue
            system_prompt = build_agent_system_prompt(
                name, mode, agent_cfg.system_prompt_override,
                custom_task_prompt=custom_task_prompt,
            )
            max_turns = agent_cfg.propose_max_turns or config.general.propose_max_turns
            cli_args = build_cli_args(
                name, system_prompt, agent_cfg.propose_model, repo_root,
                max_turns=max_turns,
                allowed_tools=agent_cfg.allowed_tools or None,
            )
            tasks[name] = tg.create_task(
                _run_single_proposer(
                    name, propose_prompt, cli_args, repo_root,
                    config.general.timeout_seconds, on_progress,
                )
            )

    for name, task_obj in tasks.items():
        proposals.append(task_obj.result())

    return proposals


async def run_review_phase(
    config: MultiAgentConfig,
    proposals: list[AgentProposal],
    file_contents: dict[str, str],
    canon: dict[str, str],
    repo_root: str,
    round_number: int,
    on_progress: Callable[[str, str], None] | None = None,
) -> list[AgentReviewResponse]:
    """Run all enabled agents in review mode (parallel).

    Each agent only reviews proposals from OTHER agents — not its own.
    """
    reviews: list[AgentReviewResponse] = []

    async with asyncio.TaskGroup() as tg:
        tasks = {}
        for name, agent_cfg in config.agents.items():
            if not agent_cfg.enabled:
                continue
            # Filter out this agent's own proposals
            other_proposals = [
                p for p in proposals if p.agent_name != name
            ]
            # If no other proposals to review, auto-approve
            if not any(p.edits for p in other_proposals):
                reviews.append(AgentReviewResponse(
                    agent_name=name,
                    all_approved=True,
                    proposal_reviews=[],
                    summary="No proposals from other agents to review.",
                ))
                continue

            review_prompt = build_review_round_prompt(
                other_proposals, file_contents, canon, round_number,
            )
            system_prompt = build_agent_system_prompt(
                name, "review", agent_cfg.system_prompt_override,
            )
            model = agent_cfg.review_model or agent_cfg.propose_model
            max_turns = agent_cfg.review_max_turns or config.general.review_max_turns
            cli_args = build_cli_args(
                name, system_prompt, model, repo_root,
                max_turns=max_turns,
            )
            tasks[name] = tg.create_task(
                _run_single_reviewer(
                    name, review_prompt, cli_args, repo_root,
                    config.general.timeout_seconds, round_number, on_progress,
                )
            )

    for name, task in tasks.items():
        reviews.append(task.result())

    return reviews


def _build_dissent_prompt(
    proposals: list[AgentProposal],
    file_contents: dict[str, str],
) -> str:
    """Build a prompt for dissenting agents to state their concerns."""
    parts = ["# PROPOSED CHANGES (going to the user despite your objections)\n"]

    for proposal in proposals:
        if not proposal.edits:
            continue
        parts.append(f"\n## Edits from {proposal.agent_name}\n")
        for i, edit in enumerate(proposal.edits):
            parts.append(f"\n### Edit {i} — {edit.file}\n")
            parts.append(f"**Replace:**\n```\n{edit.original_text}\n```\n")
            parts.append(f"**With:**\n```\n{edit.replacement_text}\n```\n")

    parts.append(
        "\n# YOUR TASK\n"
        "State your most important concern about these changes in 2-4 sentences. "
        "Return your response as JSON with a single \"opinion\" field.\n"
    )
    return "".join(parts)


async def _run_single_dissenter(
    agent_name: str,
    prompt: str,
    cli_args: list[str],
    repo_root: str,
    timeout_seconds: int,
    on_progress: Callable[[str, str], None] | None = None,
) -> Dissent:
    """Run a single agent to collect a dissenting opinion."""
    start = time.monotonic()

    if on_progress:
        on_progress(agent_name, "dissenting")

    try:
        result = await asyncio.wait_for(
            _spawn_claude(cli_args, prompt, repo_root),
            timeout=timeout_seconds,
        )

        usage = _extract_usage(result.result_json) if result.result_json else TokenUsage()
        output = _unwrap_result(result.result_json)

        opinion = ""
        if output:
            opinion = output.get("opinion", "")
        if not opinion and result.result_json:
            # Fall back to raw result text
            raw = result.result_json.get("result", "")
            if isinstance(raw, str):
                opinion = raw.strip()[:500]

        if on_progress:
            on_progress(agent_name, "done — dissent recorded")

        return Dissent(
            agent_name=agent_name,
            opinion=opinion,
            duration_seconds=time.monotonic() - start,
            usage=usage,
        )

    except (asyncio.TimeoutError, Exception):
        return Dissent(
            agent_name=agent_name,
            opinion="",
            duration_seconds=time.monotonic() - start,
        )


async def _collect_dissents(
    config: MultiAgentConfig,
    dissenting_agents: list[str],
    proposals: list[AgentProposal],
    file_contents: dict[str, str],
    repo_root: str,
    on_progress: Callable[[str, str], None] | None = None,
) -> list[Dissent]:
    """Collect brief dissenting opinions from agents that didn't approve."""
    prompt = _build_dissent_prompt(proposals, file_contents)

    dissents: list[Dissent] = []

    async with asyncio.TaskGroup() as tg:
        tasks = {}
        for name in dissenting_agents:
            agent_cfg = config.agents.get(name)
            if not agent_cfg or not agent_cfg.enabled:
                continue
            system_prompt = build_agent_system_prompt(name, "dissent")
            model = agent_cfg.review_model or agent_cfg.propose_model
            cli_args = build_cli_args(
                name, system_prompt, model, repo_root,
                max_turns=1,
            )
            tasks[name] = tg.create_task(
                _run_single_dissenter(
                    name, prompt, cli_args, repo_root,
                    config.general.timeout_seconds, on_progress,
                )
            )

    for name, task_obj in tasks.items():
        result = task_obj.result()
        if result.opinion:
            dissents.append(result)

    return dissents


@dataclass
class ContestedEdit:
    """An edit that two agents keep modifying back and forth."""
    file: str
    original_text: str
    versions: dict[str, str]  # agent_name → replacement_text
    rationales: dict[str, str]  # agent_name → rationale


@dataclass
class ArbitrationResult:
    file: str
    original_text: str
    replacement_text: str
    rationale: str
    usage: TokenUsage = field(default_factory=TokenUsage)


def _detect_stall(
    rounds: list[IterationRound],
    best_approvals: int,
) -> bool:
    """Detect if the review loop has stalled (no improvement for 2 rounds)."""
    if len(rounds) < 2:
        return False
    recent = rounds[-2:]
    for rnd in recent:
        approvals = sum(
            1 for r in rnd.reviews if r.all_approved and r.error is None
        )
        if approvals > best_approvals:
            return False
    return True


def _find_contested_edits(
    rounds: list[IterationRound],
    current_proposals: list[AgentProposal],
) -> list[ContestedEdit]:
    """Find edits that have been modified in 2+ consecutive rounds."""
    if len(rounds) < 2:
        return []

    # Track which (original_agent, edit_index) pairs were modified each round
    modified_per_round: list[set[tuple[str, int]]] = []
    for rnd in rounds:
        modified = set()
        for review in rnd.reviews:
            for pr in review.proposal_reviews:
                if pr.verdict == "MODIFY":
                    modified.add((pr.original_agent, pr.edit_index))
        modified_per_round.append(modified)

    # Find edits modified in the last 2 rounds
    if len(modified_per_round) < 2:
        return []
    contested_keys = modified_per_round[-1] & modified_per_round[-2]

    if not contested_keys:
        return []

    # Build contested edit objects with both versions
    contested = []
    last_reviews = rounds[-1].reviews
    prev_reviews = rounds[-2].reviews

    for agent_name, edit_idx in contested_keys:
        # Find the edit in current proposals
        proposal = next(
            (p for p in current_proposals if p.agent_name == agent_name),
            None,
        )
        if not proposal or edit_idx >= len(proposal.edits):
            continue

        edit = proposal.edits[edit_idx]
        versions: dict[str, str] = {}
        rationales: dict[str, str] = {}

        # Collect versions from the last two rounds' modifiers
        for rnd_reviews in (prev_reviews, last_reviews):
            for review in rnd_reviews:
                for pr in review.proposal_reviews:
                    if (pr.original_agent == agent_name
                            and pr.edit_index == edit_idx
                            and pr.verdict == "MODIFY"
                            and pr.modified_replacement):
                        versions[review.agent_name] = pr.modified_replacement
                        rationales[review.agent_name] = pr.rationale

        # Also include the current version
        versions[agent_name] = edit.replacement_text
        rationales.setdefault(agent_name, edit.rationale)

        if len(versions) >= 2:
            contested.append(ContestedEdit(
                file=edit.file,
                original_text=edit.original_text,
                versions=versions,
                rationales=rationales,
            ))

    return contested


def _build_arbitration_prompt(
    contested: ContestedEdit,
    file_contents: dict[str, str],
) -> str:
    """Build the prompt for an arbitration call."""
    parts = ["# CONTESTED EDIT\n\n"]
    parts.append(f"**File:** {contested.file}\n\n")
    parts.append(f"**Original text being replaced:**\n```\n{contested.original_text}\n```\n\n")

    parts.append("## Competing Versions\n\n")
    for agent_name, replacement in contested.versions.items():
        rationale = contested.rationales.get(agent_name, "")
        parts.append(f"### Version from {agent_name}\n")
        parts.append(f"```\n{replacement}\n```\n")
        if rationale:
            parts.append(f"**Rationale:** {rationale}\n\n")

    parts.append(
        "# YOUR TASK\n"
        "Pick the better version or merge them. Return JSON with "
        "\"replacement_text\" and \"rationale\".\n"
    )
    return "".join(parts)


async def _run_arbitration(
    contested_edits: list[ContestedEdit],
    file_contents: dict[str, str],
    config: MultiAgentConfig,
    repo_root: str,
    on_progress: Callable[[str, str], None] | None = None,
) -> list[ArbitrationResult]:
    """Run arbitration on contested edits."""
    results: list[ArbitrationResult] = []

    # Use the review_model of the first enabled agent, or fall back to default
    model = None
    for agent_cfg in config.agents.values():
        if agent_cfg.enabled:
            model = agent_cfg.review_model or agent_cfg.propose_model
            break

    async with asyncio.TaskGroup() as tg:
        tasks_list = []
        for i, contested in enumerate(contested_edits):
            prompt = _build_arbitration_prompt(contested, file_contents)
            cli_args = build_cli_args(
                f"arbitrator_{i}",
                ARBITRATOR_PROMPT,
                model,
                repo_root,
                max_turns=1,
            )

            async def _run(p=prompt, args=cli_args, ce=contested):
                if on_progress:
                    on_progress("arbitrator", f"resolving conflict in {ce.file}")
                arb_start = time.monotonic()
                arb_result = await asyncio.wait_for(
                    _spawn_claude(args, p, repo_root),
                    timeout=config.general.timeout_seconds,
                )
                usage = _extract_usage(arb_result.result_json) if arb_result.result_json else TokenUsage()
                output = _unwrap_result(arb_result.result_json)

                replacement = ""
                rationale = ""
                if output:
                    replacement = output.get("replacement_text", "")
                    rationale = output.get("rationale", "")

                if on_progress:
                    on_progress("arbitrator", "done — conflict resolved")

                return ArbitrationResult(
                    file=ce.file,
                    original_text=ce.original_text,
                    replacement_text=replacement or list(ce.versions.values())[0],
                    rationale=rationale,
                    usage=usage,
                )

            tasks_list.append(tg.create_task(_run()))

    for t in tasks_list:
        results.append(t.result())

    return results


async def run_iteration_loop(
    config: MultiAgentConfig,
    repo_root: str,
    target_files: list[str] | None = None,
    task: str | None = None,
    custom_task_prompt: str | None = None,
    on_progress: Callable[[str, str], None] | None = None,
    on_phase: Callable | None = None,
) -> IterationResult:
    """Run the full propose-review-iterate loop.

    If target_files is None, reviews staged files. Otherwise reviews the
    specified files from the working tree.
    task: "expand", "contract", "custom", or None (default review).
    custom_task_prompt: prompt text for custom tasks.
    on_phase: callback for phase transitions. Called with:
      - ("propose_done", proposals)
      - ("review_done", round_number, reviews, consensus_threshold)
    """
    root = Path(repo_root)
    start = time.monotonic()

    # 1. Load file contents
    staged_diff: str | None = None
    if target_files is None:
        staged_files = get_staged_files(root, config.general.file_patterns)
        if not staged_files:
            return IterationResult(
                consensus_reached=True,
                final_edits=[],
                proposals=[],
                rounds=[],
                total_duration_seconds=0.0,
                files_reviewed=[],
            )
        file_contents = {}
        for f in staged_files:
            file_contents[str(f)] = get_staged_content(root, f)
        staged_diff = get_staged_diff(root, config.general.file_patterns)
        files_reviewed = [str(f) for f in staged_files]
    else:
        file_contents = {}
        for f in target_files:
            file_contents[f] = (root / f).read_text()
        files_reviewed = list(target_files)

    # 2. Load canon
    canon = load_canon(
        root,
        config.general.canon_directories,
        config.general.file_patterns,
        config.general.max_canon_size_kb,
    )

    # 3. Propose phase
    total_usage = TokenUsage()

    proposals = await run_propose_phase(
        config, file_contents, canon, staged_diff, repo_root,
        task=task, custom_task_prompt=custom_task_prompt,
        on_progress=on_progress,
    )
    for p in proposals:
        total_usage += p.usage

    if on_phase:
        on_phase("propose_done", proposals)

    # 4. Validate and deduplicate
    for proposal in proposals:
        proposal.edits = validate_edits(proposal.edits, file_contents)

    all_edits = []
    for p in proposals:
        all_edits.extend(p.edits)
    all_edits = deduplicate_edits(all_edits)

    # 5. If no edits, content is fine
    if not all_edits:
        return IterationResult(
            consensus_reached=True,
            final_edits=[],
            proposals=proposals,
            rounds=[],
            total_duration_seconds=time.monotonic() - start,
            files_reviewed=files_reviewed,
            total_usage=total_usage,
        )

    # 6. Review-iterate loop
    rounds: list[IterationRound] = []
    current_proposals = proposals
    consensus = False
    stalled = False

    # Track the best (highest approval) proposals seen
    best_proposals = current_proposals
    best_approvals = 0
    best_round = -1

    for round_num in range(config.general.max_rounds):
        reviews = await run_review_phase(
            config, current_proposals, file_contents, canon,
            repo_root, round_num, on_progress,
        )
        for r in reviews:
            total_usage += r.usage

        approvals = sum(
            1 for r in reviews if r.all_approved and r.error is None
        )
        consensus_reached = approvals >= config.general.consensus_threshold

        rounds.append(IterationRound(
            round_number=round_num,
            reviews=reviews,
            consensus_reached=consensus_reached,
        ))

        if on_phase:
            on_phase("review_done", round_num, reviews,
                     config.general.consensus_threshold)

        # Track the version with the highest approval count
        if approvals >= best_approvals:
            best_approvals = approvals
            best_proposals = current_proposals
            best_round = round_num

        if consensus_reached:
            consensus = True
            break

        # Detect stall — no improvement for 2 consecutive rounds
        if _detect_stall(rounds, best_approvals):
            contested = _find_contested_edits(
                rounds, current_proposals,
            )
            if contested:
                if on_phase:
                    on_phase("arbitration_start", contested)

                arb_results = await _run_arbitration(
                    contested, file_contents, config, repo_root,
                    on_progress,
                )
                for ar in arb_results:
                    total_usage += ar.usage

                # Apply arbitration results to current proposals
                arb_lookup: dict[tuple[str, str], str] = {
                    (ar.file, ar.original_text): ar.replacement_text
                    for ar in arb_results
                }
                for proposal in current_proposals:
                    for i, edit in enumerate(proposal.edits):
                        key = (edit.file, edit.original_text)
                        if key in arb_lookup:
                            proposal.edits[i] = FileEdit(
                                file=edit.file,
                                original_text=edit.original_text,
                                replacement_text=arb_lookup[key],
                                rationale=edit.rationale,
                            )

                if on_phase:
                    on_phase("arbitration_done", arb_results)

                # Run one more review round with the arbitrated edits
                continue

            # No contested edits found but still stalled — exit early
            stalled = True
            break

        # Merge modifications for next round
        current_proposals = merge_proposals(current_proposals, reviews)

        # Re-validate after merge
        for proposal in current_proposals:
            proposal.edits = validate_edits(proposal.edits, file_contents)

    # 7. Use the best proposals if consensus wasn't reached
    if not consensus:
        current_proposals = best_proposals

    # 8. Flatten final edits
    final_edits = []
    for p in current_proposals:
        final_edits.extend(p.edits)
    final_edits = deduplicate_edits(final_edits)

    # 9. Collect dissenting opinions if consensus was not reached
    #    Use the best round's reviews to identify dissenters (not the last round)
    dissents: list[Dissent] = []
    if not consensus and rounds and final_edits:
        best_round_reviews = rounds[best_round].reviews
        dissenting_agents = [
            r.agent_name for r in best_round_reviews
            if not r.all_approved and r.error is None
        ]
        if dissenting_agents:
            dissents = await _collect_dissents(
                config, dissenting_agents, current_proposals,
                file_contents, repo_root, on_progress,
            )
            for d in dissents:
                total_usage += d.usage

            if on_phase:
                on_phase("dissents_done", dissents)

    return IterationResult(
        consensus_reached=consensus,
        final_edits=final_edits,
        proposals=current_proposals,
        rounds=rounds,
        total_duration_seconds=time.monotonic() - start,
        files_reviewed=files_reviewed,
        total_usage=total_usage,
        dissents=dissents,
        best_round=best_round,
        best_approvals=best_approvals,
        stalled=stalled,
    )
