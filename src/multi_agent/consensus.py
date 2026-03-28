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
    build_agent_system_prompt,
    build_cli_args,
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
class IterationResult:
    consensus_reached: bool
    final_edits: list[FileEdit]
    proposals: list[AgentProposal]
    rounds: list[IterationRound]
    total_duration_seconds: float
    files_reviewed: list[str]
    total_usage: TokenUsage = field(default_factory=TokenUsage)


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
            original_agent=item.get("original_agent", ""),
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
        proc = await asyncio.wait_for(
            _spawn_claude(cli_args, propose_prompt, repo_root),
            timeout=timeout_seconds,
        )

        if proc.returncode != 0:
            stderr_text = proc.stderr.strip() if proc.stderr else "unknown error"
            return AgentProposal(
                agent_name=agent_name,
                edits=[],
                summary=f"Agent process failed (exit {proc.returncode}).",
                error=stderr_text[:500],
                duration_seconds=time.monotonic() - start,
            )

        stdout = proc.stdout or ""
        output = _extract_json(stdout)

        # Extract usage from the outer CLI envelope before unwrapping
        usage = _extract_usage(output) if output else TokenUsage()

        if output and "result" in output and "edits" not in output:
            inner = output["result"]
            if isinstance(inner, str):
                output = _extract_json(inner)
            elif isinstance(inner, dict):
                output = inner

        if output is None:
            return AgentProposal(
                agent_name=agent_name,
                edits=[],
                summary="Could not parse agent response.",
                error=f"Unparseable output: {stdout[:300]}",
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
        proc = await asyncio.wait_for(
            _spawn_claude(cli_args, review_prompt, repo_root),
            timeout=timeout_seconds,
        )

        if proc.returncode != 0:
            stderr_text = proc.stderr.strip() if proc.stderr else "unknown error"
            return AgentReviewResponse(
                agent_name=agent_name,
                all_approved=True,  # failed agent doesn't block
                proposal_reviews=[],
                summary=f"Agent process failed (exit {proc.returncode}).",
                error=stderr_text[:500],
                duration_seconds=time.monotonic() - start,
            )

        stdout = proc.stdout or ""
        output = _extract_json(stdout)

        # Extract usage from the outer CLI envelope before unwrapping
        usage = _extract_usage(output) if output else TokenUsage()

        if output and "result" in output and "all_approved" not in output:
            inner = output["result"]
            if isinstance(inner, str):
                output = _extract_json(inner)
            elif isinstance(inner, dict):
                output = inner

        if output is None:
            return AgentReviewResponse(
                agent_name=agent_name,
                all_approved=True,  # can't parse → don't block
                proposal_reviews=[],
                summary="Could not parse agent response.",
                error=f"Unparseable output: {stdout[:300]}",
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
    """Remove duplicate edits (same file, original_text, replacement_text)."""
    seen: set[tuple[str, str, str]] = set()
    result = []
    for edit in edits:
        key = (edit.file, edit.original_text, edit.replacement_text)
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
    on_progress: Callable[[str, str], None] | None = None,
) -> list[AgentProposal]:
    """Run all enabled agents in propose mode (parallel)."""
    propose_prompt = build_propose_prompt(file_contents, canon, staged_diff)

    proposals: list[AgentProposal] = []

    async with asyncio.TaskGroup() as tg:
        tasks = {}
        for name, agent_cfg in config.agents.items():
            if not agent_cfg.enabled:
                continue
            system_prompt = build_agent_system_prompt(
                name, "propose", agent_cfg.system_prompt_override,
            )
            cli_args = build_cli_args(
                name, system_prompt, agent_cfg.model, repo_root,
            )
            tasks[name] = tg.create_task(
                _run_single_proposer(
                    name, propose_prompt, cli_args, repo_root,
                    config.general.timeout_seconds, on_progress,
                )
            )

    for name, task in tasks.items():
        proposals.append(task.result())

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
    """Run all enabled agents in review mode (parallel)."""
    review_prompt = build_review_round_prompt(
        proposals, file_contents, canon, round_number,
    )

    reviews: list[AgentReviewResponse] = []

    async with asyncio.TaskGroup() as tg:
        tasks = {}
        for name, agent_cfg in config.agents.items():
            if not agent_cfg.enabled:
                continue
            system_prompt = build_agent_system_prompt(
                name, "review", agent_cfg.system_prompt_override,
            )
            cli_args = build_cli_args(
                name, system_prompt, agent_cfg.model, repo_root,
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


async def run_iteration_loop(
    config: MultiAgentConfig,
    repo_root: str,
    target_files: list[str] | None = None,
    on_progress: Callable[[str, str], None] | None = None,
) -> IterationResult:
    """Run the full propose-review-iterate loop.

    If target_files is None, reviews staged files. Otherwise reviews the
    specified files from the working tree.
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
        config, file_contents, canon, staged_diff, repo_root, on_progress,
    )
    for p in proposals:
        total_usage += p.usage

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

        if consensus_reached:
            consensus = True
            break

        # Merge modifications for next round
        current_proposals = merge_proposals(current_proposals, reviews)

        # Re-validate after merge
        for proposal in current_proposals:
            proposal.edits = validate_edits(proposal.edits, file_contents)

    # 7. Flatten final edits
    final_edits = []
    for p in current_proposals:
        final_edits.extend(p.edits)
    final_edits = deduplicate_edits(final_edits)

    return IterationResult(
        consensus_reached=consensus,
        final_edits=final_edits,
        proposals=current_proposals,
        rounds=rounds,
        total_duration_seconds=time.monotonic() - start,
        files_reviewed=files_reviewed,
        total_usage=total_usage,
    )
