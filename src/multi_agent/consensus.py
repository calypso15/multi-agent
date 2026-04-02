"""Propose-review-iterate orchestration loop."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from multi_agent.agents import (
    ARBITRATOR_PROMPT,
    build_agent_system_prompt,
    build_cli_args,
    build_name_normalizer,
)
from multi_agent.config import get_display_name
from multi_agent.claude_runner import run_agent
from multi_agent.config import MultiAgentConfig
from multi_agent.context import (
    build_propose_prompt,
    build_review_round_prompt,
    get_staged_content,
    get_staged_diff,
    get_staged_files,
    load_reference,
)
from multi_agent.models import (
    AgentProposal,
    AgentReviewResponse,
    ArbitrationDone,
    ArbitrationResult,
    ArbitrationStart,
    ContestedEdit,
    Dissent,
    DissentsDone,
    FileEdit,
    IterationResult,
    IterationRound,
    PhaseEvent,
    ProposalReview,
    ProposeDone,
    ReviewDone,
    TokenUsage,
    count_approvals,
    parse_edits,
    parse_proposal_reviews,
)

# Re-exports so existing importers (tests, output, cli) continue to work.
__all__ = [
    "AgentProposal",
    "AgentReviewResponse",
    "ArbitrationResult",
    "ContestedEdit",
    "Dissent",
    "FileEdit",
    "IterationResult",
    "IterationRound",
    "PhaseEvent",
    "ProposalReview",
    "TokenUsage",
    "count_approvals",
    "merge_proposals",
    "resolve_file_args",
    "run_iteration_loop",
    "_edit_overlaps_locked",
]


# --- File resolution ---


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


# --- Edit validation and deduplication ---


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


def deduplicate_edits(edits: list[FileEdit]) -> tuple[list[FileEdit], list[FileEdit]]:
    """Remove exact duplicate edits.

    When multiple edits target the same (file, original_text, replacement_text),
    only the first is kept. Positional overlaps are handled downstream by
    diff-match-patch merge.

    Returns (kept_edits, dropped_edits).
    """
    seen: set[tuple[str, str, str]] = set()
    kept: list[FileEdit] = []
    dropped: list[FileEdit] = []
    for edit in edits:
        key = (edit.file, edit.original_text, edit.replacement_text)
        if key in seen:
            dropped.append(edit)
        else:
            seen.add(key)
            kept.append(edit)
    return kept, dropped


def _edit_overlaps_locked(
    edit: FileEdit,
    file_contents: dict[str, str],
    locked_regions: dict[str, list[tuple[int, int]]],
) -> bool:
    """Check if an edit overlaps any locked region in the original file."""
    regions = locked_regions.get(edit.file)
    if not regions:
        return False
    content = file_contents.get(edit.file, "")
    pos = content.find(edit.original_text)
    if pos < 0:
        return False
    edit_end = pos + len(edit.original_text)
    return any(pos < end and start < edit_end for start, end in regions)


# --- Proposal merging ---


def merge_proposals(
    proposals: list[AgentProposal],
    reviews: list[AgentReviewResponse],
    locked_regions: dict[str, list[tuple[int, int]]] | None = None,
    file_contents: dict[str, str] | None = None,
) -> list[AgentProposal]:
    """Apply review modifications to proposals.

    For each MODIFY review, update the corresponding edit's replacement_text.
    If multiple reviewers modify the same edit, the last one wins (next review
    round can resolve disagreements).

    Edits overlapping locked_regions are protected from modification --
    their arbitration result is final.
    """
    modifications: dict[tuple[str, int], str] = {}
    for review in reviews:
        for pr in review.proposal_reviews:
            if pr.verdict == "MODIFY" and pr.modified_replacement is not None:
                if locked_regions and file_contents:
                    orig_proposal = next(
                        (p for p in proposals if p.agent_name == pr.original_agent), None,
                    )
                    if (orig_proposal
                            and pr.edit_index < len(orig_proposal.edits)
                            and _edit_overlaps_locked(
                                orig_proposal.edits[pr.edit_index],
                                file_contents, locked_regions)):
                        continue
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
            usage=proposal.usage,
        ))

    return updated


# --- Parallel agent helper ---


# --- Individual agent runners ---


async def _run_single_proposer(
    agent_name: str,
    propose_prompt: str,
    cli_args: list[str],
    repo_root: str,
    timeout_seconds: int,
    on_progress: Callable[[str, str], None] | None = None,
    model: str | None = None,
) -> AgentProposal:
    """Run a single agent in propose mode."""
    label = f"proposing ({model})" if model else "proposing"
    raw = await run_agent(
        agent_name, propose_prompt, cli_args, repo_root,
        timeout_seconds, on_progress, progress_label=label,
    )
    if raw.error:
        return AgentProposal(
            agent_name=agent_name, edits=[], summary="Agent failed.",
            error=raw.error, duration_seconds=raw.duration_seconds, usage=raw.usage,
        )

    edits = parse_edits(raw.output.get("edits", []))
    if on_progress:
        on_progress(agent_name, f"done \u2014 {len(edits)} edit(s)")

    return AgentProposal(
        agent_name=agent_name, edits=edits,
        summary=raw.output.get("summary", ""),
        duration_seconds=raw.duration_seconds, usage=raw.usage,
    )


async def _run_single_reviewer(
    agent_name: str,
    review_prompt: str,
    cli_args: list[str],
    repo_root: str,
    timeout_seconds: int,
    round_number: int,
    on_progress: Callable[[str, str], None] | None = None,
    normalizer: Callable[[str], str] | None = None,
    model: str | None = None,
) -> AgentReviewResponse:
    """Run a single agent in review mode."""
    model_tag = f" ({model})" if model else ""
    raw = await run_agent(
        agent_name, review_prompt, cli_args, repo_root,
        timeout_seconds, on_progress,
        progress_label=f"reviewing{model_tag} (round {round_number + 1})",
    )
    if raw.error:
        return AgentReviewResponse(
            agent_name=agent_name, all_approved=True,  # failed agent doesn't block
            proposal_reviews=[], summary="Agent failed.",
            error=raw.error, duration_seconds=raw.duration_seconds, usage=raw.usage,
        )

    all_approved = raw.output.get("all_approved", True)
    proposal_reviews = parse_proposal_reviews(
        raw.output.get("proposal_reviews", []),
        normalizer=normalizer or (lambda x: x),
    )

    if on_progress:
        if all_approved:
            on_progress(agent_name, f"done \u2014 approved all (round {round_number + 1})")
        else:
            mod_count = sum(1 for r in proposal_reviews if r.verdict == "MODIFY")
            on_progress(agent_name, f"done \u2014 modified {mod_count} (round {round_number + 1})")

    return AgentReviewResponse(
        agent_name=agent_name, all_approved=all_approved,
        proposal_reviews=proposal_reviews,
        summary=raw.output.get("summary", ""),
        duration_seconds=raw.duration_seconds, usage=raw.usage,
    )


async def _run_single_dissenter(
    agent_name: str,
    prompt: str,
    cli_args: list[str],
    repo_root: str,
    timeout_seconds: int,
    on_progress: Callable[[str, str], None] | None = None,
    model: str | None = None,
) -> Dissent:
    """Run a single agent to collect a dissenting opinion."""
    label = f"dissenting ({model})" if model else "dissenting"
    raw = await run_agent(
        agent_name, prompt, cli_args, repo_root,
        timeout_seconds, on_progress,
        progress_label=label, report_tool_use=False,
    )
    if raw.error:
        return Dissent(
            agent_name=agent_name, opinion="",
            duration_seconds=raw.duration_seconds, usage=raw.usage,
        )

    opinion = raw.output.get("opinion", "")
    if on_progress:
        on_progress(agent_name, "done \u2014 dissent recorded")

    return Dissent(
        agent_name=agent_name, opinion=opinion,
        duration_seconds=raw.duration_seconds, usage=raw.usage,
    )


# --- Phase runners ---


async def run_propose_phase(
    config: MultiAgentConfig,
    file_contents: dict[str, str],
    reference: dict[str, str],
    staged_diff: str | None,
    repo_root: str,
    command_name: str | None = None,
    command_prompt: str | None = None,
    severity_filter: bool = True,
    command_propose_model: str | None = None,
    on_progress: Callable[[str, str], None] | None = None,
) -> list[AgentProposal]:
    """Run all enabled agents in propose mode (sequentially)."""
    propose_prompt = build_propose_prompt(
        file_contents, reference, staged_diff,
        min_severity=config.general.min_severity,
        severity_filter=severity_filter,
    )

    proposals: list[AgentProposal] = []
    for name, agent_cfg in config.agents.items():
        if not agent_cfg.enabled:
            continue
        system_prompt = build_agent_system_prompt(
            name, "command", agent_cfg.system_prompt,
            command_name=command_name,
            command_prompt=command_prompt,
        )
        max_turns = (
            agent_cfg.propose_max_turns
            if agent_cfg.propose_max_turns is not None
            else config.general.propose_max_turns
        )
        model = command_propose_model or agent_cfg.propose_model
        cli_args = build_cli_args(
            name, system_prompt, model, repo_root,
            max_turns=max_turns,
            allowed_tools=agent_cfg.allowed_tools or None,
        )
        proposals.append(await _run_single_proposer(
            name, propose_prompt, cli_args, repo_root,
            config.general.timeout_seconds, on_progress,
            model=model,
        ))

    return proposals


def _last_modifiers(
    reviews: list[AgentReviewResponse],
) -> dict[tuple[str, int], str]:
    """Identify who last modified each edit in a review round.

    When multiple reviewers modify the same edit, the last one in
    iteration order wins (matching merge_proposals behavior).
    Returns {(original_agent, edit_index): modifier_agent_name}.
    """
    last: dict[tuple[str, int], str] = {}
    for review in reviews:
        for pr in review.proposal_reviews:
            if pr.verdict == "MODIFY" and pr.modified_replacement is not None:
                last[(pr.original_agent, pr.edit_index)] = review.agent_name
    return last


def _filter_self_modified_edits(
    proposals: list[AgentProposal],
    agent_name: str,
    previous_reviews: list[AgentReviewResponse],
) -> list[AgentProposal]:
    """Return proposals with edits this agent last modified removed.

    After merge_proposals, an edit may contain text written by a reviewing
    agent rather than the original proposer.  There is no point in asking
    that reviewer to re-review their own text, so we strip those edits
    from the prompt they receive.
    """
    last_mods = _last_modifiers(previous_reviews)
    skip = {
        key for key, modifier in last_mods.items()
        if modifier == agent_name
    }
    if not skip:
        return proposals

    filtered = []
    for proposal in proposals:
        kept = [
            e for i, e in enumerate(proposal.edits)
            if (proposal.agent_name, i) not in skip
        ]
        filtered.append(AgentProposal(
            agent_name=proposal.agent_name,
            edits=kept,
            summary=proposal.summary,
            duration_seconds=proposal.duration_seconds,
            error=proposal.error,
            usage=proposal.usage,
        ))
    return filtered


async def run_review_phase(
    config: MultiAgentConfig,
    proposals: list[AgentProposal],
    file_contents: dict[str, str],
    reference: dict[str, str],
    repo_root: str,
    round_number: int,
    on_progress: Callable[[str, str], None] | None = None,
    previous_reviews: list[AgentReviewResponse] | None = None,
    display_names: dict[str, str] | None = None,
    normalizer: Callable[[str], str] | None = None,
    command_review_model: str | None = None,
) -> list[AgentReviewResponse]:
    """Run all enabled agents in review mode (sequentially).

    Each agent only reviews proposals from OTHER agents -- not its own.
    Additionally, edits an agent last modified in the previous round are
    excluded from their review prompt (they already approved their version).
    """
    reviews: list[AgentReviewResponse] = []

    for name, agent_cfg in config.agents.items():
        if not agent_cfg.enabled:
            continue
        other_proposals = [p for p in proposals if p.agent_name != name]

        # Also strip edits this agent was the last to modify
        if previous_reviews:
            other_proposals = _filter_self_modified_edits(
                other_proposals, name, previous_reviews,
            )

        if not any(p.edits for p in other_proposals):
            reviews.append(AgentReviewResponse(
                agent_name=name,
                all_approved=True,
                proposal_reviews=[],
                summary="No proposals from other agents to review.",
            ))
            continue

        review_prompt = build_review_round_prompt(
            other_proposals, file_contents, reference, round_number,
            display_names=display_names,
        )
        system_prompt = build_agent_system_prompt(
            name, "review", agent_cfg.system_prompt,
        )
        model = command_review_model or agent_cfg.review_model or agent_cfg.propose_model
        max_turns = (
            agent_cfg.review_max_turns
            if agent_cfg.review_max_turns is not None
            else config.general.review_max_turns
        )
        cli_args = build_cli_args(
            name, system_prompt, model, repo_root,
            max_turns=max_turns,
        )
        reviews.append(await _run_single_reviewer(
            name, review_prompt, cli_args, repo_root,
            config.general.timeout_seconds, round_number, on_progress,
            normalizer=normalizer,
            model=model,
        ))

    return reviews


# --- Dissent collection ---


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
            parts.append(f"\n### Edit {i} \u2014 {edit.file}\n")
            parts.append(f"**Replace:**\n```\n{edit.original_text}\n```\n")
            parts.append(f"**With:**\n```\n{edit.replacement_text}\n```\n")

    parts.append(
        "\n# YOUR TASK\n"
        "State your most important concern about these changes in 2-4 sentences. "
        "Return your response as JSON with a single \"opinion\" field.\n"
    )
    return "".join(parts)


async def _collect_dissents(
    config: MultiAgentConfig,
    dissenting_agents: list[str],
    proposals: list[AgentProposal],
    file_contents: dict[str, str],
    repo_root: str,
    on_progress: Callable[[str, str], None] | None = None,
    command_review_model: str | None = None,
) -> list[Dissent]:
    """Collect brief dissenting opinions from agents that didn't approve."""
    prompt = _build_dissent_prompt(proposals, file_contents)

    dissents: list[Dissent] = []
    for name in dissenting_agents:
        agent_cfg = config.agents.get(name)
        if not agent_cfg or not agent_cfg.enabled:
            continue
        system_prompt = build_agent_system_prompt(name, "dissent", agent_cfg.system_prompt)
        model = command_review_model or agent_cfg.review_model or agent_cfg.propose_model
        cli_args = build_cli_args(
            name, system_prompt, model, repo_root,
            max_turns=1,
        )
        result = await _run_single_dissenter(
            name, prompt, cli_args, repo_root,
            config.general.timeout_seconds, on_progress,
            model=model,
        )
        if result.opinion:
            dissents.append(result)

    return dissents


# --- Stall detection and arbitration ---


def _detect_stall(rounds: list[IterationRound]) -> bool:
    """Detect if the review loop has stalled (no improvement in the latest round)."""
    if len(rounds) < 2:
        return False
    return count_approvals(rounds[-1].reviews) <= count_approvals(rounds[-2].reviews)


def _find_contested_edits(
    rounds: list[IterationRound],
    current_proposals: list[AgentProposal],
) -> list[ContestedEdit]:
    """Find edits that have been modified in 2+ consecutive rounds."""
    if len(rounds) < 2:
        return []

    modified_per_round: list[set[tuple[str, int]]] = []
    for rnd in rounds:
        modified = set()
        for review in rnd.reviews:
            for pr in review.proposal_reviews:
                if pr.verdict == "MODIFY":
                    modified.add((pr.original_agent, pr.edit_index))
        modified_per_round.append(modified)

    if len(modified_per_round) < 2:
        return []
    contested_keys = modified_per_round[-1] & modified_per_round[-2]

    if not contested_keys:
        return []

    contested = []
    last_reviews = rounds[-1].reviews
    prev_reviews = rounds[-2].reviews

    approved_in_latest = {
        r.agent_name for r in last_reviews
        if r.all_approved and r.error is None
    }

    for agent_name, edit_idx in contested_keys:
        proposal = next(
            (p for p in current_proposals if p.agent_name == agent_name),
            None,
        )
        if not proposal or edit_idx >= len(proposal.edits):
            continue

        edit = proposal.edits[edit_idx]
        versions: dict[str, str] = {}
        rationales: dict[str, str] = {}

        for rnd_reviews in (prev_reviews, last_reviews):
            for review in rnd_reviews:
                if review.agent_name in approved_in_latest:
                    continue
                for pr in review.proposal_reviews:
                    if (pr.original_agent == agent_name
                            and pr.edit_index == edit_idx
                            and pr.verdict == "MODIFY"
                            and pr.modified_replacement):
                        versions[review.agent_name] = pr.modified_replacement
                        rationales[review.agent_name] = pr.rationale

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
    command_review_model: str | None = None,
) -> list[ArbitrationResult]:
    """Run arbitration on contested edits using run_agent for proper error handling."""
    model = command_review_model
    if model is None:
        for agent_cfg in config.agents.values():
            if agent_cfg.enabled:
                model = agent_cfg.review_model or agent_cfg.propose_model
                break

    results: list[ArbitrationResult] = []
    for i, contested in enumerate(contested_edits):
        key = f"arbitrator_{i}"
        prompt = _build_arbitration_prompt(contested, file_contents)
        cli_args = build_cli_args(
            key, ARBITRATOR_PROMPT, model, repo_root, max_turns=1,
        )
        results.append(await _run_single_arbitrator(
            key, contested, prompt, cli_args, repo_root,
            config.general.timeout_seconds, on_progress,
        ))

    return results


async def _run_single_arbitrator(
    arb_name: str,
    contested: ContestedEdit,
    prompt: str,
    cli_args: list[str],
    repo_root: str,
    timeout_seconds: int,
    on_progress: Callable[[str, str], None] | None = None,
) -> ArbitrationResult:
    """Run a single arbitration call via run_agent (with full error handling)."""
    if on_progress:
        on_progress("arbitrator", f"resolving conflict in {contested.file}")

    raw = await run_agent(
        arb_name, prompt, cli_args, repo_root,
        timeout_seconds, on_progress=None,
        progress_label="arbitrating", report_tool_use=False,
    )

    replacement = ""
    rationale = ""
    if raw.output:
        replacement = raw.output.get("replacement_text", "")
        rationale = raw.output.get("rationale", "")

    if on_progress:
        on_progress("arbitrator", "done \u2014 conflict resolved")

    return ArbitrationResult(
        file=contested.file,
        original_text=contested.original_text,
        replacement_text=replacement or list(contested.versions.values())[0],
        rationale=rationale,
        usage=raw.usage,
    )


# --- Main iteration loop ---


async def run_iteration_loop(
    config: MultiAgentConfig,
    repo_root: str,
    target_files: list[str] | None = None,
    command_name: str | None = None,
    command_prompt: str | None = None,
    severity_filter: bool = True,
    command_propose_model: str | None = None,
    command_review_model: str | None = None,
    on_progress: Callable[[str, str], None] | None = None,
    on_phase: Callable[[PhaseEvent], None] | None = None,
) -> IterationResult:
    """Run the full propose-review-iterate loop.

    If target_files is None, reviews staged files. Otherwise reviews the
    specified files from the working tree.
    command_name/command_prompt: the TOML command driving this run.
    severity_filter: whether to apply min_severity filtering.
    command_propose_model/command_review_model: per-command model overrides.
    on_phase: callback for typed phase events (ProposeDone, ReviewDone, etc.).
    """
    root = Path(repo_root)
    start = time.monotonic()

    normalizer = build_name_normalizer(config.agents)
    display_names = {k: get_display_name(k, v) for k, v in config.agents.items()}

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

    # 2. Load reference files
    reference = load_reference(
        root,
        config.general.reference_directories,
        config.general.file_patterns,
        config.general.max_reference_size_kb,
    )

    # 3. Propose phase
    total_usage = TokenUsage()

    proposals = await run_propose_phase(
        config, file_contents, reference, staged_diff, repo_root,
        command_name=command_name, command_prompt=command_prompt,
        severity_filter=severity_filter,
        command_propose_model=command_propose_model,
        on_progress=on_progress,
    )
    for p in proposals:
        total_usage += p.usage

    if on_phase:
        on_phase(ProposeDone(proposals=proposals))

    # 4. Validate and deduplicate
    for proposal in proposals:
        proposal.edits = validate_edits(proposal.edits, file_contents)

    all_edits = []
    for p in proposals:
        all_edits.extend(p.edits)
    all_edits, initial_dropped = deduplicate_edits(all_edits)

    # Sync per-agent proposals to remove exact duplicates
    if initial_dropped:
        kept_set = {(e.file, e.original_text, e.replacement_text) for e in all_edits}
        for proposal in proposals:
            proposal.edits = [
                e for e in proposal.edits
                if (e.file, e.original_text, e.replacement_text) in kept_set
            ]

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

    # Regions resolved by arbitration -- no further changes allowed.
    locked_regions: dict[str, list[tuple[int, int]]] = {}

    # Track the best (highest approval) proposals seen
    best_proposals = current_proposals
    best_approvals = 0
    best_round = -1

    previous_reviews: list[AgentReviewResponse] | None = None

    for round_num in range(config.general.max_rounds):
        reviews = await run_review_phase(
            config, current_proposals, file_contents, reference,
            repo_root, round_num, on_progress,
            previous_reviews=previous_reviews,
            display_names=display_names,
            normalizer=normalizer,
            command_review_model=command_review_model,
        )
        for r in reviews:
            total_usage += r.usage

        approvals = count_approvals(reviews)
        consensus_reached = approvals >= config.general.consensus_threshold

        rounds.append(IterationRound(
            round_number=round_num,
            reviews=reviews,
            consensus_reached=consensus_reached,
        ))

        if on_phase:
            on_phase(ReviewDone(
                round_number=round_num,
                reviews=reviews,
                consensus_threshold=config.general.consensus_threshold,
            ))

        # Track the version with the highest approval count
        if approvals >= best_approvals:
            best_approvals = approvals
            best_proposals = current_proposals
            best_round = round_num

        if consensus_reached:
            consensus = True
            break

        # Detect stall -- no improvement for 2 consecutive rounds
        if _detect_stall(rounds):
            contested = _find_contested_edits(
                rounds, current_proposals,
            )
            if contested:
                if on_phase:
                    on_phase(ArbitrationStart(contested=contested))

                arb_results = await _run_arbitration(
                    contested, file_contents, config, repo_root,
                    on_progress,
                    command_review_model=command_review_model,
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

                # Lock arbitrated regions
                for ar in arb_results:
                    pos = file_contents.get(ar.file, "").find(ar.original_text)
                    if pos >= 0:
                        locked_regions.setdefault(ar.file, []).append(
                            (pos, pos + len(ar.original_text))
                        )

                if on_phase:
                    on_phase(ArbitrationDone(results=arb_results))

                # Run one more review round with the arbitrated edits
                continue

            # No contested edits found but still stalled -- exit early
            stalled = True
            break

        # Track reviews for next round's self-modification filtering
        previous_reviews = reviews

        # Merge modifications for next round (locked regions are protected)
        current_proposals = merge_proposals(
            current_proposals, reviews, locked_regions, file_contents,
        )

        # If an agent's only modifications targeted locked regions, they
        # effectively approved -- update their review for consensus counting.
        if locked_regions:
            for review in reviews:
                if review.all_approved or review.error:
                    continue
                has_unlocked_mod = False
                for pr in review.proposal_reviews:
                    if pr.verdict != "MODIFY":
                        continue
                    orig = next((p for p in current_proposals if p.agent_name == pr.original_agent), None)
                    if orig and pr.edit_index < len(orig.edits):
                        if not _edit_overlaps_locked(
                            orig.edits[pr.edit_index], file_contents, locked_regions,
                        ):
                            has_unlocked_mod = True
                            break
                if not has_unlocked_mod:
                    review.all_approved = True

        # Re-validate after merge
        for proposal in current_proposals:
            proposal.edits = validate_edits(proposal.edits, file_contents)

    # 7. Use the best proposals if consensus wasn't reached
    if not consensus:
        current_proposals = best_proposals

    # 8. Merge all agents' edits using diff-match-patch
    from multi_agent.merge import apply_arbitration_to_merged, merge_agent_edits

    merge_result = merge_agent_edits(file_contents, current_proposals)

    # 8a. Arbitrate any merge conflicts (skip if consensus -- agents already
    #     approved each other's work, overlaps just need mechanical resolution)
    if merge_result.failed_patches and not consensus:
        contested: list[ContestedEdit] = []
        for fp in merge_result.failed_patches:
            original = file_contents.get(fp.file, "")

            failing_proposal = next(
                (p for p in current_proposals if p.agent_name == fp.agent_name), None,
            )
            if not failing_proposal:
                continue
            failing_edit = None
            for edit in failing_proposal.edits:
                if edit.file == fp.file and edit.original_text in original:
                    failing_edit = edit
                    break
            if not failing_edit:
                continue

            f_start = original.find(failing_edit.original_text)
            f_end = f_start + len(failing_edit.original_text)

            for other in current_proposals:
                if other.agent_name == fp.agent_name:
                    continue
                for other_edit in other.edits:
                    if other_edit.file != fp.file:
                        continue
                    o_start = original.find(other_edit.original_text)
                    if o_start < 0:
                        continue
                    o_end = o_start + len(other_edit.original_text)
                    if f_start < o_end and o_start < f_end:
                        union_start = min(f_start, o_start)
                        union_end = max(f_end, o_end)
                        union_text = original[union_start:union_end]

                        other_version = union_text.replace(
                            other_edit.original_text, other_edit.replacement_text, 1,
                        )
                        failing_version = union_text.replace(
                            failing_edit.original_text, failing_edit.replacement_text, 1,
                        )

                        contested.append(ContestedEdit(
                            file=fp.file,
                            original_text=union_text,
                            versions={
                                other.agent_name: other_version,
                                fp.agent_name: failing_version,
                            },
                            rationales={
                                other.agent_name: other_edit.rationale,
                                fp.agent_name: failing_edit.rationale,
                            },
                        ))
                        break
                else:
                    continue
                break

        if contested:
            if on_phase:
                on_phase(ArbitrationStart(contested=contested))

            winner_texts: list[str] = []
            for ce, cfp in zip(contested, merge_result.failed_patches):
                for agent_name, version in ce.versions.items():
                    if agent_name != cfp.agent_name:
                        winner_texts.append(version)
                        break

            arb_results = await _run_arbitration(
                contested, file_contents, config, repo_root, on_progress,
                command_review_model=command_review_model,
            )
            for ar, winner_text in zip(arb_results, winner_texts):
                total_usage += ar.usage
                apply_arbitration_to_merged(
                    merge_result.merged_texts, ar.file,
                    winner_text, ar.replacement_text,
                )

            if on_phase:
                on_phase(ArbitrationDone(results=arb_results))

    # Derive final_edits (one per changed file) for truthiness checks
    final_edits = [
        FileEdit(file=fp, original_text=file_contents.get(fp, ""),
                 replacement_text=mt, rationale="merged from all agent proposals")
        for fp, mt in merge_result.merged_texts.items()
    ]

    # 9. Collect dissenting opinions if consensus was not reached
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
                command_review_model=command_review_model,
            )
            for d in dissents:
                total_usage += d.usage

            if on_phase:
                on_phase(DissentsDone(dissents=dissents))

    return IterationResult(
        consensus_reached=consensus,
        final_edits=final_edits,
        proposals=current_proposals,
        rounds=rounds,
        total_duration_seconds=time.monotonic() - start,
        files_reviewed=files_reviewed,
        merged_texts=merge_result.merged_texts,
        total_usage=total_usage,
        dissents=dissents,
        best_round=best_round,
        best_approvals=best_approvals,
        stalled=stalled,
    )
