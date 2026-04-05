"""Propose-review-iterate orchestration loop."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Callable

from multi_agent.agents import (
    build_agent_system_prompt,
    build_name_normalizer,
)
from multi_agent.arbitration import (
    build_dissent_prompt,
    collect_dissents,
    detect_stall,
    find_contested_edits,
    find_dissenting_agents,
    run_arbitration,
)
from multi_agent.backend import AgentBackend
from multi_agent.config import (
    ResolvedRunConfig,
    get_display_name,
)
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
    ArbitrationStart,
    ContestedEdit,
    Dissent,
    DissentsDone,
    FileEdit,
    IterationResult,
    IterationRound,
    PhaseEvent,
    ProposeDone,
    ProposeStart,
    ReviewDone,
    ReviewStart,
    TokenUsage,
    count_blocking_approvals,
    filter_edits_by_severity,
    parse_edits,
    parse_proposal_reviews,
)

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
                    severity=edit.severity,
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


# --- Individual agent runners ---


async def _run_single_proposer(
    agent_name: str,
    propose_prompt: str,
    backend: AgentBackend,
    system_prompt: str,
    repo_root: str,
    timeout_seconds: int,
    on_progress: Callable[[str, str], None] | None = None,
    model: str | None = None,
    max_turns: int = 0,
    allowed_tools: list[str] | None = None,
) -> AgentProposal:
    """Run a single agent in propose mode."""
    label = f"proposing ({model})" if model else "proposing"
    raw = await backend.run_agent(
        agent_name, propose_prompt, system_prompt, repo_root,
        timeout_seconds, model=model, max_turns=max_turns,
        allowed_tools=allowed_tools, on_progress=on_progress,
        progress_label=label,
    )
    if raw.error:
        return AgentProposal(
            agent_name=agent_name, edits=[], summary="Agent failed.",
            error=raw.error, duration_seconds=raw.duration_seconds, usage=raw.usage,
            turns_taken=raw.turns_taken, tool_usage=raw.tool_usage,
        )

    edits = parse_edits(raw.output.get("edits", []))
    if on_progress:
        on_progress(agent_name, f"done \u2014 {len(edits)} edit(s)")

    return AgentProposal(
        agent_name=agent_name, edits=edits,
        summary=raw.output.get("summary", ""),
        duration_seconds=raw.duration_seconds, usage=raw.usage,
        turns_taken=raw.turns_taken, tool_usage=raw.tool_usage,
    )


async def _run_single_reviewer(
    agent_name: str,
    review_prompt: str,
    backend: AgentBackend,
    system_prompt: str,
    repo_root: str,
    timeout_seconds: int,
    round_number: int,
    on_progress: Callable[[str, str], None] | None = None,
    normalizer: Callable[[str], str] | None = None,
    model: str | None = None,
    max_turns: int = 0,
) -> AgentReviewResponse:
    """Run a single agent in review mode."""
    model_tag = f" ({model})" if model else ""
    raw = await backend.run_agent(
        agent_name, review_prompt, system_prompt, repo_root,
        timeout_seconds, model=model, max_turns=max_turns,
        on_progress=on_progress,
        progress_label=f"reviewing{model_tag} (round {round_number + 1})",
    )
    if raw.error:
        return AgentReviewResponse(
            agent_name=agent_name, all_approved=True,  # failed agent doesn't block
            proposal_reviews=[], summary="Agent failed.",
            error=raw.error, duration_seconds=raw.duration_seconds, usage=raw.usage,
            turns_taken=raw.turns_taken, tool_usage=raw.tool_usage,
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
        turns_taken=raw.turns_taken, tool_usage=raw.tool_usage,
    )


# --- Phase runners ---


async def run_propose_phase(
    resolved: ResolvedRunConfig,
    file_contents: dict[str, str],
    per_agent_reference: dict[str, dict[str, str]],
    staged_diff: str | None,
    repo_root: str,
    backend: AgentBackend,
    on_progress: Callable[[str, str], None] | None = None,
) -> list[AgentProposal]:
    """Run all enabled agents in propose mode (sequentially).

    per_agent_reference: pre-loaded {agent_name: {path: content}} reference
    files, keyed by agent name.
    """
    proposals: list[AgentProposal] = []
    for name, agent_cfg in resolved.agents.items():
        if not agent_cfg.enabled:
            continue
        settings = resolved.agent_settings[name]

        agent_ref = per_agent_reference.get(name, {})

        propose_prompt = build_propose_prompt(
            file_contents, agent_ref, staged_diff,
            min_severity=resolved.min_severity,
            propose_instructions=resolved.command_propose_instructions,
        )
        system_prompt = build_agent_system_prompt(
            name, "command", agent_cfg.system_prompt,
            command_name=resolved.command_name,
            command_prompt=resolved.command_prompt,
        )
        proposals.append(await _run_single_proposer(
            name, propose_prompt, backend, system_prompt, repo_root,
            settings.timeout_seconds, on_progress,
            model=settings.propose_model,
            max_turns=settings.propose_max_turns,
            allowed_tools=settings.allowed_tools or None,
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


async def run_review_phase(
    resolved: ResolvedRunConfig,
    proposals: list[AgentProposal],
    file_contents: dict[str, str],
    reference: dict[str, str],
    repo_root: str,
    round_number: int,
    backend: AgentBackend,
    on_progress: Callable[[str, str], None] | None = None,
    previous_reviews: list[AgentReviewResponse] | None = None,
    display_names: dict[str, str] | None = None,
    normalizer: Callable[[str], str] | None = None,
) -> list[AgentReviewResponse]:
    """Run all enabled agents in review mode (sequentially).

    Each agent only reviews proposals from OTHER agents -- not its own.
    Additionally, edits an agent last modified in the previous round are
    excluded from their review prompt (they already approved their version).
    """
    reviews: list[AgentReviewResponse] = []

    # Pre-compute which edits each agent last modified so they can be
    # skipped in the prompt (without removing them, which would shift
    # indices and break merge_proposals).
    last_mods = _last_modifiers(previous_reviews) if previous_reviews else {}

    for name, agent_cfg in resolved.agents.items():
        if not agent_cfg.enabled:
            continue
        settings = resolved.agent_settings[name]
        other_proposals = [p for p in proposals if p.agent_name != name]

        # Build skip set: edits this agent last modified
        skip_edits = {
            key for key, modifier in last_mods.items()
            if modifier == name
        }

        # Check if any visible edits remain
        has_visible = any(
            (p.agent_name, i) not in skip_edits
            for p in other_proposals
            for i in range(len(p.edits))
        )
        if not has_visible:
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
            skip_edits=skip_edits,
        )
        system_prompt = build_agent_system_prompt(
            name, "review", agent_cfg.system_prompt,
        )
        reviews.append(await _run_single_reviewer(
            name, review_prompt, backend, system_prompt, repo_root,
            settings.timeout_seconds, round_number, on_progress,
            normalizer=normalizer,
            model=settings.review_model,
            max_turns=settings.review_max_turns,
        ))

    return reviews


# --- Main iteration loop ---


async def run_iteration_loop(
    resolved: ResolvedRunConfig,
    repo_root: str,
    backend: AgentBackend,
    target_files: list[str] | None = None,
    on_progress: Callable[[str, str], None] | None = None,
    on_phase: Callable[[PhaseEvent], None] | None = None,
) -> IterationResult:
    """Run the full propose-review-iterate loop.

    resolved: fully resolved config (cascading already applied).
    If target_files is None, reviews staged files. Otherwise reviews the
    specified files from the working tree.
    """
    root = Path(repo_root)
    start = time.monotonic()

    normalizer = build_name_normalizer(resolved.agents)
    display_names = {k: get_display_name(k, v) for k, v in resolved.agents.items()}

    # Use the first agent's file_patterns for staged file resolution
    # (all agents see the same target files; per-agent patterns affect references)
    first_settings = next(iter(resolved.agent_settings.values()))

    # 1. Load file contents
    staged_diff: str | None = None
    if target_files is None:
        staged_files = get_staged_files(root, first_settings.file_patterns)
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
        staged_diff = get_staged_diff(root, first_settings.file_patterns)
        files_reviewed = [str(f) for f in staged_files]
    else:
        file_contents = {}
        for f in target_files:
            file_contents[f] = (root / f).read_text()
        files_reviewed = list(target_files)

    # 2. Load per-agent reference files once (shared by propose and review)
    per_agent_reference: dict[str, dict[str, str]] = {}
    for name, settings in resolved.agent_settings.items():
        per_agent_reference[name] = load_reference(
            root, settings.reference_directories,
            settings.file_patterns, settings.max_reference_size_kb,
        )

    # 3. Propose phase
    total_usage = TokenUsage()

    if on_phase:
        on_phase(ProposeStart())

    proposals = await run_propose_phase(
        resolved, file_contents, per_agent_reference, staged_diff, repo_root,
        backend, on_progress=on_progress,
    )
    for p in proposals:
        total_usage += p.usage

    if on_phase:
        on_phase(ProposeDone(proposals=proposals))

    # 4. Validate, filter by severity, and deduplicate
    proposals = [
        dataclasses.replace(p, edits=filter_edits_by_severity(
            validate_edits(p.edits, file_contents), resolved.min_severity,
        ))
        for p in proposals
    ]

    # Deduplicate identical edits across proposals — keep the first
    # occurrence and remove from all other proposals.
    seen_edits: set[tuple[str, str, str]] = set()
    deduped: list[AgentProposal] = []
    for proposal in proposals:
        kept: list[FileEdit] = []
        for edit in proposal.edits:
            key = (edit.file, edit.original_text, edit.replacement_text)
            if key not in seen_edits:
                seen_edits.add(key)
                kept.append(edit)
        deduped.append(dataclasses.replace(proposal, edits=kept))
    proposals = deduped

    all_edits = [e for p in proposals for e in p.edits]

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

    # Merge all agents' reference files for review rounds
    review_reference: dict[str, str] = {}
    for ref in per_agent_reference.values():
        review_reference.update(ref)

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

    for round_num in range(resolved.max_rounds):
        if on_phase:
            on_phase(ReviewStart(round_number=round_num))

        reviews = await run_review_phase(
            resolved, current_proposals, file_contents, review_reference,
            repo_root, round_num, backend, on_progress,
            previous_reviews=previous_reviews,
            display_names=display_names,
            normalizer=normalizer,
        )
        for r in reviews:
            total_usage += r.usage

        approvals = count_blocking_approvals(
            reviews, current_proposals,
            resolved.min_blocking_severity,
        )
        consensus_reached = approvals >= resolved.consensus_threshold

        rounds.append(IterationRound(
            round_number=round_num,
            reviews=reviews,
            consensus_reached=consensus_reached,
            approvals=approvals,
        ))

        if on_phase:
            on_phase(ReviewDone(
                round_number=round_num,
                reviews=reviews,
                consensus_threshold=resolved.consensus_threshold,
                blocking_approvals=approvals,
            ))

        # Track the version with the highest approval count
        if approvals >= best_approvals:
            best_approvals = approvals
            best_proposals = current_proposals
            best_round = round_num

        if consensus_reached:
            # Apply any non-blocking modifications before exiting.
            current_proposals = merge_proposals(
                current_proposals, reviews, locked_regions, file_contents,
            )
            consensus = True
            break

        # Detect stall -- no improvement for 2 consecutive rounds
        if detect_stall(rounds):
            contested = find_contested_edits(
                rounds, current_proposals,
            )
            if contested:
                if on_phase:
                    on_phase(ArbitrationStart(contested=contested))

                arb_results = await run_arbitration(
                    contested, file_contents, resolved, repo_root, backend,
                    on_progress,
                )
                for ar in arb_results:
                    total_usage += ar.usage

                # Apply arbitration results to current proposals
                arb_lookup: dict[tuple[str, str], str] = {
                    (ar.file, ar.original_text): ar.replacement_text
                    for ar in arb_results
                }
                current_proposals = [
                    dataclasses.replace(proposal, edits=[
                        dataclasses.replace(edit, replacement_text=arb_lookup[key])
                        if (key := (edit.file, edit.original_text)) in arb_lookup
                        else edit
                        for edit in proposal.edits
                    ])
                    for proposal in current_proposals
                ]

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
        # effectively approved -- build updated reviews for consensus counting.
        if locked_regions:
            updated_reviews: list[AgentReviewResponse] = []
            for review in reviews:
                if review.all_approved or review.error:
                    updated_reviews.append(review)
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
                if has_unlocked_mod:
                    updated_reviews.append(review)
                else:
                    updated_reviews.append(
                        dataclasses.replace(review, all_approved=True)
                    )
            reviews = updated_reviews

        # Re-validate after merge
        current_proposals = [
            dataclasses.replace(p, edits=validate_edits(p.edits, file_contents))
            for p in current_proposals
        ]

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

            arb_results = await run_arbitration(
                contested, file_contents, resolved, repo_root, backend,
                on_progress,
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
                 replacement_text=mt, rationale="merged from all agent proposals",
                 severity="major")
        for fp, mt in merge_result.merged_texts.items()
    ]

    # 9. Collect dissenting opinions if consensus was not reached
    dissents: list[Dissent] = []
    if not consensus and rounds and final_edits:
        best_round_reviews = rounds[best_round].reviews
        dissenting_agents = find_dissenting_agents(
            best_round_reviews, current_proposals,
            resolved.min_blocking_severity,
        )
        if dissenting_agents:
            dissents = await collect_dissents(
                resolved, dissenting_agents, current_proposals,
                file_contents, repo_root, backend, on_progress,
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
