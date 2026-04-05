"""Stall detection, arbitration of contested edits, and dissent collection."""

from __future__ import annotations

from typing import Callable

from multi_agent.agents import ARBITRATOR_PROMPT, build_agent_system_prompt
from multi_agent.backend import AgentBackend
from multi_agent.config import ResolvedRunConfig
from multi_agent.models import (
    AgentProposal,
    AgentReviewResponse,
    ArbitrationResult,
    ContestedEdit,
    Dissent,
    FileEdit,
    IterationRound,
    is_blocking_severity,
)


# --- Stall detection ---


def detect_stall(rounds: list[IterationRound]) -> bool:
    """Detect if the review loop has stalled (no improvement in the latest round)."""
    if len(rounds) < 2:
        return False
    return rounds[-1].approvals <= rounds[-2].approvals


def find_contested_edits(
    rounds: list[IterationRound],
    current_proposals: list[AgentProposal],
) -> list[ContestedEdit]:
    """Find edits that have been modified in 2+ consecutive rounds."""
    if len(rounds) < 2:
        return []

    modified_per_round: list[set[tuple[str, int]]] = []
    for rnd in rounds[-2:]:
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


# --- Arbitration ---


def build_arbitration_prompt(
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


async def run_arbitration(
    contested_edits: list[ContestedEdit],
    file_contents: dict[str, str],
    resolved: ResolvedRunConfig,
    repo_root: str,
    backend: AgentBackend,
    on_progress: Callable[[str, str], None] | None = None,
) -> list[ArbitrationResult]:
    """Run arbitration on contested edits using the backend."""
    # Use the first agent's review model for arbitration
    model = None
    for settings in resolved.agent_settings.values():
        model = settings.review_model or settings.propose_model
        if model:
            break

    # Use the first agent's timeout
    timeout = next(iter(resolved.agent_settings.values())).timeout_seconds

    results: list[ArbitrationResult] = []
    for i, contested in enumerate(contested_edits):
        key = f"arbitrator_{i}"
        prompt = build_arbitration_prompt(contested, file_contents)
        results.append(await _run_single_arbitrator(
            key, contested, prompt, backend, repo_root,
            timeout, on_progress,
            model=model,
        ))

    return results


async def _run_single_arbitrator(
    arb_name: str,
    contested: ContestedEdit,
    prompt: str,
    backend: AgentBackend,
    repo_root: str,
    timeout_seconds: int,
    on_progress: Callable[[str, str], None] | None = None,
    model: str | None = None,
) -> ArbitrationResult:
    """Run a single arbitration call via the backend (with full error handling)."""
    if on_progress:
        on_progress("arbitrator", f"resolving conflict in {contested.file}")

    raw = await backend.run_agent(
        arb_name, prompt, ARBITRATOR_PROMPT, repo_root,
        timeout_seconds, model=model, max_turns=1,
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


# --- Dissent collection ---


def build_dissent_prompt(
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


async def collect_dissents(
    resolved: ResolvedRunConfig,
    dissenting_agents: list[str],
    proposals: list[AgentProposal],
    file_contents: dict[str, str],
    repo_root: str,
    backend: AgentBackend,
    on_progress: Callable[[str, str], None] | None = None,
) -> list[Dissent]:
    """Collect brief dissenting opinions from agents that didn't approve."""
    prompt = build_dissent_prompt(proposals, file_contents)

    dissents: list[Dissent] = []
    for name in dissenting_agents:
        agent_cfg = resolved.agents.get(name)
        if not agent_cfg or not agent_cfg.enabled:
            continue
        settings = resolved.agent_settings[name]
        system_prompt = build_agent_system_prompt(name, "dissent", agent_cfg.system_prompt)
        result = await _run_single_dissenter(
            name, prompt, backend, system_prompt, repo_root,
            settings.timeout_seconds, on_progress,
            model=settings.review_model,
        )
        if result.opinion:
            dissents.append(result)

    return dissents


async def _run_single_dissenter(
    agent_name: str,
    prompt: str,
    backend: AgentBackend,
    system_prompt: str,
    repo_root: str,
    timeout_seconds: int,
    on_progress: Callable[[str, str], None] | None = None,
    model: str | None = None,
) -> Dissent:
    """Run a single agent to collect a dissenting opinion."""
    label = f"dissenting ({model})" if model else "dissenting"
    raw = await backend.run_agent(
        agent_name, prompt, system_prompt, repo_root,
        timeout_seconds, model=model, max_turns=1,
        on_progress=on_progress,
        progress_label=label, report_tool_use=False,
    )
    if raw.error:
        return Dissent(
            agent_name=agent_name, opinion="",
            duration_seconds=raw.duration_seconds, usage=raw.usage,
            turns_taken=raw.turns_taken, tool_usage=raw.tool_usage,
        )

    opinion = raw.output.get("opinion", "")
    if on_progress:
        on_progress(agent_name, "done \u2014 dissent recorded")

    return Dissent(
        agent_name=agent_name, opinion=opinion,
        duration_seconds=raw.duration_seconds, usage=raw.usage,
        turns_taken=raw.turns_taken, tool_usage=raw.tool_usage,
    )


def find_dissenting_agents(
    reviews: list[AgentReviewResponse],
    proposals: list[AgentProposal],
    min_blocking_severity: str,
) -> list[str]:
    """Identify agents that objected to blocking-severity edits."""
    proposal_map = {p.agent_name: p for p in proposals}
    dissenting: list[str] = []
    for r in reviews:
        if r.error is not None or r.all_approved:
            continue
        has_blocking_objection = False
        for pr in r.proposal_reviews:
            if pr.verdict != "MODIFY":
                continue
            prop = proposal_map.get(pr.original_agent)
            if prop is None or pr.edit_index >= len(prop.edits):
                has_blocking_objection = True
                break
            if is_blocking_severity(
                prop.edits[pr.edit_index].severity,
                min_blocking_severity,
            ):
                has_blocking_objection = True
                break
        if has_blocking_objection:
            dissenting.append(r.agent_name)
    return dissenting
