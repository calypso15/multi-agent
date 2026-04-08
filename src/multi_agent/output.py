"""Rich terminal output formatting for review results."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Sequence, Union

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from multi_agent.models import SEVERITY_ORDER

if TYPE_CHECKING:
    from multi_agent.config import AgentConfig
    from multi_agent.models import (
        AgentProposal,
        AgentReviewResponse,
        ArbitrationResult,
        ContestedEdit,
        Dissent,
        TokenUsage,
    )

_COLOR_PALETTE = [
    "cyan", "magenta", "yellow", "green", "blue",
    "red", "bright_cyan", "bright_magenta",
]

_DISPLAY_NAMES: dict[str, str] = {}
_COLORS: dict[str, str] = {}
_VERBOSE = False


def set_verbose(value: bool) -> None:
    """Enable or disable verbose output."""
    global _VERBOSE
    _VERBOSE = value


def is_verbose() -> bool:
    """Return whether verbose output is enabled."""
    return _VERBOSE


_DETAIL = False


def set_detail(value: bool) -> None:
    """Enable or disable detailed edit output."""
    global _DETAIL
    _DETAIL = value


def is_detail() -> bool:
    """Return whether detailed edit output is enabled."""
    return _DETAIL


def init_agent_styles(agents: dict[str, AgentConfig]) -> None:
    """Initialize display names and colors from agent config.

    Call once after loading config, before any output functions.
    """
    from multi_agent.config import get_display_name

    _DISPLAY_NAMES.clear()
    _COLORS.clear()
    for i, (key, cfg) in enumerate(agents.items()):
        _DISPLAY_NAMES[key] = get_display_name(key, cfg)
        _COLORS[key] = _COLOR_PALETTE[i % len(_COLOR_PALETTE)]


def _agent_style(agent_name: str) -> str:
    """Get the color style for an agent."""
    return _COLORS.get(agent_name, "white")


def _truncate_text(text: str, max_lines: int = 20) -> str:
    """Truncate text to *max_lines*, appending a count of omitted lines."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    omitted = len(lines) - max_lines
    return "\n".join(lines[:max_lines]) + f"\n[...{omitted} more lines]"


console = Console(stderr=True)


def _print_errors(
    items: Sequence[Union[AgentProposal, AgentReviewResponse]],
) -> None:
    """Print error details for any agents that failed."""
    for item in items:
        if item.error:
            display = _DISPLAY_NAMES.get(item.agent_name, item.agent_name)
            color = _agent_style(item.agent_name)
            console.print(f"\n  [{color}]{display} error:[/{color}] [red]{item.error}[/red]")


def print_header(
    files: list[str],
    ref_count: int,
    ref_size_kb: float,
    uncommitted_ref: int = 0,
    task: str | None = None,
) -> None:
    """Print the review header showing what's being reviewed."""
    file_list = ", ".join(files)
    task_label = f"  [bold cyan]Task: {task}[/bold cyan]" if task else ""
    lines = [f"Reviewing {len(files)} file(s): {file_list}{task_label}"]
    if ref_count > 0:
        ref_line = f"Reference context: {ref_count} file(s) ({ref_size_kb:.0f} KB)"
        if uncommitted_ref > 0:
            ref_line += (
                f" [yellow]({uncommitted_ref} uncommitted file(s) "
                "not included — commit to add as context)[/yellow]"
            )
        lines.append(ref_line)
    elif uncommitted_ref > 0:
        lines.append(
            f"Reference context: none [yellow]({uncommitted_ref} uncommitted "
            "file(s) found — commit them to use as context)[/yellow]"
        )
    else:
        lines.append("Reference context: none")

    console.print(Panel(
        "\n".join(lines),
        title="Multi-Agent Review",
        border_style="blue",
    ))


def print_progress(agent_name: str, status: str) -> None:
    """Print a status update for an agent."""
    display = _DISPLAY_NAMES.get(agent_name, agent_name)
    color = _agent_style(agent_name)
    console.print(f"  [{color}]{display}:[/{color}] {status}", highlight=False)


def print_no_files() -> None:
    """Print message when no reviewable files are staged."""
    console.print("[dim]No text files staged for review. Skipping.[/dim]")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


# --- Iteration loop output ---


def _agent_table(rows: list[tuple[str, str, Text, float]]) -> Table:
    """Build a compact table of agent results.

    rows: list of (display_name, agent_key, status_text, duration)
    """
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 2),
        pad_edge=False,
    )
    table.add_column("Agent", min_width=20)
    table.add_column("Status")
    table.add_column("Time", justify="right", style="dim")

    for name, agent_key, status, duration in rows:
        color = _agent_style(agent_key)
        table.add_row(f"[{color}]{name}[/{color}]", status, f"{duration:.1f}s")

    return table


def print_propose_start() -> None:
    """Print the Propose Phase header before agents run."""
    console.print()
    console.print(Rule("[bold magenta]Propose Phase[/bold magenta]", style="magenta"))


def print_review_start(round_number: int) -> None:
    """Print the Review Round header before agents run."""
    label = f"Review Round {round_number + 1}"
    console.print()
    console.print(Rule(f"[bold cyan]{label}[/bold cyan]", style="cyan"))


def print_proposals_summary(proposals: list[AgentProposal]) -> None:
    """Print a summary of what each agent proposed."""
    console.print()

    rows = []
    for proposal in proposals:
        display = _DISPLAY_NAMES.get(proposal.agent_name, proposal.agent_name)

        if proposal.error:
            status = Text("ERROR", style="bold red")
        elif not proposal.edits:
            status = Text("no edits", style="dim")
        else:
            from collections import Counter
            sev_counts = Counter(e.severity for e in proposal.edits)
            sev_parts = ", ".join(
                f"{cnt} {sev}" for sev, cnt in sorted(
                    sev_counts.items(),
                    key=lambda x: SEVERITY_ORDER.index(x[0]),
                )
            )
            files = sorted({e.file for e in proposal.edits})
            status = Text(f"{len(proposal.edits)} edit(s) ", style="bold cyan")
            status.append(f"[{sev_parts}] ", style="dim")
            status.append(f"({', '.join(files)})", style="dim")

        rows.append((display, proposal.agent_name, status, proposal.duration_seconds))

    console.print(_agent_table(rows))
    _print_errors(proposals)

    total_edits = sum(len(p.edits) for p in proposals)
    console.print(
        f"\n  [dim]{total_edits} total edit(s) proposed[/dim]"
    )


def print_proposal_details(proposals: list[AgentProposal]) -> None:
    """Print full edit content for each agent's proposal."""
    for proposal in proposals:
        if proposal.error or not proposal.edits:
            continue
        display = _DISPLAY_NAMES.get(proposal.agent_name, proposal.agent_name)
        color = _agent_style(proposal.agent_name)

        parts: list[str] = []
        if proposal.summary:
            parts.append(f"[dim]Summary:[/dim] {proposal.summary}\n")

        for i, edit in enumerate(proposal.edits):
            parts.append(f"[bold]Edit {i}[/bold] ({edit.severity}) \u2014 {edit.file}")
            if edit.rationale:
                parts.append(f"[dim]Rationale:[/dim] {edit.rationale}")
            parts.append(f"[dim]\u2500\u2500 original \u2500\u2500[/dim]")
            parts.append(_truncate_text(edit.original_text))
            parts.append(f"[dim]\u2500\u2500 replacement \u2500\u2500[/dim]")
            parts.append(_truncate_text(edit.replacement_text))
            parts.append("")

        console.print(Panel(
            "\n".join(parts).rstrip(),
            title=f"[{color}]{display} proposals[/{color}]",
            border_style=color,
            padding=(0, 2),
        ))


def print_review_details(
    reviews: list[AgentReviewResponse],
    proposals: list[AgentProposal],
) -> None:
    """Print full modification content for each reviewer's changes."""
    proposal_map = {p.agent_name: p for p in proposals}

    for review in reviews:
        if review.all_approved or review.error:
            continue
        mods = [r for r in review.proposal_reviews if r.verdict == "MODIFY"]
        if not mods:
            continue

        display = _DISPLAY_NAMES.get(review.agent_name, review.agent_name)
        color = _agent_style(review.agent_name)

        parts: list[str] = []
        if review.summary:
            parts.append(f"[dim]Summary:[/dim] {review.summary}\n")

        for mod in mods:
            orig_display = _DISPLAY_NAMES.get(mod.original_agent, mod.original_agent)
            orig_color = _agent_style(mod.original_agent)

            # Look up the original edit for file name and replacement text
            prop = proposal_map.get(mod.original_agent)
            if prop and mod.edit_index < len(prop.edits):
                edit = prop.edits[mod.edit_index]
                file_label = f" \u2014 {edit.file}"
                original_replacement = edit.replacement_text
            else:
                file_label = ""
                original_replacement = None

            parts.append(
                f"[bold]Edit {mod.edit_index}[/bold] from "
                f"[{orig_color}]{orig_display}[/{orig_color}]{file_label}"
            )
            if mod.rationale:
                parts.append(f"[dim]Rationale:[/dim] {mod.rationale}")

            if original_replacement is not None:
                parts.append(f"[dim]\u2500\u2500 original replacement \u2500\u2500[/dim]")
                parts.append(_truncate_text(original_replacement))
            if mod.modified_replacement is not None:
                parts.append(f"[dim]\u2500\u2500 modified replacement \u2500\u2500[/dim]")
                parts.append(_truncate_text(mod.modified_replacement))
            parts.append("")

        console.print(Panel(
            "\n".join(parts).rstrip(),
            title=f"[{color}]{display} review details[/{color}]",
            border_style=color,
            padding=(0, 2),
        ))


def print_review_round(
    round_number: int,
    reviews: list[AgentReviewResponse],
    consensus_threshold: int,
    blocking_approvals: int | None = None,
) -> None:
    """Print the results of a review round."""
    from multi_agent.models import count_approvals

    approvals = count_approvals(reviews)
    total = len(reviews)
    effective = blocking_approvals if blocking_approvals is not None else approvals
    reached = effective >= consensus_threshold

    color = "green" if reached else "yellow"
    if blocking_approvals is not None and blocking_approvals != approvals:
        status = (
            f"{blocking_approvals}/{total} blocking-approved, "
            f"{approvals}/{total} fully approved (need {consensus_threshold})"
        )
    else:
        status = f"{approvals}/{total} approved (need {consensus_threshold})"

    console.print()
    console.print(f"  [{color} bold]{status}[/{color} bold]")

    rows = []
    for review in reviews:
        display = _DISPLAY_NAMES.get(review.agent_name, review.agent_name)

        if review.error:
            status_text = Text("ERROR", style="bold red")
        elif review.all_approved:
            status_text = Text("APPROVED ALL", style="bold green")
        else:
            mod_count = sum(
                1 for r in review.proposal_reviews if r.verdict == "MODIFY"
            )
            status_text = Text(f"MODIFIED {mod_count}", style="bold yellow")

        rows.append((display, review.agent_name, status_text, review.duration_seconds))

    console.print(_agent_table(rows))
    _print_errors(reviews)

    # Show modification details when not all approved
    for review in reviews:
        if review.all_approved or review.error or not review.proposal_reviews:
            continue
        display = _DISPLAY_NAMES.get(review.agent_name, review.agent_name)
        color = _agent_style(review.agent_name)
        mods = [r for r in review.proposal_reviews if r.verdict == "MODIFY"]
        if mods:
            console.print(f"\n  [{color}]{display} modifications:[/{color}]")
            for mod in mods:
                orig_display = _DISPLAY_NAMES.get(mod.original_agent, mod.original_agent)
                orig_color = _agent_style(mod.original_agent)
                console.print(
                    f"    edit {mod.edit_index} "
                    f"from [{orig_color}]{orig_display}[/{orig_color}]: {mod.rationale}"
                )


def print_final_diff(diff_text: str) -> None:
    """Render a unified diff with colored lines in a bordered panel."""
    if not diff_text.strip():
        return
    console.print()
    body = Text()
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            body.append(line + "\n", style="bold")
        elif line.startswith("@@"):
            body.append(line + "\n", style="cyan")
        elif line.startswith("+"):
            body.append(line + "\n", style="green")
        elif line.startswith("-"):
            body.append(line + "\n", style="red")
        else:
            body.append(line + "\n")
    # Strip trailing newline to avoid blank line at bottom of panel
    body.rstrip()
    console.print(Panel(
        body,
        title="[bold cyan]Proposed Changes[/bold cyan]",
        border_style="cyan",
        padding=(0, 1),
    ))


def print_edit_list(
    numbered_edits: list[tuple[int, str, "FileEdit"]],
) -> None:
    """Print numbered edits with full content for selection.

    numbered_edits: list of (1-based index, agent_display_name, FileEdit).
    """
    console.print()
    for num, agent_display, edit in numbered_edits:
        header = Text.assemble(
            (f"{num}. ", "bold"),
            (f"[{edit.severity}] ", ""),
            (edit.file, "dim"),
            ("  ", ""),
            (f"({agent_display})", "dim"),
        )
        body = Text()
        if edit.rationale:
            body.append("Rationale: ", style="dim")
            body.append(edit.rationale + "\n", style="cyan")
        body.append("\u2500\u2500 original \u2500\u2500\n", style="red dim")
        body.append(_truncate_text(edit.original_text) + "\n", style="red")
        body.append("\u2500\u2500 replacement \u2500\u2500\n", style="green dim")
        body.append(_truncate_text(edit.replacement_text), style="green")
        console.print(Panel(
            body,
            title=header,
            title_align="left",
            border_style="dim",
            padding=(0, 2),
        ))


def prompt_edit_selection(total: int) -> set[int] | None:
    """Prompt the user to select which edits to apply.

    Returns a set of 1-based edit indices, or None to reject all.
    Returns the full set {1..total} for "all".
    """
    console.print()
    if not sys.stdin.isatty():
        console.print("[dim]No TTY available — skipping confirmation.[/dim]")
        return None
    try:
        response = console.input(
            "[bold]Apply: \\[a]ll, \\[n]one, or edit numbers (e.g. 1,3): [/bold]"
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    if response in ("a", "all", "y", "yes"):
        return set(range(1, total + 1))
    if response in ("n", "none", "no", ""):
        return None

    # Parse comma/space-separated numbers
    selected: set[int] = set()
    for part in response.replace(",", " ").split():
        try:
            num = int(part)
            if 1 <= num <= total:
                selected.add(num)
        except ValueError:
            continue
    return selected if selected else None


def print_confirmation_prompt() -> bool:
    """Ask the user whether to apply changes. Returns True if confirmed."""
    console.print()
    if not sys.stdin.isatty():
        console.print("[dim]No TTY available — skipping confirmation.[/dim]")
        return False
    try:
        response = console.input("[bold]Apply these changes? [y/N] [/bold]")
        return response.strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def print_no_edits() -> None:
    """Print message when agents proposed no changes."""
    console.print()
    console.print(Panel(
        "All agents found the content acceptable. No changes proposed.",
        title="NO CHANGES",
        border_style="green",
    ))


def print_iteration_exhausted(
    rounds_run: int,
    max_rounds: int,
    approvals: int,
    total: int,
    best_round: int = -1,
    stalled: bool = False,
) -> None:
    """Warn when consensus was not reached."""
    console.print()
    if stalled:
        reason = (
            f"Stalled after {rounds_run} round(s) "
            f"(no improvement, max was {max_rounds})."
        )
    else:
        reason = f"All {max_rounds} rounds completed without full consensus."

    round_note = ""
    if best_round >= 0:
        round_note = (
            f"\nUsing proposals from round {best_round + 1} "
            f"({approvals}/{total} approved) — the highest approval seen."
        )
    console.print(Panel(
        reason + round_note + "\nReview carefully before accepting.",
        title="CONSENSUS NOT REACHED",
        border_style="yellow",
    ))


def print_iteration_success(approvals: int, total: int) -> None:
    """Print success when iteration consensus is reached."""
    console.print()
    console.print(Panel(
        f"Consensus reached ({approvals}/{total}). "
        "All agents approve the proposed changes.",
        title="CONSENSUS",
        border_style="green",
    ))


def print_arbitration_start(contested: list[ContestedEdit]) -> None:
    """Print that arbitration is starting."""
    console.print()
    console.print(Rule(
        f"[bold yellow]Arbitration[/bold yellow]  "
        f"[dim]({len(contested)} contested edit(s))[/dim]",
        style="yellow",
    ))
    for ce in contested:
        agent_names = []
        for key in ce.versions:
            display = _DISPLAY_NAMES.get(key, key)
            color = _agent_style(key)
            agent_names.append(f"[{color}]{display}[/{color}]")
        console.print(f"  {ce.file} — competing versions from: {', '.join(agent_names)}")


def print_arbitration_done(results: list[ArbitrationResult]) -> None:
    """Print arbitration results."""
    for ar in results:
        console.print(f"\n  [green]Resolved:[/green] {ar.file}")
        console.print(f"  [dim]{ar.rationale}[/dim]")


def print_dissents(dissents: list[Dissent]) -> None:
    """Print dissenting opinions from agents that didn't approve."""
    if not dissents:
        return
    console.print()
    console.print(Rule("[bold red]Dissenting Opinions[/bold red]", style="red"))
    for dissent in dissents:
        display = _DISPLAY_NAMES.get(dissent.agent_name, dissent.agent_name)
        color = _agent_style(dissent.agent_name)
        console.print(f"\n  [{color} bold]{display}:[/{color} bold]")
        console.print(f"  {dissent.opinion}")
    console.print()


def print_answer(answer_text: str) -> None:
    """Display a consensus answer in a prominent panel."""
    from rich.markdown import Markdown

    console.print()
    console.print(Rule("[bold cyan]Answer[/bold cyan]", style="cyan"))
    console.print()
    console.print(Panel(
        Markdown(answer_text),
        border_style="green",
        padding=(1, 2),
    ))


def print_changes_applied(files: list[str]) -> None:
    """Print confirmation that changes were applied."""
    console.print()
    console.print(
        f"[green]Changes applied to {len(files)} file(s):[/green] "
        + ", ".join(files)
    )


def print_resolved_config(resolved) -> None:
    """Print resolved configuration details for each agent."""
    from multi_agent.config import get_display_name

    console.print()
    console.print(Rule("[bold blue]Resolved Config[/bold blue]", style="blue"))

    # Run-level settings
    run_table = Table(show_header=False, box=None, padding=(0, 2))
    run_table.add_column(style="dim")
    run_table.add_column()
    run_table.add_row("Max rounds", str(resolved.max_rounds))
    run_table.add_row("Consensus threshold", str(resolved.consensus_threshold))
    run_table.add_row("Min severity", resolved.min_severity)
    run_table.add_row("Min blocking severity", resolved.min_blocking_severity)
    run_table.add_row("Backend", resolved.backend)
    console.print(run_table)

    # Per-agent settings
    for name, settings in resolved.agent_settings.items():
        display = _DISPLAY_NAMES.get(name, name)
        color = _agent_style(name)
        console.print(f"\n  [{color} bold]{display}[/{color} bold]")

        agent_table = Table(show_header=False, box=None, padding=(0, 2))
        agent_table.add_column(style="dim")
        agent_table.add_column()

        if settings.weight != 1:
            agent_table.add_row("    Weight", str(settings.weight))

        propose_model = settings.propose_model or "default"
        review_model = settings.review_model or "default"
        if propose_model == review_model:
            agent_table.add_row("    Model", propose_model)
        else:
            agent_table.add_row("    Propose model", propose_model)
            agent_table.add_row("    Review model", review_model)

        propose_turns = str(settings.propose_max_turns) if settings.propose_max_turns else "unlimited"
        review_turns = str(settings.review_max_turns) if settings.review_max_turns else "unlimited"
        if propose_turns == review_turns:
            agent_table.add_row("    Max turns", propose_turns)
        else:
            agent_table.add_row("    Propose max turns", propose_turns)
            agent_table.add_row("    Review max turns", review_turns)

        agent_table.add_row("    Timeout", f"{settings.timeout_seconds}s")
        agent_table.add_row("    Tools", ", ".join(sorted(settings.allowed_tools)) or "none")
        if settings.file_patterns:
            agent_table.add_row("    File patterns", ", ".join(settings.file_patterns))
        if settings.reference_directories:
            agent_table.add_row("    Reference dirs", ", ".join(settings.reference_directories))

        console.print(agent_table)


def print_agent_verbose_stats(
    agent_name: str,
    turns_taken: int,
    tool_usage: dict[str, int],
    usage: TokenUsage,
) -> None:
    """Print verbose per-agent stats: turns, tools, tokens."""
    display = _DISPLAY_NAMES.get(agent_name, agent_name)
    color = _agent_style(agent_name)

    parts: list[str] = []

    # Turns
    parts.append(f"turns: {turns_taken}")

    # Tool usage
    if tool_usage:
        tool_parts = ", ".join(f"{t} x{c}" for t, c in sorted(tool_usage.items()))
        parts.append(f"tools: {tool_parts}")

    # Token breakdown
    token_parts = [f"in: {usage.total_input_tokens:,}", f"out: {usage.output_tokens:,}"]
    if usage.cache_read_input_tokens:
        token_parts.append(f"cache read: {usage.cache_read_input_tokens:,}")
    if usage.cache_creation_input_tokens:
        token_parts.append(f"cache write: {usage.cache_creation_input_tokens:,}")
    parts.append(f"tokens: {', '.join(token_parts)}")

    console.print(f"    [{color}]{display}:[/{color}] [dim]{' | '.join(parts)}[/dim]")


def print_token_usage(usage: TokenUsage, duration: float) -> None:
    """Print a summary of token usage and cost."""
    console.print()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(justify="right")

    table.add_row("Input tokens", f"{usage.total_input_tokens:,}")
    if usage.cache_read_input_tokens:
        table.add_row("  Cache read", f"{usage.cache_read_input_tokens:,}")
    if usage.cache_creation_input_tokens:
        table.add_row("  Cache write", f"{usage.cache_creation_input_tokens:,}")
    if usage.input_tokens:
        table.add_row("  Uncached", f"{usage.input_tokens:,}")
    table.add_row("Output tokens", f"{usage.output_tokens:,}")
    table.add_row("Total cost", f"${usage.cost_usd:.4f}")
    table.add_row("Duration", f"{duration:.1f}s")

    console.print(Panel(table, title="Usage", border_style="dim"))
