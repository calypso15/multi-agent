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


def print_proposals_summary(proposals: list[AgentProposal]) -> None:
    """Print a summary of what each agent proposed."""
    console.print()
    console.print(Rule("[bold magenta]Propose Phase[/bold magenta]", style="magenta"))

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
                    key=lambda x: ["critical", "major", "minor", "suggestion"].index(x[0]),
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
    label = f"Review Round {round_number + 1}"
    if blocking_approvals is not None and blocking_approvals != approvals:
        status = (
            f"{blocking_approvals}/{total} blocking-approved, "
            f"{approvals}/{total} fully approved (need {consensus_threshold})"
        )
    else:
        status = f"{approvals}/{total} approved (need {consensus_threshold})"

    console.print()
    console.print(Rule(
        f"[bold {color}]{label}[/bold {color}]  [dim]{status}[/dim]",
        style=color,
    ))

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
    """Render a unified diff with syntax highlighting."""
    if not diff_text.strip():
        return
    console.print()
    console.print(Rule("[bold cyan]Proposed Changes[/bold cyan]", style="cyan"))
    console.print()
    syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=False, word_wrap=True)
    console.print(syntax)


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


def print_token_usage(usage: TokenUsage, duration: float) -> None:
    """Print a summary of token usage and cost."""
    console.print()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(justify="right")

    total_input = (
        usage.input_tokens
        + usage.cache_read_input_tokens
        + usage.cache_creation_input_tokens
    )
    table.add_row("Input tokens", f"{total_input:,}")
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
