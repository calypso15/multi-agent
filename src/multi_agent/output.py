"""Rich terminal output formatting for review results."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from multi_agent.agents import AGENT_DISPLAY_NAMES
from multi_agent.consensus import (
    AgentProposal,
    AgentReview,
    AgentReviewResponse,
    ConsensusResult,
    IterationResult,
    TokenUsage,
)

console = Console(stderr=True)


def print_header(
    files: list[str],
    canon_count: int,
    canon_size_kb: float,
    uncommitted_canon: int = 0,
) -> None:
    """Print the review header showing what's being reviewed."""
    file_list = ", ".join(files)
    lines = [f"Reviewing {len(files)} file(s): {file_list}"]
    if canon_count > 0:
        canon_line = f"Canon context: {canon_count} file(s) ({canon_size_kb:.0f} KB)"
        if uncommitted_canon > 0:
            canon_line += (
                f" [yellow]({uncommitted_canon} uncommitted file(s) "
                "not included — commit to add as context)[/yellow]"
            )
        lines.append(canon_line)
    elif uncommitted_canon > 0:
        lines.append(
            f"Canon context: none [yellow]({uncommitted_canon} uncommitted "
            "file(s) found — commit them to use as context)[/yellow]"
        )
    else:
        lines.append("Canon context: none (first contribution)")

    console.print(Panel(
        "\n".join(lines),
        title="Multi-Agent Fiction Review",
        border_style="blue",
    ))


def print_progress(agent_name: str, status: str) -> None:
    """Print a status update for an agent."""
    display = AGENT_DISPLAY_NAMES.get(agent_name, agent_name)
    console.print(f"  [dim]{display}:[/dim] {status}", highlight=False)


def print_results(result: ConsensusResult) -> None:
    """Print the full consensus result."""
    # Agent verdict summary
    console.print()
    for review in result.reviews:
        display = AGENT_DISPLAY_NAMES.get(review.agent_name, review.agent_name)
        name_text = f"  {display:<22}"

        if review.error:
            verdict_text = Text("ERROR", style="bold red")
        elif review.verdict == "APPROVE":
            verdict_text = Text("APPROVE", style="bold green")
        else:
            verdict_text = Text("REQUEST_CHANGES", style="bold red")

        time_text = f"  {review.duration_seconds:.1f}s"

        line = Text(name_text)
        line.append(verdict_text)
        line.append(time_text, style="dim")
        console.print(line)

    # Verdict panel
    console.print()
    approvals = sum(1 for r in result.reviews if r.verdict == "APPROVE")
    total = len(result.reviews)

    if result.approved:
        console.print(Panel(
            f"Consensus reached ({approvals}/{total}). Commit may proceed.",
            title="APPROVED",
            border_style="green",
        ))
    else:
        console.print(Panel(
            f"Consensus not reached ({approvals}/{total}). "
            "Address the issues below and try again.\n"
            "Use [bold]git commit --no-verify[/bold] to bypass.",
            title="BLOCKED",
            border_style="red",
        ))

    # Per-agent issues
    for review in result.reviews:
        if not review.issues and not review.error:
            continue

        display = AGENT_DISPLAY_NAMES.get(review.agent_name, review.agent_name)

        if review.error:
            console.print(f"\n[bold red]{display} - Error:[/bold red]")
            console.print(f"  {review.error}")
            continue

        if review.issues:
            console.print(f"\n[bold]{display} Issues:[/bold]")
            if review.summary:
                console.print(f"  [dim]{review.summary}[/dim]\n")

            for issue in review.issues:
                severity_colors = {
                    "critical": "bold red",
                    "major": "red",
                    "minor": "yellow",
                    "suggestion": "cyan",
                }
                style = severity_colors.get(issue.severity, "white")
                console.print(f"  [{style}][{issue.severity}][/{style}]", end="")
                if issue.file:
                    console.print(f" [dim]{issue.file}[/dim]")
                else:
                    console.print()

                if issue.quote:
                    console.print(f"    [italic]\"{issue.quote}\"[/italic]")
                console.print(f"    {issue.issue}")
                console.print(f"    [green]Suggestion:[/green] {issue.suggestion}")
                console.print()

    # Duration summary
    console.print(
        f"[dim]Duration: {result.total_duration_seconds:.1f}s[/dim]"
    )


def print_no_files() -> None:
    """Print message when no reviewable files are staged."""
    console.print("[dim]No text files staged for review. Skipping.[/dim]")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


# --- Iteration loop output ---


def print_proposals_summary(proposals: list[AgentProposal]) -> None:
    """Print a summary of what each agent proposed."""
    console.print()
    console.print(Panel("Propose Phase Complete", border_style="blue"))

    for proposal in proposals:
        display = AGENT_DISPLAY_NAMES.get(proposal.agent_name, proposal.agent_name)
        name_text = f"  {display:<22}"

        if proposal.error:
            status = Text("ERROR", style="bold red")
        elif not proposal.edits:
            status = Text("no edits", style="dim")
        else:
            files = sorted({e.file for e in proposal.edits})
            status = Text(
                f"{len(proposal.edits)} edit(s) ({', '.join(files)})",
                style="bold cyan",
            )

        time_text = f"  {proposal.duration_seconds:.1f}s"

        line = Text(name_text)
        line.append(status)
        line.append(time_text, style="dim")
        console.print(line)


def print_review_round(
    round_number: int,
    reviews: list[AgentReviewResponse],
    consensus_threshold: int,
) -> None:
    """Print the results of a review round."""
    console.print()
    approvals = sum(1 for r in reviews if r.all_approved and r.error is None)
    total = len(reviews)
    console.print(Panel(
        f"Review Round {round_number + 1}  ({approvals}/{total} approved, "
        f"need {consensus_threshold})",
        border_style="blue",
    ))

    for review in reviews:
        display = AGENT_DISPLAY_NAMES.get(review.agent_name, review.agent_name)
        name_text = f"  {display:<22}"

        if review.error:
            status = Text("ERROR", style="bold red")
        elif review.all_approved:
            status = Text("APPROVED ALL", style="bold green")
        else:
            mod_count = sum(
                1 for r in review.proposal_reviews if r.verdict == "MODIFY"
            )
            status = Text(f"MODIFIED {mod_count}", style="bold yellow")

        time_text = f"  {review.duration_seconds:.1f}s"

        line = Text(name_text)
        line.append(status)
        line.append(time_text, style="dim")
        console.print(line)


def print_final_diff(diff_text: str) -> None:
    """Render a unified diff with syntax highlighting."""
    if not diff_text.strip():
        return
    console.print()
    console.print(Panel("Proposed Changes", border_style="cyan"))
    syntax = Syntax(diff_text, "diff", theme="monokai", line_numbers=False)
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
    max_rounds: int,
    approvals: int,
    total: int,
) -> None:
    """Warn when consensus was not reached within max rounds."""
    console.print()
    console.print(Panel(
        f"Maximum rounds ({max_rounds}) reached without full consensus "
        f"({approvals}/{total} approved).\n"
        "Showing best available proposals. Review carefully before accepting.",
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
