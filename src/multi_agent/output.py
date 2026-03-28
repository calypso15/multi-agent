"""Rich terminal output formatting for review results."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from multi_agent.agents import AGENT_DISPLAY_NAMES
from multi_agent.consensus import AgentReview, ConsensusResult

console = Console(stderr=True)


def print_header(files: list[str], canon_count: int, canon_size_kb: float) -> None:
    """Print the review header showing what's being reviewed."""
    file_list = ", ".join(files)
    lines = [f"Reviewing {len(files)} file(s): {file_list}"]
    if canon_count > 0:
        lines.append(f"Canon context: {canon_count} files ({canon_size_kb:.0f} KB)")
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
