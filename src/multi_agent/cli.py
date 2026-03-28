"""CLI entry point using Click."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from multi_agent.config import load_config
from multi_agent.context import find_git_root
from multi_agent.output import (
    console,
    print_changes_applied,
    print_confirmation_prompt,
    print_error,
    print_final_diff,
    print_header,
    print_iteration_exhausted,
    print_iteration_success,
    print_no_edits,
    print_no_files,
    print_progress,
    print_proposals_summary,
    print_results,
    print_review_round,
    print_token_usage,
)


@click.group()
@click.option("--repo", "repo_path", type=click.Path(exists=True), default=None,
              help="Path to a git repository to review. Defaults to current directory.")
@click.pass_context
def main(ctx: click.Context, repo_path: str | None) -> None:
    """Multi-agent consensus reviewer for collaborative fiction."""
    ctx.ensure_object(dict)
    ctx.obj["repo_path"] = repo_path


def _resolve_repo(ctx: click.Context) -> Path:
    """Resolve the target git repo root from --repo or cwd."""
    repo_path = ctx.obj.get("repo_path")
    start = Path(repo_path) if repo_path else None
    return find_git_root(start)


def _run_iteration_and_present(
    config,
    repo_root: Path,
    target_files: list[str] | None,
    files_display: list[str],
    canon_count: int,
    canon_size_kb: float,
    uncommitted_canon: int,
    dry_run: bool,
    hook_mode: bool,
) -> int:
    """Run the iteration loop and present results. Returns exit code."""
    from multi_agent.consensus import run_iteration_loop
    from multi_agent.context import apply_edits, build_diff_preview

    print_header(files_display, canon_count, canon_size_kb, uncommitted_canon)

    result = asyncio.run(run_iteration_loop(
        config, str(repo_root), target_files=target_files,
        on_progress=print_progress,
    ))

    # Show proposal summary
    if result.proposals:
        print_proposals_summary(result.proposals)

    # Show review rounds
    for rnd in result.rounds:
        print_review_round(
            rnd.round_number, rnd.reviews,
            config.general.consensus_threshold,
        )

    # No edits?
    if not result.final_edits:
        print_no_edits()
        print_token_usage(result.total_usage, result.total_duration_seconds)
        return 0

    # Build file contents for diff preview
    file_contents: dict[str, str] = {}
    for f in result.files_reviewed:
        path = repo_root / f
        if path.is_file():
            file_contents[f] = path.read_text()

    diff_text = build_diff_preview(result.final_edits, file_contents)

    # Consensus status
    if result.consensus_reached:
        last_round = result.rounds[-1] if result.rounds else None
        if last_round:
            approvals = sum(
                1 for r in last_round.reviews
                if r.all_approved and r.error is None
            )
            print_iteration_success(approvals, len(last_round.reviews))
        else:
            print_iteration_success(0, 0)
    else:
        last_round = result.rounds[-1] if result.rounds else None
        approvals = 0
        total = 0
        if last_round:
            approvals = sum(
                1 for r in last_round.reviews
                if r.all_approved and r.error is None
            )
            total = len(last_round.reviews)
        print_iteration_exhausted(config.general.max_rounds, approvals, total)

    # Show diff
    print_final_diff(diff_text)
    print_token_usage(result.total_usage, result.total_duration_seconds)

    if dry_run:
        console.print("[dim]Dry run — no changes applied.[/dim]")
        return 0

    # Confirm and apply
    if hook_mode and not sys.stdin.isatty():
        # Non-interactive hook: don't apply, exit 1 so user can review
        console.print(
            "[yellow]Changes proposed but cannot confirm in non-interactive mode.[/yellow]\n"
            "Run [bold]multi-agent review-files[/bold] to review and apply interactively."
        )
        return 1

    if print_confirmation_prompt():
        modified = apply_edits(repo_root, result.final_edits)
        print_changes_applied(modified)
        if hook_mode:
            console.print(
                "[yellow]Changes applied to working tree. "
                "Review the modifications, stage them, and commit again.[/yellow]"
            )
            return 1  # exit 1 so the current commit is aborted
        return 0
    else:
        console.print("[dim]Changes not applied.[/dim]")
        return 1 if hook_mode else 0


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Path to multi_agent.toml config file.")
@click.option("--hook-mode", is_flag=True, hidden=True,
              help="Running as a git pre-commit hook.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show proposed changes without applying.")
@click.option("--max-rounds", type=int, default=None,
              help="Override max iteration rounds.")
@click.pass_context
def review(
    ctx: click.Context,
    config_path: str | None,
    hook_mode: bool,
    dry_run: bool,
    max_rounds: int | None,
) -> None:
    """Review staged files and propose changes via consensus."""
    try:
        repo_root = _resolve_repo(ctx)
    except Exception:
        print_error("Not in a git repository.")
        sys.exit(1)

    config = load_config(
        Path(config_path) if config_path else None,
        search_from=repo_root,
    )
    if max_rounds is not None:
        config.general.max_rounds = max_rounds

    from multi_agent.context import (
        count_uncommitted_canon,
        get_staged_files,
        load_canon,
    )

    staged = get_staged_files(repo_root, config.general.file_patterns)
    if not staged:
        print_no_files()
        sys.exit(0)

    canon = load_canon(
        repo_root,
        config.general.canon_directories,
        config.general.file_patterns,
        config.general.max_canon_size_kb,
    )
    canon_size_kb = sum(len(v.encode()) for v in canon.values()) / 1024
    uncommitted = count_uncommitted_canon(
        repo_root,
        config.general.canon_directories,
        config.general.file_patterns,
        set(canon.keys()),
    )

    exit_code = _run_iteration_and_present(
        config=config,
        repo_root=repo_root,
        target_files=None,  # use staged files
        files_display=[str(f) for f in staged],
        canon_count=len(canon),
        canon_size_kb=canon_size_kb,
        uncommitted_canon=uncommitted,
        dry_run=dry_run,
        hook_mode=hook_mode,
    )
    sys.exit(exit_code)


@main.command("review-files")
@click.argument("files", nargs=-1, required=True)
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Path to multi_agent.toml config file.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show proposed changes without applying.")
@click.option("--max-rounds", type=int, default=None,
              help="Override max iteration rounds.")
@click.pass_context
def review_files(
    ctx: click.Context,
    files: tuple[str, ...],
    config_path: str | None,
    dry_run: bool,
    max_rounds: int | None,
) -> None:
    """Review existing files and propose changes via consensus.

    Paths are resolved relative to the repo root (or --repo). You can pass
    files or directories (directories are searched for matching file patterns).

    \b
    Examples:
        python -m multi_agent review-files canon/chapter-01.md
        python -m multi_agent review-files canon/
        python -m multi_agent --repo ~/git/my-novel review-files docs/
    """
    try:
        repo_root = _resolve_repo(ctx)
    except Exception:
        print_error("Not in a git repository.")
        sys.exit(1)

    config = load_config(
        Path(config_path) if config_path else None,
        search_from=repo_root,
    )
    if max_rounds is not None:
        config.general.max_rounds = max_rounds

    from multi_agent.consensus import resolve_file_args
    from multi_agent.context import count_uncommitted_canon, load_canon

    try:
        resolved = resolve_file_args(list(files), repo_root, config)
    except FileNotFoundError as exc:
        print_error(str(exc))
        sys.exit(1)

    canon = load_canon(
        repo_root,
        config.general.canon_directories,
        config.general.file_patterns,
        config.general.max_canon_size_kb,
    )
    canon_size_kb = sum(len(v.encode()) for v in canon.values()) / 1024
    uncommitted = count_uncommitted_canon(
        repo_root,
        config.general.canon_directories,
        config.general.file_patterns,
        set(canon.keys()),
    )

    exit_code = _run_iteration_and_present(
        config=config,
        repo_root=repo_root,
        target_files=resolved,
        files_display=resolved,
        canon_count=len(canon),
        canon_size_kb=canon_size_kb,
        uncommitted_canon=uncommitted,
        dry_run=dry_run,
        hook_mode=False,
    )
    sys.exit(exit_code)


@main.command("install-hook")
@click.pass_context
def install_hook(ctx: click.Context) -> None:
    """Install the git pre-commit hook."""
    from multi_agent.hook import install_hook as do_install

    try:
        repo_root = _resolve_repo(ctx)
    except Exception:
        print_error("Not in a git repository.")
        sys.exit(1)

    do_install(repo_root)
    console.print("[green]Pre-commit hook installed.[/green]")


@main.command("uninstall-hook")
@click.pass_context
def uninstall_hook(ctx: click.Context) -> None:
    """Remove the git pre-commit hook."""
    from multi_agent.hook import uninstall_hook as do_uninstall

    try:
        repo_root = _resolve_repo(ctx)
    except Exception:
        print_error("Not in a git repository.")
        sys.exit(1)

    do_uninstall(repo_root)
    console.print("[green]Pre-commit hook removed.[/green]")


@main.command("check-config")
@click.option("--config", "config_path", type=click.Path(exists=True), default=None)
@click.pass_context
def check_config(ctx: click.Context, config_path: str | None) -> None:
    """Validate and display the current configuration."""
    try:
        repo_root = _resolve_repo(ctx)
    except Exception:
        repo_root = None
    try:
        config = load_config(
            Path(config_path) if config_path else None,
            search_from=repo_root,
        )
    except Exception as exc:
        print_error(f"Invalid configuration: {exc}")
        sys.exit(1)

    console.print("[bold]Configuration:[/bold]")
    console.print(f"  File patterns: {config.general.file_patterns}")
    console.print(f"  Consensus threshold: {config.general.consensus_threshold}")
    console.print(f"  Timeout: {config.general.timeout_seconds}s")
    console.print(f"  Max rounds: {config.general.max_rounds}")
    console.print(f"  Min severity: {config.general.min_severity}")
    console.print(f"  Propose max turns: {config.general.propose_max_turns}")
    console.print(f"  Review max turns: {config.general.review_max_turns}")
    console.print(f"  Canon directories: {config.general.canon_directories}")
    console.print()
    console.print("[bold]Agents:[/bold]")
    for name, agent in config.agents.items():
        status = "[green]enabled[/green]" if agent.enabled else "[red]disabled[/red]"
        model = agent.model or "default"
        review_model = agent.review_model or model
        tools = ", ".join(agent.allowed_tools) if agent.allowed_tools else "none"
        console.print(f"  {name}: {status} (propose: {model}, review: {review_model}, tools: {tools})")
