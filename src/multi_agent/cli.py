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
    print_arbitration_done,
    print_arbitration_start,
    print_changes_applied,
    print_confirmation_prompt,
    print_dissents,
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


BUILTIN_TASKS = {"expand", "contract"}


def _resolve_task(task_name: str | None, config) -> tuple[str | None, str | None]:
    """Resolve a task name to (mode, custom_prompt).

    Returns (None, None) for default review mode.
    Returns ("expand", None) or ("contract", None) for built-ins.
    Returns ("custom", prompt) for config-defined tasks.
    Raises click.BadParameter if task is unknown.
    """
    if task_name is None:
        return None, None
    if task_name in BUILTIN_TASKS:
        return task_name, None
    if task_name in config.tasks:
        return "custom", config.tasks[task_name].prompt
    available = sorted(BUILTIN_TASKS | set(config.tasks.keys()))
    raise click.BadParameter(
        f"Unknown task '{task_name}'. Available: {', '.join(available)}",
        param_hint="'--task'",
    )


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
    task: str | None = None,
    custom_task_prompt: str | None = None,
    task_label: str | None = None,
) -> int:
    """Run the iteration loop and present results. Returns exit code."""
    from multi_agent.consensus import run_iteration_loop
    from multi_agent.context import apply_edits, build_diff_preview

    def on_phase(event, *args):
        if event == "propose_done":
            print_proposals_summary(args[0])
        elif event == "review_done":
            print_review_round(args[0], args[1], args[2])
        elif event == "arbitration_start":
            print_arbitration_start(args[0])
        elif event == "arbitration_done":
            print_arbitration_done(args[0])
        elif event == "dissents_done":
            print_dissents(args[0])

    display_task = task_label or task
    print_header(files_display, canon_count, canon_size_kb, uncommitted_canon, task=display_task)

    result = asyncio.run(run_iteration_loop(
        config, str(repo_root), target_files=target_files,
        task=task, custom_task_prompt=custom_task_prompt,
        on_progress=print_progress,
        on_phase=on_phase,
    ))

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
        total = len(result.rounds[-1].reviews) if result.rounds else 0
        print_iteration_exhausted(
            config.general.max_rounds,
            result.best_approvals,
            total,
            result.best_round,
        )

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
            "Run [bold]multi-agent review[/bold] to review and apply interactively."
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


def _review_common(
    ctx: click.Context,
    files: tuple[str, ...],
    config_path: str | None,
    hook_mode: bool,
    dry_run: bool,
    max_rounds: int | None,
    task_name: str | None,
    prompt: str | None = None,
) -> None:
    """Shared implementation for review, expand, and contract commands."""
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

    task, custom_task_prompt = _resolve_task(task_name, config)

    # --prompt: append user instructions to the task prompt
    task_label = task  # preserve original task name for display
    if prompt:
        if task is None:
            # No task specified — treat prompt as a custom task
            task = "custom"
            task_label = "prompt"
            custom_task_prompt = prompt
        elif task == "custom":
            # Config-defined task — append user instructions
            custom_task_prompt = f"{custom_task_prompt}\n\n{prompt}"
        else:
            # Built-in task (expand/contract) — switch to custom with
            # the built-in's suffix + user instructions
            from multi_agent.agents import _MODE_SUFFIXES
            builtin_instructions = _MODE_SUFFIXES.get(task, "")
            task_label = task  # keep showing "expand" or "contract"
            task = "custom"
            custom_task_prompt = f"{builtin_instructions.strip()}\n\nAdditional instructions: {prompt}"

    from multi_agent.consensus import resolve_file_args
    from multi_agent.context import (
        count_uncommitted_canon,
        get_staged_files,
        load_canon,
    )

    # If files provided, review those files. Otherwise review staged files.
    if files:
        try:
            resolved = resolve_file_args(list(files), repo_root, config)
        except FileNotFoundError as exc:
            print_error(str(exc))
            sys.exit(1)
        target_files: list[str] | None = resolved
        files_display = resolved
    else:
        staged = get_staged_files(repo_root, config.general.file_patterns)
        if not staged:
            print_no_files()
            sys.exit(0)
        target_files = None  # signals: use staged files
        files_display = [str(f) for f in staged]

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
        target_files=target_files,
        files_display=files_display,
        canon_count=len(canon),
        canon_size_kb=canon_size_kb,
        uncommitted_canon=uncommitted,
        dry_run=dry_run,
        hook_mode=hook_mode,
        task=task,
        custom_task_prompt=custom_task_prompt,
        task_label=task_label,
    )
    sys.exit(exit_code)


@main.command()
@click.argument("files", nargs=-1)
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Path to multi_agent.toml config file.")
@click.option("--hook-mode", is_flag=True, hidden=True,
              help="Running as a git pre-commit hook.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show proposed changes without applying.")
@click.option("--max-rounds", type=int, default=None,
              help="Override max iteration rounds.")
@click.option("--task", "task_name", default=None,
              help="Task mode: expand, contract, or a custom task from config.")
@click.option("--prompt", "prompt", default=None,
              help="Additional instructions for the agents.")
@click.pass_context
def review(
    ctx: click.Context,
    files: tuple[str, ...],
    config_path: str | None,
    hook_mode: bool,
    dry_run: bool,
    max_rounds: int | None,
    task_name: str | None,
    prompt: str | None,
) -> None:
    """Review files and propose changes via consensus.

    When FILES are given, reviews those files on disk. When no FILES are given,
    reviews staged files (used by the pre-commit hook).

    \b
    Examples:
        multi-agent review canon/chapter-01.md
        multi-agent review --task expand canon/chapter-03.md
        multi-agent review --prompt "Add epigraphs to each section" canon/
        multi-agent review                  # reviews staged files
    """
    _review_common(ctx, files, config_path, hook_mode, dry_run, max_rounds, task_name, prompt)


@main.command()
@click.argument("files", nargs=-1, required=True)
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Path to multi_agent.toml config file.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show proposed changes without applying.")
@click.option("--max-rounds", type=int, default=None,
              help="Override max iteration rounds.")
@click.option("--prompt", "prompt", default=None,
              help="Additional instructions for the agents.")
@click.pass_context
def expand(
    ctx: click.Context,
    files: tuple[str, ...],
    config_path: str | None,
    dry_run: bool,
    max_rounds: int | None,
    prompt: str | None,
) -> None:
    """Expand files with richer detail via consensus.

    Shortcut for: review --task expand FILES

    \b
    Examples:
        multi-agent expand canon/chapter-03.md
        multi-agent expand --prompt "Focus on the descent sequence" canon/chapter-05.md
    """
    _review_common(ctx, files, config_path, False, dry_run, max_rounds, "expand", prompt)


@main.command()
@click.argument("files", nargs=-1, required=True)
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Path to multi_agent.toml config file.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show proposed changes without applying.")
@click.option("--max-rounds", type=int, default=None,
              help="Override max iteration rounds.")
@click.option("--prompt", "prompt", default=None,
              help="Additional instructions for the agents.")
@click.pass_context
def contract(
    ctx: click.Context,
    files: tuple[str, ...],
    config_path: str | None,
    dry_run: bool,
    max_rounds: int | None,
    prompt: str | None,
) -> None:
    """Tighten prose and cut filler via consensus.

    Shortcut for: review --task contract FILES

    \b
    Examples:
        multi-agent contract canon/chapter-03.md
        multi-agent contract --prompt "Preserve all dialogue" canon/chapter-05.md
    """
    _review_common(ctx, files, config_path, False, dry_run, max_rounds, "contract", prompt)


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
        model = agent.propose_model or "default"
        review_model = agent.review_model or model
        tools = ", ".join(agent.allowed_tools) if agent.allowed_tools else "none"
        console.print(f"  {name}: {status} (propose: {model}, review: {review_model}, tools: {tools})")

    if config.tasks:
        console.print()
        console.print("[bold]Custom Tasks:[/bold]")
        for name, task_cfg in config.tasks.items():
            prompt_preview = task_cfg.prompt[:60] + "..." if len(task_cfg.prompt) > 60 else task_cfg.prompt
            console.print(f"  {name}: {prompt_preview}")
    console.print()
    console.print("[dim]Built-in tasks: expand, contract[/dim]")
