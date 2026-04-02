"""CLI entry point using Click."""

from __future__ import annotations

import asyncio
import dataclasses
import sys
from pathlib import Path

import click

from multi_agent.config import CommandConfig, load_config
from multi_agent.context import find_git_root
from multi_agent.output import (
    console,
    init_agent_styles,
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
    print_review_round,
    print_token_usage,
)


class ConfigGroup(click.Group):
    """Click group that auto-generates commands from TOML [commands] config."""

    def _get_toml_commands(self, ctx: click.Context) -> dict[str, CommandConfig]:
        cache = ctx.meta.get("_toml_commands")
        if cache is not None:
            return cache
        try:
            # ctx.params may not have repo_path yet because --help is
            # eager and fires before non-eager options are resolved.
            repo = ctx.params.get("repo_path") if ctx.params else None
            if repo is None:
                repo = self._repo_from_argv()
            search = Path(repo) if repo else None
            config = load_config(search_from=search)
            ctx.meta["_toml_commands"] = config.commands
        except Exception:
            ctx.meta["_toml_commands"] = {}
        return ctx.meta["_toml_commands"]

    @staticmethod
    def _repo_from_argv() -> str | None:
        """Extract --repo value from sys.argv as a fallback."""
        import sys
        args = sys.argv[1:]
        for i, arg in enumerate(args):
            if arg == "--repo" and i + 1 < len(args):
                return args[i + 1]
        return None

    def list_commands(self, ctx: click.Context) -> list[str]:
        builtin = set(super().list_commands(ctx))
        dynamic = [k for k in self._get_toml_commands(ctx) if k not in builtin]
        return sorted(builtin | set(dynamic))

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd
        toml_cmds = self._get_toml_commands(ctx)
        if cmd_name in toml_cmds:
            return _make_toml_command(cmd_name, toml_cmds[cmd_name])
        return None


def _make_toml_command(cmd_name: str, cmd_config: CommandConfig) -> click.Command:
    """Create a Click command from a TOML [commands] entry."""

    @click.command(
        cmd_name,
        help=cmd_config.description
        or f"Run the '{cmd_name}' command via consensus.",
    )
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
    def cmd(
        ctx: click.Context,
        files: tuple[str, ...],
        config_path: str | None,
        dry_run: bool,
        max_rounds: int | None,
        prompt: str | None,
    ) -> None:
        _review_common(
            ctx, files, config_path, False, dry_run, max_rounds, cmd_name, prompt,
        )

    return cmd


@click.group(cls=ConfigGroup)
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


def _resolve_task(
    task_name: str | None, config,
) -> tuple[str | None, CommandConfig | None]:
    """Resolve a --task name to (command_name, CommandConfig).

    Returns (None, None) when no --task is given (uses default review command).
    Raises click.BadParameter if the task name is not in config.commands.
    """
    if task_name is None:
        return None, None
    if task_name in config.commands:
        return task_name, config.commands[task_name]
    available = sorted(config.commands.keys())
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
    command_name: str | None = None,
    command_prompt: str | None = None,
    severity_filter: bool = True,
    command_propose_model: str | None = None,
    command_review_model: str | None = None,
    task_label: str | None = None,
) -> int:
    """Run the iteration loop and present results. Returns exit code."""
    from multi_agent.consensus import run_iteration_loop
    from multi_agent.context import apply_merged_texts, build_diff_preview_from_merged
    from multi_agent.models import (
        ArbitrationDone,
        ArbitrationStart,
        DissentsDone,
        PhaseEvent,
        ProposeDone,
        ReviewDone,
        count_approvals,
    )

    def on_phase(event: PhaseEvent) -> None:
        match event:
            case ProposeDone(proposals=proposals):
                print_proposals_summary(proposals)
            case ReviewDone(round_number=rn, reviews=reviews, consensus_threshold=ct):
                print_review_round(rn, reviews, ct)
            case ArbitrationStart(contested=contested):
                print_arbitration_start(contested)
            case ArbitrationDone(results=results):
                print_arbitration_done(results)
            case DissentsDone(dissents=dissents):
                print_dissents(dissents)

    display_task = task_label or command_name
    print_header(files_display, canon_count, canon_size_kb, uncommitted_canon, task=display_task)

    result = asyncio.run(run_iteration_loop(
        config, str(repo_root), target_files=target_files,
        command_name=command_name, command_prompt=command_prompt,
        severity_filter=severity_filter,
        command_propose_model=command_propose_model,
        command_review_model=command_review_model,
        on_progress=print_progress,
        on_phase=on_phase,
    ))

    # No edits?
    if not result.merged_texts:
        print_no_edits()
        print_token_usage(result.total_usage, result.total_duration_seconds)
        return 0

    # Build file contents for diff preview
    file_contents: dict[str, str] = {}
    for f in result.files_reviewed:
        path = repo_root / f
        if path.is_file():
            file_contents[f] = path.read_text()

    diff_text = build_diff_preview_from_merged(result.merged_texts, file_contents)

    # Consensus status
    if result.consensus_reached:
        last_round = result.rounds[-1] if result.rounds else None
        if last_round:
            approvals = count_approvals(last_round.reviews)
            print_iteration_success(approvals, len(last_round.reviews))
        else:
            print_iteration_success(0, 0)
    else:
        total = len(result.rounds[-1].reviews) if result.rounds else 0
        print_iteration_exhausted(
            rounds_run=len(result.rounds),
            max_rounds=config.general.max_rounds,
            approvals=result.best_approvals,
            total=total,
            best_round=result.best_round,
            stalled=result.stalled,
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
        modified = apply_merged_texts(repo_root, result.merged_texts)
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
    """Shared implementation for review and TOML-defined commands."""
    try:
        repo_root = _resolve_repo(ctx)
    except Exception:
        print_error("Not in a git repository.")
        sys.exit(1)

    config = load_config(
        Path(config_path) if config_path else None,
        search_from=repo_root,
    )
    init_agent_styles(config.agents)
    if max_rounds is not None:
        config = dataclasses.replace(
            config,
            general=dataclasses.replace(config.general, max_rounds=max_rounds),
        )

    cmd_name, cmd_config = _resolve_task(task_name, config)

    # Determine command and severity filtering.
    if cmd_name is None:
        # No --task: default to the "review" command with severity filtering.
        cmd_name = "review"
        cmd_config = config.commands["review"]
        severity_filter = True
    else:
        severity_filter = False

    task_label = cmd_name

    # --prompt: append or override user instructions
    if prompt:
        if task_name is None:
            # No --task given: treat bare --prompt as a standalone command.
            cmd_name = "prompt"
            cmd_config = CommandConfig(prompt=prompt)
            severity_filter = False
            task_label = "prompt"
        else:
            # Append user instructions to the command's prompt.
            cmd_config = dataclasses.replace(
                cmd_config,
                prompt=f"{cmd_config.prompt}\n\nAdditional instructions: {prompt}",
            )

    command_prompt = cmd_config.prompt

    # Apply per-command agent filtering and consensus_threshold.
    if cmd_config.agents or cmd_config.consensus_threshold is not None:
        agents = (
            {k: v for k, v in config.agents.items() if k in cmd_config.agents}
            if cmd_config.agents else config.agents
        )
        threshold = cmd_config.consensus_threshold or config.general.consensus_threshold
        threshold = min(threshold, len(agents))
        config = dataclasses.replace(
            config,
            agents=agents,
            general=dataclasses.replace(config.general, consensus_threshold=threshold),
        )
        init_agent_styles(config.agents)

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
        command_name=cmd_name,
        command_prompt=command_prompt,
        severity_filter=severity_filter,
        command_propose_model=cmd_config.propose_model,
        command_review_model=cmd_config.review_model,
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
              help="Run a command from [commands] config (e.g. expand, contract).")
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

    init_agent_styles(config.agents)

    from multi_agent.config import get_display_name

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
        display = get_display_name(name, agent)
        status = "[green]enabled[/green]" if agent.enabled else "[red]disabled[/red]"
        model = agent.propose_model or "default"
        review_model = agent.review_model or model
        tools = ", ".join(agent.allowed_tools) if agent.allowed_tools else "none"
        has_prompt = "yes" if agent.system_prompt else "[red]missing[/red]"
        console.print(
            f"  {display} ({name}): {status} "
            f"(propose: {model}, review: {review_model}, "
            f"tools: {tools}, prompt: {has_prompt})"
        )

    if config.commands:
        console.print()
        console.print("[bold]Commands:[/bold]")
        for name, cmd_cfg in config.commands.items():
            desc = cmd_cfg.description
            if not desc:
                desc = cmd_cfg.prompt[:60] + "..." if len(cmd_cfg.prompt) > 60 else cmd_cfg.prompt
            console.print(f"  {name}: {desc}")
