"""CLI entry point using Click."""

from __future__ import annotations

import asyncio
import dataclasses
import sys
from pathlib import Path

import click

from multi_agent.config import CommandConfig, load_config, resolve_run_config
from multi_agent.context import find_git_root
from multi_agent.output import (
    console,
    init_agent_styles,
    is_detail,
    is_verbose,
    print_agent_verbose_stats,
    print_answer,
    print_arbitration_done,
    print_arbitration_start,
    print_changes_applied,
    print_dissents,
    print_error,
    print_final_diff,
    print_header,
    print_iteration_exhausted,
    print_iteration_success,
    print_edit_list,
    print_no_edits,
    print_no_files,
    print_progress,
    print_proposal_details,
    prompt_edit_selection,
    print_propose_start,
    print_proposals_summary,
    print_resolved_config,
    print_review_details,
    print_review_round,
    print_review_start,
    print_token_usage,
    set_detail,
    set_verbose,
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
    @click.option("--detail", "-d", is_flag=True, default=False,
                  help="Show full edit content for each round.")
    @click.option("--verbose", "-v", is_flag=True, default=False,
                  help="Show detailed agent activity, config, and token usage.")
    @click.pass_context
    def cmd(
        ctx: click.Context,
        files: tuple[str, ...],
        config_path: str | None,
        dry_run: bool,
        max_rounds: int | None,
        prompt: str | None,
        detail: bool,
        verbose: bool,
    ) -> None:
        set_detail(detail)
        set_verbose(verbose)
        _review_common(
            ctx, files, config_path, False, dry_run, max_rounds, cmd_name, prompt,
        )

    return cmd


@click.group(cls=ConfigGroup)
@click.option("--repo", "repo_path", type=click.Path(exists=True), default=None,
              help="Path to a git repository to review. Defaults to current directory.")
@click.pass_context
def main(ctx: click.Context, repo_path: str | None) -> None:
    """Multi-agent consensus review system."""
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


def _create_backend(resolved):
    """Create the agent backend and validate tool configuration."""
    if resolved.backend == "claude-cli":
        from multi_agent.claude_runner import ClaudeCliBackend
        backend = ClaudeCliBackend()
        for name, settings in resolved.agent_settings.items():
            if settings.allowed_tools:
                backend.validate_tools(name, settings.allowed_tools)
        return backend
    raise ValueError(f"Unknown backend: {resolved.backend}")


def _make_phase_handler(resolved=None):
    """Create a callback that prints phase results as they complete."""
    from multi_agent.models import (
        ArbitrationDone,
        ArbitrationStart,
        DissentsDone,
        PhaseEvent,
        ProposeDone,
        ProposeStart,
        ReviewDone,
        ReviewStart,
    )

    def on_phase(event: PhaseEvent) -> None:
        match event:
            case ProposeStart():
                print_propose_start()
            case ReviewStart(round_number=rn):
                print_review_start(rn)
            case ProposeDone(proposals=proposals):
                print_proposals_summary(proposals)
                if is_detail():
                    print_proposal_details(proposals)
                if is_verbose() and resolved:
                    for p in proposals:
                        print_agent_verbose_stats(
                            p.agent_name, p.turns_taken,
                            p.tool_usage, p.usage,
                        )
            case ReviewDone(round_number=rn, reviews=reviews,
                            consensus_threshold=ct,
                            blocking_approvals=ba,
                            proposals=props):
                print_review_round(rn, reviews, ct, blocking_approvals=ba)
                if is_detail():
                    print_review_details(reviews, props)
                if is_verbose() and resolved:
                    for r in reviews:
                        print_agent_verbose_stats(
                            r.agent_name, r.turns_taken,
                            r.tool_usage, r.usage,
                        )
            case ArbitrationStart(contested=contested):
                print_arbitration_start(contested)
            case ArbitrationDone(results=results):
                print_arbitration_done(results)
            case DissentsDone(dissents=dissents):
                print_dissents(dissents)

    return on_phase


def _load_reference_context(repo_root: Path, settings):
    """Load reference files and return (reference, ref_size_kb, uncommitted_count)."""
    from multi_agent.context import count_uncommitted_reference, load_reference

    ref = load_reference(
        repo_root,
        settings.reference_directories,
        settings.file_patterns,
        settings.max_reference_size_kb,
    )
    ref_size_kb = sum(len(v.encode()) for v in ref.values()) / 1024
    uncommitted = count_uncommitted_reference(
        repo_root,
        settings.reference_directories,
        settings.file_patterns,
        set(ref.keys()),
    )
    return ref, ref_size_kb, uncommitted


def _print_consensus_status(result, resolved) -> None:
    """Print consensus-reached or exhausted status from an IterationResult."""
    if result.consensus_reached:
        last_round = result.rounds[-1] if result.rounds else None
        if last_round:
            print_iteration_success(last_round.approvals, len(last_round.reviews))
        else:
            print_iteration_success(0, 0)
    else:
        total = len(result.rounds[-1].reviews) if result.rounds else 0
        print_iteration_exhausted(
            rounds_run=len(result.rounds),
            max_rounds=resolved.max_rounds,
            approvals=result.best_approvals,
            total=total,
            best_round=result.best_round,
            stalled=result.stalled,
        )


def _run_iteration_and_present(
    resolved,
    repo_root: Path,
    target_files: list[str] | None,
    files_display: list[str],
    ref_count: int,
    ref_size_kb: float,
    uncommitted_ref: int,
    dry_run: bool,
    hook_mode: bool,
    task_label: str | None = None,
) -> int:
    """Run the iteration loop and present results. Returns exit code."""
    from multi_agent.consensus import run_iteration_loop
    from multi_agent.context import apply_merged_texts, build_diff_preview_from_merged

    backend = _create_backend(resolved)
    on_phase = _make_phase_handler(resolved)

    display_task = task_label or resolved.command_name
    print_header(files_display, ref_count, ref_size_kb, uncommitted_ref, task=display_task)
    if is_verbose():
        print_resolved_config(resolved)

    result = asyncio.run(run_iteration_loop(
        resolved, str(repo_root), backend, target_files=target_files,
        on_progress=print_progress,
        on_phase=on_phase,
    ))

    # Write structured log before any user interaction
    from multi_agent.logging import write_run_log
    write_run_log(repo_root, result, resolved)

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

    _print_consensus_status(result, resolved)
    print_token_usage(result.total_usage, result.total_duration_seconds)

    # Build numbered edit list
    from multi_agent.config import get_display_name
    numbered_edits: list[tuple[int, str, object]] = []
    edit_owners: list[tuple[str, int]] = []  # (agent_name, edit_index_in_proposal)
    num = 1
    for proposal in result.proposals:
        display = get_display_name(proposal.agent_name,
                                    resolved.agents[proposal.agent_name])
        for i, edit in enumerate(proposal.edits):
            numbered_edits.append((num, display, edit))
            edit_owners.append((proposal.agent_name, i))
            num += 1

    # Show numbered edits with full content
    print_edit_list(numbered_edits)

    if dry_run:
        console.print("[dim]Dry run — no changes applied.[/dim]")
        return 0

    # Non-interactive hook: show unified diff, don't apply
    if hook_mode and not sys.stdin.isatty():
        diff_text = build_diff_preview_from_merged(result.merged_texts, file_contents)
        print_final_diff(diff_text)
        console.print(
            "[yellow]Changes proposed but cannot confirm in non-interactive mode.[/yellow]\n"
            "Run [bold]multi-agent review[/bold] to review and apply interactively."
        )
        return 1

    selected = prompt_edit_selection(len(numbered_edits))

    if not selected:
        console.print("[dim]Changes not applied.[/dim]")
        return 1 if hook_mode else 0

    # Determine what to apply
    all_selected = selected == set(range(1, len(numbered_edits) + 1))
    if all_selected:
        texts_to_apply = result.merged_texts
    else:
        from multi_agent.merge import merge_agent_edits
        selected_keys = {edit_owners[n - 1] for n in selected}
        filtered = [
            dataclasses.replace(proposal, edits=[
                edit for i, edit in enumerate(proposal.edits)
                if (proposal.agent_name, i) in selected_keys
            ])
            for proposal in result.proposals
        ]
        re_merged = merge_agent_edits(file_contents, filtered)
        texts_to_apply = re_merged.merged_texts

        if not texts_to_apply:
            console.print("[dim]No changes to apply after filtering.[/dim]")
            return 0

    # Show unified diff of what will be applied
    diff_text = build_diff_preview_from_merged(texts_to_apply, file_contents)
    if not all_selected:
        console.print(
            f"\n  [bold cyan]Applying {len(selected)}/{len(numbered_edits)}"
            f" edit(s)[/bold cyan]"
        )
    print_final_diff(diff_text)

    modified = apply_merged_texts(repo_root, texts_to_apply)
    print_changes_applied(modified)
    if hook_mode:
        console.print(
            "[yellow]Changes applied to working tree. "
            "Review the modifications, stage them, and commit again.[/yellow]"
        )
        return 1  # exit 1 so the current commit is aborted
    return 0


def _prepare_command(
    ctx: click.Context,
    config_path: str | None,
    max_rounds: int | None,
    command_name: str,
    *,
    task_name: str | None = None,
    prompt: str | None = None,
):
    """Load config, resolve command, and build ResolvedRunConfig.

    Returns (resolved, repo_root).  Every CLI command flows through here
    so new commands inherit shared behaviour automatically.
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
        config = dataclasses.replace(
            config,
            general=dataclasses.replace(config.general, max_rounds=max_rounds),
        )

    # Resolve which command to run.
    cmd_name, cmd_config = _resolve_task(task_name, config)
    if cmd_name is None:
        cmd_name = command_name
        cmd_config = config.commands[command_name]

    # --prompt: append or override user instructions
    if prompt:
        if task_name is None:
            cmd_name = "prompt"
            cmd_config = CommandConfig(prompt=prompt)
        else:
            cmd_config = dataclasses.replace(
                cmd_config,
                prompt=f"{cmd_config.prompt}\n\nAdditional instructions: {prompt}",
            )

    resolved = resolve_run_config(config, cmd_name, cmd_config)
    init_agent_styles(resolved.agents)
    return resolved, repo_root


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
    resolved, repo_root = _prepare_command(
        ctx, config_path, max_rounds, "review",
        task_name=task_name, prompt=prompt,
    )

    from multi_agent.context import get_staged_files, resolve_file_args

    # Union of all agents' file_patterns for target file resolution
    all_patterns = sorted({
        pat for s in resolved.agent_settings.values() for pat in s.file_patterns
    })

    # If files provided, review those files. Otherwise review staged files.
    if files:
        try:
            file_list = resolve_file_args(
                list(files), repo_root, all_patterns,
            )
        except FileNotFoundError as exc:
            print_error(str(exc))
            sys.exit(1)
        target_files: list[str] | None = file_list
        files_display = file_list
    else:
        staged = get_staged_files(repo_root, all_patterns)
        if not staged:
            print_no_files()
            sys.exit(0)
        target_files = None  # signals: use staged files
        files_display = [str(f) for f in staged]

    # Use first agent's settings for the header's reference summary display
    first_settings = next(iter(resolved.agent_settings.values()))
    ref, ref_size_kb, uncommitted = _load_reference_context(repo_root, first_settings)

    exit_code = _run_iteration_and_present(
        resolved=resolved,
        repo_root=repo_root,
        target_files=target_files,
        files_display=files_display,
        ref_count=len(ref),
        ref_size_kb=ref_size_kb,
        uncommitted_ref=uncommitted,
        dry_run=dry_run,
        hook_mode=hook_mode,
        task_label=resolved.command_name,
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
@click.option("--detail", "-d", is_flag=True, default=False,
              help="Show full edit content for each round.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show detailed agent activity, config, and token usage.")
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
    detail: bool,
    verbose: bool,
) -> None:
    """Review files and propose changes via consensus.

    When FILES are given, reviews those files on disk. When no FILES are given,
    reviews staged files (used by the pre-commit hook).

    \b
    Examples:
        multi-agent review chapter-01.md
        multi-agent review --task expand chapter-03.md
        multi-agent review --prompt "Add epigraphs to each section" docs/
        multi-agent review                  # reviews staged files
    """
    set_detail(detail)
    set_verbose(verbose)
    _review_common(ctx, files, config_path, hook_mode, dry_run, max_rounds, task_name, prompt)


def _run_ask(
    resolved,
    repo_root: Path,
    question: str,
) -> int:
    """Write question to a temp file, run consensus, clean up afterward."""
    from multi_agent.consensus import run_iteration_loop

    first_settings = next(iter(resolved.agent_settings.values()))

    ref, ref_size_kb, uncommitted = _load_reference_context(repo_root, first_settings)

    backend = _create_backend(resolved)
    on_phase = _make_phase_handler(resolved)

    answer_name = ".multi_agent_ask_answer.md"
    answer_path = repo_root / answer_name

    # Write the question so agents have a file to edit.
    answer_path.write_text(question)

    print_header(
        [answer_name], len(ref), ref_size_kb, uncommitted, task=resolved.command_name,
    )
    if is_verbose():
        print_resolved_config(resolved)

    try:
        result = asyncio.run(run_iteration_loop(
            resolved, str(repo_root), backend,
            target_files=[answer_name],
            on_progress=print_progress,
            on_phase=on_phase,
        ))

        from multi_agent.logging import write_run_log
        write_run_log(repo_root, result, resolved)

        if not result.merged_texts:
            print_no_edits()
            print_token_usage(result.total_usage, result.total_duration_seconds)
            return 0

        answer = result.merged_texts.get(answer_name, "")

        _print_consensus_status(result, resolved)

        print_answer(answer)
        print_token_usage(result.total_usage, result.total_duration_seconds)
        return 0
    finally:
        # Clean up temp file — the answer is in the run log and console output.
        answer_path.unlink(missing_ok=True)


@main.command()
@click.argument("question", nargs=-1, required=True)
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Path to multi_agent.toml config file.")
@click.option("--max-rounds", type=int, default=None,
              help="Override max iteration rounds.")
@click.option("--detail", "-d", is_flag=True, default=False,
              help="Show full edit content for each round.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show detailed agent activity, config, and token usage.")
@click.pass_context
def ask(
    ctx: click.Context,
    question: tuple[str, ...],
    config_path: str | None,
    max_rounds: int | None,
    detail: bool,
    verbose: bool,
) -> None:
    """Ask a question and get a consensus answer from all agents.

    \b
    Examples:
        multi-agent ask "What are the main themes of this project?"
        multi-agent ask What is the best approach for character development
        multi-agent ask --max-rounds 5 "How should we handle the timeline?"
    """
    set_detail(detail)
    set_verbose(verbose)
    question_text = " ".join(question)
    if not question_text.strip():
        raise click.BadParameter("Question cannot be empty.", param_hint="'QUESTION'")

    resolved, repo_root = _prepare_command(ctx, config_path, max_rounds, "ask")

    exit_code = _run_ask(resolved, repo_root, question_text)
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

    init_agent_styles(config.agents)

    from multi_agent.config import get_display_name

    # Show global defaults
    g = config.general
    console.print("[bold]Global Defaults:[/bold]")
    console.print(f"  propose_model: {g.propose_model or '[dim]backend default[/dim]'}")
    console.print(f"  review_model: {g.review_model or '[dim]falls back to propose_model[/dim]'}")
    console.print(f"  timeout_seconds: {g.timeout_seconds}")
    console.print(f"  file_patterns: {g.file_patterns}")
    console.print(f"  reference_directories: {g.reference_directories}")
    console.print(f"  max_reference_size_kb: {g.max_reference_size_kb}")
    console.print(f"  allowed_tools: {g.allowed_tools or '[dim]none[/dim]'}")
    console.print(f"  propose_max_turns: {g.propose_max_turns}")
    console.print(f"  review_max_turns: {g.review_max_turns}")
    console.print(f"  consensus_threshold: {g.consensus_threshold}")
    console.print(f"  max_rounds: {g.max_rounds}")
    console.print(f"  min_severity: {g.min_severity}")
    console.print(f"  min_blocking_severity: {g.min_blocking_severity}")

    # Show agents
    console.print()
    console.print("[bold]Agents:[/bold]")
    for name, agent in config.agents.items():
        display = get_display_name(name, agent)
        status = "[green]enabled[/green]" if agent.enabled else "[red]disabled[/red]"
        has_prompt = "yes" if agent.system_prompt else "[red]missing[/red]"
        console.print(f"  [bold]{display}[/bold] ({name}): {status}, prompt: {has_prompt}")
        # Show overrides only (fields that differ from None/inherit)
        overrides = []
        if agent.propose_model is not None:
            overrides.append(f"propose_model={agent.propose_model}")
        if agent.review_model is not None:
            overrides.append(f"review_model={agent.review_model}")
        if agent.timeout_seconds is not None:
            overrides.append(f"timeout_seconds={agent.timeout_seconds}")
        if agent.allowed_tools is not None:
            overrides.append(f"allowed_tools={agent.allowed_tools}")
        if agent.file_patterns is not None:
            overrides.append(f"file_patterns={agent.file_patterns}")
        if agent.reference_directories is not None:
            overrides.append(f"reference_directories={agent.reference_directories}")
        if agent.max_reference_size_kb is not None:
            overrides.append(f"max_reference_size_kb={agent.max_reference_size_kb}")
        if agent.propose_max_turns is not None:
            overrides.append(f"propose_max_turns={agent.propose_max_turns}")
        if agent.review_max_turns is not None:
            overrides.append(f"review_max_turns={agent.review_max_turns}")
        if overrides:
            console.print(f"    overrides: {', '.join(overrides)}")

    # Show commands with resolved settings per agent
    if config.commands:
        console.print()
        console.print("[bold]Commands:[/bold]")
        for cmd_name, cmd_cfg in config.commands.items():
            desc = cmd_cfg.description
            if not desc:
                desc = cmd_cfg.prompt[:60] + "..." if len(cmd_cfg.prompt) > 60 else cmd_cfg.prompt
            console.print(f"  [bold]{cmd_name}[/bold]: {desc}")

            try:
                resolved = resolve_run_config(config, cmd_name, cmd_cfg)
            except ValueError as exc:
                console.print(f"    [red]Error resolving: {exc}[/red]")
                continue

            console.print(
                f"    max_rounds={resolved.max_rounds}, "
                f"consensus_threshold={resolved.consensus_threshold}, "
                f"min_severity={resolved.min_severity}, "
                f"min_blocking_severity={resolved.min_blocking_severity}"
            )
            for agent_name, settings in resolved.agent_settings.items():
                display = get_display_name(agent_name, config.agents[agent_name])
                propose = settings.propose_model or "default"
                review = settings.review_model or "default"
                tools = ", ".join(settings.allowed_tools) if settings.allowed_tools else "none"
                console.print(
                    f"    {display}: "
                    f"propose={propose}, review={review}, "
                    f"timeout={settings.timeout_seconds}s, tools=[{tools}]"
                )
