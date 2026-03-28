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
    print_error,
    print_header,
    print_no_files,
    print_progress,
    print_results,
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


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Path to multi_agent.toml config file.")
@click.option("--hook-mode", is_flag=True, hidden=True,
              help="Running as a git pre-commit hook.")
@click.pass_context
def review(ctx: click.Context, config_path: str | None, hook_mode: bool) -> None:
    """Review staged files for consensus before committing."""
    try:
        repo_root = _resolve_repo(ctx)
    except Exception:
        print_error("Not in a git repository.")
        sys.exit(1)

    config = load_config(Path(config_path) if config_path else None)

    from multi_agent.consensus import run_consensus
    from multi_agent.context import get_staged_files, load_canon

    # Check for staged files early
    staged = get_staged_files(repo_root, config.general.file_patterns)
    if not staged:
        print_no_files()
        sys.exit(0)

    # Show header
    canon = load_canon(
        repo_root,
        config.general.canon_directories,
        config.general.file_patterns,
        config.general.max_canon_size_kb,
    )
    canon_size_kb = sum(len(v.encode()) for v in canon.values()) / 1024
    print_header([str(f) for f in staged], len(canon), canon_size_kb)

    # Run consensus
    result = asyncio.run(run_consensus(
        config, str(repo_root), on_progress=print_progress,
    ))

    print_results(result)

    sys.exit(0 if result.approved else 1)


@main.command("review-files")
@click.argument("files", nargs=-1, required=True)
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Path to multi_agent.toml config file.")
@click.pass_context
def review_files(ctx: click.Context, files: tuple[str, ...], config_path: str | None) -> None:
    """Review existing files (already committed or on disk).

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

    config = load_config(Path(config_path) if config_path else None)

    from multi_agent.consensus import resolve_file_args, run_file_review

    try:
        resolved = resolve_file_args(list(files), repo_root, config)
    except FileNotFoundError as exc:
        print_error(str(exc))
        sys.exit(1)

    print_header(resolved, 0, 0)

    result = asyncio.run(run_file_review(
        config, str(repo_root), resolved, on_progress=print_progress,
    ))

    print_results(result)

    sys.exit(0 if result.approved else 1)


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
def check_config(config_path: str | None) -> None:
    """Validate and display the current configuration."""
    try:
        config = load_config(Path(config_path) if config_path else None)
    except Exception as exc:
        print_error(f"Invalid configuration: {exc}")
        sys.exit(1)

    console.print("[bold]Configuration:[/bold]")
    console.print(f"  File patterns: {config.general.file_patterns}")
    console.print(f"  Consensus threshold: {config.general.consensus_threshold}")
    console.print(f"  Timeout: {config.general.timeout_seconds}s")
    console.print(f"  Canon directories: {config.general.canon_directories}")
    console.print()
    console.print("[bold]Agents:[/bold]")
    for name, agent in config.agents.items():
        status = "[green]enabled[/green]" if agent.enabled else "[red]disabled[/red]"
        model = agent.model or "default"
        console.print(f"  {name}: {status} (model: {model})")
