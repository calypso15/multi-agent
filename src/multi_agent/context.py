"""Git diff extraction and reference file loading."""

from __future__ import annotations

import difflib
import fnmatch
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from multi_agent.models import SEVERITY_ORDER

if TYPE_CHECKING:
    from multi_agent.models import AgentProposal, FileEdit


def find_git_root(start: Path | None = None) -> Path:
    """Find the root of a git repository.

    If start is given, finds the repo containing that path.
    Otherwise uses the current working directory.
    """
    kwargs: dict = {"capture_output": True, "text": True, "check": True}
    if start is not None:
        kwargs["cwd"] = str(start)
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], **kwargs,
    )
    return Path(result.stdout.strip())


def _has_commits(repo_root: Path) -> bool:
    """Check if the repo has any commits."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=repo_root,
    )
    return result.returncode == 0


def get_staged_files(repo_root: Path, patterns: list[str]) -> list[Path]:
    """Get staged files matching the configured patterns.

    Returns paths relative to repo root.
    """
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True, check=True, cwd=repo_root,
    )
    all_files = [f for f in result.stdout.strip().splitlines() if f]

    matched = []
    for filepath in all_files:
        name = Path(filepath).name
        if any(fnmatch.fnmatch(name, pat) for pat in patterns):
            matched.append(Path(filepath))

    return matched


def get_staged_diff(repo_root: Path, patterns: list[str]) -> str:
    """Get the unified diff for staged files matching patterns."""
    staged = get_staged_files(repo_root, patterns)
    if not staged:
        return ""

    result = subprocess.run(
        ["git", "diff", "--cached", "--"] + [str(f) for f in staged],
        capture_output=True, text=True, check=True, cwd=repo_root,
    )
    return result.stdout


def get_staged_content(repo_root: Path, filepath: Path) -> str:
    """Read a file's content from the staging area (index)."""
    result = subprocess.run(
        ["git", "show", f":{filepath}"],
        capture_output=True, text=True, check=True, cwd=repo_root,
    )
    return result.stdout


def load_reference(
    repo_root: Path,
    directories: list[str],
    patterns: list[str],
    max_size_kb: int = 500,
) -> dict[str, str]:
    """Load committed reference files for context.

    Returns {relative_path: file_contents} for all matching files
    in the specified directories. Reads the HEAD versions (committed state).
    """
    if not _has_commits(repo_root):
        return {}

    reference: dict[str, str] = {}
    total_size = 0
    max_bytes = max_size_kb * 1024

    # List all tracked files
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        capture_output=True, text=True, check=True, cwd=repo_root,
    )
    tracked_files = sorted(result.stdout.strip().splitlines())

    for filepath in tracked_files:
        # Check if file is in a reference directory
        in_ref_dir = any(
            filepath.startswith(d + "/") or filepath.startswith(d + "\\")
            for d in directories
        )
        if not in_ref_dir:
            continue

        # Check if file matches patterns
        name = Path(filepath).name
        if not any(fnmatch.fnmatch(name, pat) for pat in patterns):
            continue

        # Read committed content
        content_result = subprocess.run(
            ["git", "show", f"HEAD:{filepath}"],
            capture_output=True, text=True, cwd=repo_root,
        )
        if content_result.returncode != 0:
            continue

        content = content_result.stdout
        total_size += len(content.encode())
        if total_size > max_bytes:
            break

        reference[filepath] = content

    return reference


def count_uncommitted_reference(
    repo_root: Path,
    directories: list[str],
    patterns: list[str],
    committed_paths: set[str],
) -> int:
    """Count files on disk in reference directories that are not committed."""
    count = 0
    for d in directories:
        dir_path = repo_root / d
        if not dir_path.is_dir():
            continue
        for pattern in patterns:
            for path in dir_path.rglob(pattern):
                rel = str(path.relative_to(repo_root))
                if rel not in committed_paths:
                    count += 1
    return count


def resolve_file_args(
    file_paths: list[str],
    repo_root: Path,
    file_patterns: list[str],
) -> list[str]:
    """Resolve file arguments to relative paths, expanding directories."""
    resolved: list[Path] = []
    for fp in file_paths:
        abs_path = Path(fp) if Path(fp).is_absolute() else repo_root / fp
        if abs_path.is_dir():
            for pattern in file_patterns:
                resolved.extend(sorted(abs_path.rglob(pattern)))
        elif abs_path.is_file():
            resolved.append(abs_path)
        else:
            raise FileNotFoundError(f"File not found: {fp}")

    if not resolved:
        raise FileNotFoundError(
            f"No files matching {file_patterns} found in: "
            + ", ".join(file_paths)
        )

    return [str(p.relative_to(repo_root)) for p in resolved]


def _reference_section(reference: dict[str, str]) -> str:
    """Build the reference context section of a prompt.

    Lists file paths and sizes instead of inlining content.
    Agents use the Read tool to explore files they consider relevant.
    """
    parts: list[str] = []
    if reference:
        parts.append("# REFERENCE FILES (established context)\n")
        parts.append("These files are the authoritative reference material. Use the Read "
                      "tool to examine any files relevant to your review.\n\n")
        for path, content in sorted(reference.items()):
            size_kb = len(content.encode()) / 1024
            parts.append(f"- {path} ({size_kb:.1f} KB)\n")
    else:
        parts.append("# REFERENCE FILES\n")
        parts.append("No reference files exist yet. This appears to be the first "
                      "contribution. Focus on internal consistency.\n")
    return "".join(parts)



# --- Propose / Review prompt builders ---

def _propose_instructions(min_severity: str) -> str:
    """Build propose instructions with severity threshold."""
    idx = SEVERITY_ORDER.index(min_severity) if min_severity in SEVERITY_ORDER else 2
    allowed = SEVERITY_ORDER[:idx + 1]
    severity_note = ""
    if min_severity != "suggestion":
        severity_note = (
            f"\nEdits below {min_severity} severity will be filtered out. "
            f"Focus on issues that are {' or '.join(allowed)} severity.\n"
        )
    return (
        "\n# YOUR TASK\n"
        "Apply the task described in your system prompt to the content above. "
        "Use the Read tool to examine any reference files relevant to your review.\n"
        + severity_note
        + "\nClassify each edit's severity. Return your response as JSON.\n"
    )

_REVIEW_ROUND_INSTRUCTIONS = (
    "\n# YOUR TASK\n"
    "Review ALL proposals above from your specialty perspective. For each edit, "
    "decide whether it is acceptable or needs modification.\n\n"
    "Use the Read tool to examine any reference files relevant to your review.\n\n"
    "Return your response as JSON.\n"
)


def build_propose_prompt(
    file_contents: dict[str, str],
    reference: dict[str, str],
    staged_diff: str | None = None,
    min_severity: str = "minor",
    propose_instructions: str | None = None,
) -> str:
    """Assemble the prompt for the propose phase."""
    parts: list[str] = [_reference_section(reference)]

    if propose_instructions:
        parts.append("\n# FILES\n")
    else:
        parts.append("\n# FILES FOR REVIEW\n")
        parts.append("These are the files to review and propose edits for.\n")
    for path, content in sorted(file_contents.items()):
        parts.append(f"\n## {path}\n```\n{content}\n```\n")

    if staged_diff:
        parts.append("\n# DIFF (changes being made)\n")
        parts.append(f"```diff\n{staged_diff}\n```\n")

    parts.append(propose_instructions or _propose_instructions(min_severity))
    return "".join(parts)


def build_review_round_prompt(
    proposals: list[AgentProposal],
    file_contents: dict[str, str],
    reference: dict[str, str],
    round_number: int,
    display_names: dict[str, str] | None = None,
    skip_edits: set[tuple[str, int]] | None = None,
) -> str:
    """Assemble the prompt for a review round.

    skip_edits: set of (agent_name, edit_index) pairs to omit from the
    prompt while preserving original edit indices.
    """
    _display_names = display_names or {}
    _skip = skip_edits or set()

    parts: list[str] = [_reference_section(reference)]

    parts.append("\n# FILES UNDER REVIEW (drafts being improved — not authoritative)\n")
    parts.append("These files are the subject of editing. They may contain errors, "
                  "inconsistencies, or outdated terminology that the proposed edits "
                  "aim to fix. Only files in the reference directories are authoritative.\n")
    for path, content in sorted(file_contents.items()):
        parts.append(f"\n## {path}\n```\n{content}\n```\n")

    parts.append(f"\n# PROPOSALS TO REVIEW (Round {round_number + 1})\n")
    for proposal in proposals:
        visible = [
            (i, edit) for i, edit in enumerate(proposal.edits)
            if (proposal.agent_name, i) not in _skip
        ]
        if not visible:
            continue
        display = _display_names.get(proposal.agent_name, proposal.agent_name)
        parts.append(f"\n## Proposals from {display} ({proposal.agent_name})\n")
        parts.append(f"Summary: {proposal.summary}\n")
        for i, edit in visible:
            parts.append(f"\n### Edit {i} ({edit.severity}) \u2014 {edit.file}\n")
            parts.append(f"**Original text:**\n```\n{edit.original_text}\n```\n")
            parts.append(f"**Replacement:**\n```\n{edit.replacement_text}\n```\n")
            parts.append(f"**Rationale:** {edit.rationale}\n")

    parts.append(_REVIEW_ROUND_INSTRUCTIONS)
    return "".join(parts)


_EDIT_REVIEW_INSTRUCTIONS = (
    "\n# YOUR TASK\n"
    "Review the proposed edit above from your specialty perspective. "
    "Decide whether it is acceptable or needs modification.\n\n"
    "Use the Read tool to examine any reference files relevant to your review.\n\n"
    "Return your response as JSON.\n"
)


def build_edit_review_prompt(
    edit: FileEdit,
    proposer_display: str,
    file_contents: dict[str, str],
    reference: dict[str, str],
    round_number: int,
) -> str:
    """Assemble the prompt for reviewing a single edit."""
    parts: list[str] = [_reference_section(reference)]

    parts.append("\n# FILE UNDER REVIEW (draft being improved — not authoritative)\n")
    content = file_contents.get(edit.file, "")
    parts.append(f"\n## {edit.file}\n```\n{content}\n```\n")

    parts.append(f"\n# PROPOSED EDIT TO REVIEW (Round {round_number + 1})\n")
    parts.append(f"Proposed by: {proposer_display}\n")
    parts.append(f"Severity: {edit.severity}\n")
    parts.append(f"Rationale: {edit.rationale}\n")
    parts.append(f"\n**Original text:**\n```\n{edit.original_text}\n```\n")
    parts.append(f"**Replacement:**\n```\n{edit.replacement_text}\n```\n")

    parts.append(_EDIT_REVIEW_INSTRUCTIONS)
    return "".join(parts)


def _build_unified_diff(
    filepath: str,
    original: str,
    modified: str,
) -> str:
    """Generate a unified diff between original and modified text for a file."""
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
    )
    return "".join(diff)


def build_diff_preview_from_merged(
    merged_texts: dict[str, str],
    file_contents: dict[str, str],
) -> str:
    """Generate a unified diff preview from pre-merged texts."""
    diff_parts = []
    for filepath in sorted(merged_texts.keys()):
        original = file_contents.get(filepath, "")
        modified = merged_texts[filepath]
        if original == modified:
            continue
        diff_parts.append(_build_unified_diff(filepath, original, modified))
    return "\n".join(part for part in diff_parts if part)


def apply_merged_texts(
    repo_root: Path,
    merged_texts: dict[str, str],
) -> list[str]:
    """Write merged texts to disk. Returns list of modified file paths."""
    modified = []
    resolved_root = repo_root.resolve()
    for filepath in sorted(merged_texts.keys()):
        full_path = (repo_root / filepath).resolve()
        if not full_path.is_relative_to(resolved_root):
            raise ValueError(f"Path escapes repo root: {filepath}")
        full_path.write_text(merged_texts[filepath])
        modified.append(filepath)
    return modified
