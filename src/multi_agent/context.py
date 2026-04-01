"""Git diff extraction and canon loading."""

from __future__ import annotations

import difflib
import fnmatch
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multi_agent.consensus import AgentProposal, FileEdit


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


def load_canon(
    repo_root: Path,
    directories: list[str],
    patterns: list[str],
    max_size_kb: int = 500,
) -> dict[str, str]:
    """Load committed canon files for context.

    Returns {relative_path: file_contents} for all matching files
    in the specified directories. Reads the HEAD versions (committed state).
    """
    if not _has_commits(repo_root):
        return {}

    canon: dict[str, str] = {}
    total_size = 0
    max_bytes = max_size_kb * 1024

    # List all tracked files
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        capture_output=True, text=True, check=True, cwd=repo_root,
    )
    tracked_files = sorted(result.stdout.strip().splitlines())

    for filepath in tracked_files:
        # Check if file is in a canon directory
        in_canon_dir = any(
            filepath.startswith(d + "/") or filepath.startswith(d + "\\")
            for d in directories
        )
        if not in_canon_dir:
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

        canon[filepath] = content

    return canon


def count_uncommitted_canon(
    repo_root: Path,
    directories: list[str],
    patterns: list[str],
    committed_paths: set[str],
) -> int:
    """Count files on disk in canon directories that are not committed."""
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



def _canon_section(canon: dict[str, str]) -> str:
    """Build the canon context section of a prompt.

    Lists file paths and sizes instead of inlining content.
    Agents use the Read tool to explore files they consider relevant.
    """
    parts: list[str] = []
    if canon:
        parts.append("# EXISTING CANON (established files)\n")
        parts.append("These files form the established universe. Use the Read "
                      "tool to examine any files relevant to your review.\n\n")
        for path, content in sorted(canon.items()):
            size_kb = len(content.encode()) / 1024
            parts.append(f"- {path} ({size_kb:.1f} KB)\n")
    else:
        parts.append("# EXISTING CANON\n")
        parts.append("No prior canon exists yet. This appears to be the first "
                      "contribution. Focus on internal consistency.\n")
    return "".join(parts)



# --- Propose / Review prompt builders ---

_SEVERITY_ORDER = ["critical", "major", "minor", "suggestion"]


def _propose_instructions(min_severity: str, task: str | None = None) -> str:
    """Build propose instructions with severity threshold and optional task."""
    if task in ("expand", "contract", "custom"):
        # For task modes, severity filtering doesn't apply — the
        # system prompt already describes the goal.
        return (
            "\n# YOUR TASK\n"
            "Apply the task described in your system prompt to the content above. "
            "Use the Read tool to examine any canon files relevant to your review.\n\n"
            "Return your response as JSON.\n"
        )

    # Default: review for issues
    idx = _SEVERITY_ORDER.index(min_severity) if min_severity in _SEVERITY_ORDER else 2
    allowed = _SEVERITY_ORDER[:idx + 1]
    severity_note = ""
    if min_severity != "suggestion":
        severity_note = (
            f"\nIMPORTANT: Only propose edits for issues that are "
            f"{' or '.join(allowed)} severity. "
            f"Do NOT propose edits for purely stylistic or cosmetic issues"
            f"{' or minor concerns' if min_severity in ('critical', 'major') else ''}.\n"
        )
    return (
        "\n# YOUR TASK\n"
        "Review the content above from your specialty perspective and propose "
        "concrete edits. Use the Read tool to examine canon files as needed.\n"
        + severity_note
        + "\nReturn your response as JSON.\n"
    )

_REVIEW_ROUND_INSTRUCTIONS = (
    "\n# YOUR TASK\n"
    "Review ALL proposals above from your specialty perspective. For each edit, "
    "decide whether it is acceptable or needs modification.\n\n"
    "Use the Read tool to examine any canon files relevant to your review.\n\n"
    "Return your response as JSON.\n"
)


def build_propose_prompt(
    file_contents: dict[str, str],
    canon: dict[str, str],
    staged_diff: str | None = None,
    min_severity: str = "minor",
    task: str | None = None,
) -> str:
    """Assemble the prompt for the propose phase."""
    parts: list[str] = [_canon_section(canon)]

    parts.append("\n# FILES FOR REVIEW\n")
    parts.append("These are the files to review and propose edits for.\n")
    for path, content in sorted(file_contents.items()):
        parts.append(f"\n## {path}\n```\n{content}\n```\n")

    if staged_diff:
        parts.append("\n# DIFF (changes being made)\n")
        parts.append(f"```diff\n{staged_diff}\n```\n")

    parts.append(_propose_instructions(min_severity, task))
    return "".join(parts)


def build_review_round_prompt(
    proposals: list[AgentProposal],
    file_contents: dict[str, str],
    canon: dict[str, str],
    round_number: int,
) -> str:
    """Assemble the prompt for a review round."""
    from multi_agent.agents import AGENT_DISPLAY_NAMES

    parts: list[str] = [_canon_section(canon)]

    parts.append("\n# ORIGINAL FILE CONTENTS\n")
    for path, content in sorted(file_contents.items()):
        parts.append(f"\n## {path}\n```\n{content}\n```\n")

    parts.append(f"\n# PROPOSALS TO REVIEW (Round {round_number + 1})\n")
    for proposal in proposals:
        if not proposal.edits:
            continue
        display = AGENT_DISPLAY_NAMES.get(proposal.agent_name, proposal.agent_name)
        parts.append(f"\n## Proposals from {display} ({proposal.agent_name})\n")
        parts.append(f"Summary: {proposal.summary}\n")
        for i, edit in enumerate(proposal.edits):
            parts.append(f"\n### Edit {i} — {edit.file}\n")
            parts.append(f"**Original text:**\n```\n{edit.original_text}\n```\n")
            parts.append(f"**Replacement:**\n```\n{edit.replacement_text}\n```\n")
            parts.append(f"**Rationale:** {edit.rationale}\n")

    parts.append(_REVIEW_ROUND_INSTRUCTIONS)
    return "".join(parts)


def _apply_edits_to_text(content: str, edits: list[FileEdit]) -> str:
    """Apply a single agent's edits to text.

    Locates each edit by position, sorts, and applies back-to-front so
    earlier replacements don't shift later positions. Used internally by
    the merge module to build per-agent file versions.
    """
    located: list[tuple[int, int, str]] = []
    for edit in edits:
        pos = content.find(edit.original_text)
        if pos < 0:
            continue
        located.append((pos, pos + len(edit.original_text), edit.replacement_text))

    located.sort(key=lambda t: t[0])

    for start, end, replacement in reversed(located):
        content = content[:start] + replacement + content[end:]

    return content


def _group_edits_by_file(edits: list[FileEdit]) -> dict[str, list[FileEdit]]:
    """Group edits by their target file path."""
    by_file: dict[str, list[FileEdit]] = {}
    for edit in edits:
        by_file.setdefault(edit.file, []).append(edit)
    return by_file


def apply_edits(repo_root: Path, edits: list[FileEdit]) -> list[str]:
    """Apply edits to files in the working tree.

    Returns list of modified file paths.
    """
    modified = []
    for filepath, file_edits in sorted(_group_edits_by_file(edits).items()):
        full_path = repo_root / filepath
        content = full_path.read_text()
        content = _apply_edits_to_text(content, file_edits)
        full_path.write_text(content)
        modified.append(filepath)

    return modified


def build_diff_preview(edits: list[FileEdit], file_contents: dict[str, str]) -> str:
    """Generate a unified diff preview of what edits would produce."""
    edits_by_file = _group_edits_by_file(edits)

    diff_parts = []
    for filepath in sorted(edits_by_file.keys()):
        original = file_contents.get(filepath, "")
        modified = _apply_edits_to_text(original, edits_by_file[filepath])

        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
        )
        diff_parts.append("".join(diff))

    return "\n".join(part for part in diff_parts if part)


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
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
        )
        diff_parts.append("".join(diff))
    return "\n".join(part for part in diff_parts if part)


def apply_merged_texts(
    repo_root: Path,
    merged_texts: dict[str, str],
) -> list[str]:
    """Write merged texts to disk. Returns list of modified file paths."""
    modified = []
    for filepath in sorted(merged_texts.keys()):
        full_path = repo_root / filepath
        full_path.write_text(merged_texts[filepath])
        modified.append(filepath)
    return modified
