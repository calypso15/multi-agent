"""Git diff extraction and canon loading."""

from __future__ import annotations

import fnmatch
import subprocess
from pathlib import Path


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


_REVIEW_INSTRUCTIONS = (
    "\n# YOUR TASK\n"
    "Review the content above according to your specialty. "
    "You may use the Read, Glob, and Grep tools to explore the repository "
    "for additional context if needed.\n\n"
    "Return your structured verdict as JSON with:\n"
    '- "verdict": "APPROVE" or "REQUEST_CHANGES"\n'
    '- "summary": one-paragraph summary of your review\n'
    '- "issues": array of issues found (even if approving, note minor items)\n\n'
    "Each issue should have:\n"
    '- "severity": "critical", "major", "minor", or "suggestion"\n'
    '- "file": which file (if applicable)\n'
    '- "quote": the exact text with the issue\n'
    '- "issue": what is wrong\n'
    '- "suggestion": how to fix it\n\n'
    "APPROVE if no critical or major issues. "
    "REQUEST_CHANGES if any critical or major issue exists.\n"
)


def _canon_section(canon: dict[str, str]) -> str:
    """Build the canon context section of a prompt."""
    parts: list[str] = []
    if canon:
        parts.append("# EXISTING CANON (committed files)\n")
        parts.append("These are the established files in the universe. "
                      "Use them to check for consistency.\n")
        for path, content in sorted(canon.items()):
            parts.append(f"\n## {path}\n```\n{content}\n```\n")
    else:
        parts.append("# EXISTING CANON\n")
        parts.append("No prior canon exists yet. This appears to be the first "
                      "contribution. Focus on internal consistency.\n")
    return "".join(parts)


def build_review_prompt(
    staged_diff: str,
    staged_file_contents: dict[str, str],
    canon: dict[str, str],
) -> str:
    """Assemble the review prompt for staged changes."""
    parts: list[str] = [_canon_section(canon)]

    parts.append("\n# NEW/CHANGED CONTENT\n")
    parts.append("These are the files being submitted for review.\n")
    for path, content in sorted(staged_file_contents.items()):
        parts.append(f"\n## {path}\n```\n{content}\n```\n")

    parts.append("\n# DIFF\n")
    parts.append("The exact changes being made:\n")
    parts.append(f"```diff\n{staged_diff}\n```\n")

    parts.append(_REVIEW_INSTRUCTIONS)
    return "".join(parts)


def build_file_review_prompt(
    target_files: list[str],
    canon_files: list[str],
) -> str:
    """Assemble the review prompt for existing files.

    Instead of inlining all content (which can be huge), we list the file
    paths and instruct agents to read them using their tools.
    """
    parts: list[str] = []

    if canon_files:
        other_canon = sorted(set(canon_files) - set(target_files))
        if other_canon:
            parts.append("# EXISTING CANON\n")
            parts.append("These files form the established universe. Use Read, "
                          "Glob, and Grep to cross-reference as needed:\n\n")
            for path in other_canon:
                parts.append(f"- {path}\n")
    else:
        parts.append("# EXISTING CANON\n")
        parts.append("No prior canon exists yet. Focus on internal consistency.\n")

    parts.append("\n# FILES TO REVIEW\n")
    parts.append("Review the following files according to your specialty. "
                  "Use the Read tool to read each file, and Grep/Glob to "
                  "search the repository for cross-references.\n\n")
    for path in sorted(target_files):
        parts.append(f"- {path}\n")

    parts.append(_REVIEW_INSTRUCTIONS)
    return "".join(parts)
