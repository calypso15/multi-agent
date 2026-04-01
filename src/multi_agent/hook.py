"""Git hook installer and uninstaller."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

HOOK_MARKER = "# multi-agent-consensus-hook"

HOOK_SCRIPT_TEMPLATE = """\
#!/usr/bin/env bash
{hook_marker}
# Multi-agent consensus pre-commit hook
# Installed by: python -m multi_agent install-hook

PYTHON="{python_path}"

# Check that multi_agent is available
if ! "$PYTHON" -c "import multi_agent" 2>/dev/null; then
    echo "Error: multi_agent package not found." >&2
    echo "Install it with: pip install -e /path/to/multi-agent" >&2
    exit 1
fi

echo "Running multi-agent fiction review..."
"$PYTHON" -m multi_agent review --hook-mode
exit $?
"""


def install_hook(repo_root: Path) -> None:
    """Install the pre-commit hook."""
    hooks_dir = repo_root / ".git" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / "pre-commit"

    # Back up existing hook if it exists and isn't ours
    if hook_path.exists():
        content = hook_path.read_text()
        if HOOK_MARKER in content:
            # Already installed, overwrite
            pass
        else:
            backup = hook_path.with_suffix(".bak")
            shutil.copy2(hook_path, backup)

    python_path = sys.executable
    script = HOOK_SCRIPT_TEMPLATE.format(
        hook_marker=HOOK_MARKER, python_path=python_path,
    )
    hook_path.write_text(script)
    os.chmod(hook_path, 0o755)


def uninstall_hook(repo_root: Path) -> None:
    """Remove the pre-commit hook if it's ours."""
    hook_path = repo_root / ".git" / "hooks" / "pre-commit"

    if not hook_path.exists():
        return

    content = hook_path.read_text()
    if HOOK_MARKER not in content:
        return

    hook_path.unlink()

    # Restore backup if one exists
    backup = hook_path.with_suffix(".bak")
    if backup.exists():
        shutil.move(str(backup), str(hook_path))
