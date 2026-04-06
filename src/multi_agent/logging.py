"""Structured run logging to JSON files."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multi_agent.config import ResolvedRunConfig
    from multi_agent.models import IterationResult

LOG_DIR = ".multi_agent_runs"


def write_run_log(
    repo_root: Path,
    result: IterationResult,
    resolved: ResolvedRunConfig,
) -> Path:
    """Write a structured JSON log of a completed run.

    Returns the path to the written log file.
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    command = resolved.command_name or "review"
    filename = f"{timestamp}_{command}.json"
    log_path = repo_root / LOG_DIR / filename

    try:
        log_dir = repo_root / LOG_DIR
        log_dir.mkdir(exist_ok=True)

        # Serialize the full result, then strip merged_texts values
        data = dataclasses.asdict(result)
        data["files_modified"] = sorted(data.get("merged_texts", {}).keys())
        data.pop("merged_texts", None)

        log = {
            "version": 1,
            "timestamp": now.isoformat(),
            "command": command,
            "files_reviewed": result.files_reviewed,
            "config": _serialize_config(resolved),
            "proposals": data["proposals"],
            "rounds": data["rounds"],
            "consensus_reached": data["consensus_reached"],
            "stalled": data["stalled"],
            "best_round": data["best_round"],
            "best_approvals": data["best_approvals"],
            "dissents": data["dissents"],
            "files_modified": data["files_modified"],
            "total_usage": data["total_usage"],
            "total_duration_seconds": data["total_duration_seconds"],
        }

        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n")
    except Exception as exc:
        import sys
        print(f"Warning: failed to write run log: {exc}", file=sys.stderr)

    return log_path


def _serialize_config(resolved: ResolvedRunConfig) -> dict:
    """Extract loggable config summary from resolved config."""
    agents = {}
    for name, settings in resolved.agent_settings.items():
        agents[name] = dataclasses.asdict(settings)

    return {
        "max_rounds": resolved.max_rounds,
        "consensus_threshold": resolved.consensus_threshold,
        "min_severity": resolved.min_severity,
        "min_blocking_severity": resolved.min_blocking_severity,
        "backend": resolved.backend,
        "agents": agents,
    }
