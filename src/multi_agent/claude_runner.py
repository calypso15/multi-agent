"""Claude CLI subprocess management with streaming and error handling."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from multi_agent.models import TokenUsage, extract_usage, unwrap_result


@dataclass
class ClaudeResult:
    """Parsed result from a claude CLI invocation."""
    returncode: int
    result_json: dict[str, Any] | None
    stdout: str
    stderr: str


def _make_tool_callback(
    agent_name: str,
    on_progress: Callable[[str, str], None] | None,
) -> Callable[[str, str], None] | None:
    """Wrap on_progress to format tool-use events for an agent."""
    if not on_progress:
        return None

    def callback(tool_name: str, summary: str):
        detail = f" {summary}" if summary else ""
        on_progress(agent_name, f"  \u2192 {tool_name}{detail}")

    return callback


def _summarize_tool_call(tool_name: str, tool_input: dict) -> str:
    """Build a short human-readable summary of a tool call."""
    if tool_name in ("Read", "read_file"):
        path = tool_input.get("file_path", tool_input.get("path", ""))
        name = path.split("/")[-1] if "/" in str(path) else str(path)
        offset = tool_input.get("offset")
        limit = tool_input.get("limit")
        if offset or limit:
            parts = []
            if offset:
                parts.append(f"offset={offset}")
            if limit:
                parts.append(f"limit={limit}")
            name += f" ({', '.join(parts)})"
        return name
    if tool_name in ("WebSearch", "web_search"):
        query = tool_input.get("query", tool_input.get("search_query", ""))
        return f'"{query}"' if query else ""
    if tool_name in ("WebFetch", "web_fetch"):
        url = tool_input.get("url", "")
        return url[:60] if url else ""
    if tool_name in ("Grep", "grep"):
        pattern = tool_input.get("pattern", "")
        return f'"{pattern}"' if pattern else ""
    if tool_name in ("Glob", "glob"):
        pattern = tool_input.get("pattern", "")
        return pattern
    # Fallback: show first string value
    for v in tool_input.values():
        if isinstance(v, str) and v:
            return v[:40]
    return ""


async def _drive_process(
    proc: asyncio.subprocess.Process,
    prompt: str,
    on_tool_use: Callable[[str, str], None] | None = None,
) -> ClaudeResult:
    """Drive an already-created subprocess: send prompt, stream events, return result.

    Reads stdout line-by-line for stream-json events. Reports tool_use events
    via on_tool_use(tool_name, summary). Returns the final result event's JSON.
    """
    proc.stdin.write(prompt.encode())
    await proc.stdin.drain()
    proc.stdin.close()

    result_json: dict[str, Any] | None = None
    all_lines: list[str] = []
    seen_tool_ids: set[str] = set()

    while True:
        line_bytes = await proc.stdout.readline()
        if not line_bytes:
            break
        line = line_bytes.decode().strip()
        if not line:
            continue
        all_lines.append(line)

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")

        if event_type == "assistant" and on_tool_use:
            content = event.get("message", {}).get("content", [])
            for item in content:
                if item.get("type") == "tool_use":
                    tool_id = item.get("id")
                    if tool_id is None:
                        tool_name = item.get("name", "unknown")
                        tool_input = item.get("input", {})
                        tool_id = f"{tool_name}:{json.dumps(tool_input, sort_keys=True)}"
                    if tool_id in seen_tool_ids:
                        continue
                    seen_tool_ids.add(tool_id)
                    tool_name = item.get("name", "unknown")
                    tool_input = item.get("input", {})
                    summary = _summarize_tool_call(tool_name, tool_input)
                    on_tool_use(tool_name, summary)

        elif event_type == "result":
            result_json = event

    # Read stderr with a timeout to avoid hanging on large output
    try:
        stderr_bytes = await asyncio.wait_for(proc.stderr.read(), timeout=10)
    except asyncio.TimeoutError:
        stderr_bytes = b"(stderr read timed out)"

    await proc.wait()

    return ClaudeResult(
        returncode=proc.returncode,
        result_json=result_json,
        stdout="\n".join(all_lines),
        stderr=stderr_bytes.decode() if stderr_bytes else "",
    )


async def spawn_claude(
    cli_args: list[str],
    prompt: str,
    cwd: str,
    on_tool_use: Callable[[str, str], None] | None = None,
) -> ClaudeResult:
    """Spawn a claude CLI process, stream events, and return the result.

    With --output-format stream-json, reads events line by line.
    Reports tool_use events via on_tool_use(tool_name, summary).
    Returns the final result event's JSON.
    """
    proc = await asyncio.create_subprocess_exec(
        *cli_args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        limit=10 * 1024 * 1024,  # 10 MB -- stream-json lines can be large
    )
    return await _drive_process(proc, prompt, on_tool_use)


@dataclass
class AgentResult:
    """Raw result from spawning an agent."""
    output: dict[str, Any] | None
    usage: TokenUsage
    duration_seconds: float
    error: str | None = None


async def run_agent(
    agent_name: str,
    prompt: str,
    cli_args: list[str],
    repo_root: str,
    timeout_seconds: int,
    on_progress: Callable[[str, str], None] | None = None,
    progress_label: str = "running",
    report_tool_use: bool = True,
) -> AgentResult:
    """Spawn a claude CLI agent and return the parsed result or error.

    Common boilerplate for propose, review, dissent, and arbitration phases.
    Kills the subprocess on timeout to prevent zombie processes.
    """
    start = time.monotonic()
    proc: asyncio.subprocess.Process | None = None

    if on_progress:
        on_progress(agent_name, progress_label)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cli_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=repo_root,
            limit=10 * 1024 * 1024,
        )

        result = await asyncio.wait_for(
            _drive_process(
                proc, prompt,
                on_tool_use=_make_tool_callback(agent_name, on_progress) if report_tool_use else None,
            ),
            timeout=timeout_seconds,
        )

        usage = extract_usage(result.result_json) if result.result_json else TokenUsage()

        if result.returncode != 0:
            stderr_text = result.stderr.strip() if result.stderr else "unknown error"
            return AgentResult(
                output=None, usage=usage,
                duration_seconds=time.monotonic() - start,
                error=stderr_text[:500],
            )

        output = unwrap_result(result.result_json)
        if output is None:
            return AgentResult(
                output=None, usage=usage,
                duration_seconds=time.monotonic() - start,
                error=f"Unparseable output: {result.stdout[:300]}",
            )

        return AgentResult(
            output=output, usage=usage,
            duration_seconds=time.monotonic() - start,
        )

    except asyncio.TimeoutError:
        if proc and proc.returncode is None:
            proc.kill()
            await proc.wait()
        return AgentResult(
            output=None, usage=TokenUsage(),
            duration_seconds=time.monotonic() - start,
            error=f"Timed out after {timeout_seconds}s",
        )
    except Exception as exc:
        if proc and proc.returncode is None:
            proc.kill()
            await proc.wait()
        return AgentResult(
            output=None, usage=TokenUsage(),
            duration_seconds=time.monotonic() - start,
            error=str(exc),
        )
