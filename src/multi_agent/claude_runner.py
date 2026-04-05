"""Claude CLI subprocess management with streaming and error handling."""

from __future__ import annotations

import asyncio
import json
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from multi_agent.backend import AgentResult
from multi_agent.models import TokenUsage, extract_usage, unwrap_result

# After a timeout, SIGINT the process and try to resume the session
# for partial results before giving up.
_INTERRUPT_DRAIN_TIMEOUT = 10  # seconds to wait for process exit after SIGINT
_GRACE_TIMEOUT = 60  # seconds for the resumed session to return results

class _MaxTurnsExceeded(Exception):
    """Raised by _drive_process when the agent exceeds its turn limit."""


_RESUME_PROMPT = """\
You have run out of time. Immediately return your response as JSON. \
Use whatever information you have gathered so far. Do not make any more \
tool calls.

Return JSON matching this EXACT schema:
{
  "summary": "one-paragraph summary of your proposed changes",
  "edits": [
    {
      "file": "relative file path",
      "original_text": "exact verbatim text from the file to replace",
      "replacement_text": "your proposed replacement",
      "rationale": "why this edit is needed"
    }
  ]
}

If you were reviewing proposals instead of proposing, return:
{
  "all_approved": true or false,
  "summary": "summary of your review",
  "proposal_reviews": []
}
"""


@dataclass
class ClaudeResult:
    """Parsed result from a claude CLI invocation."""
    returncode: int
    result_json: dict[str, Any] | None
    stdout: str
    stderr: str
    turns_taken: int = 0
    tool_usage: dict[str, int] = field(default_factory=dict)


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


def _make_turn_callback(
    agent_name: str,
    on_progress: Callable[[str, str], None] | None,
) -> Callable[[int], None] | None:
    """Wrap on_progress to report turn changes (verbose only).

    Turn counts come from ``message_stop`` stream events, which mark
    completed turn boundaries.  The final count is overwritten by
    ``num_turns`` from the CLI result event when available.
    """
    if not on_progress:
        return None
    from multi_agent.output import is_verbose
    if not is_verbose():
        return None

    def callback(turn_number: int):
        on_progress(agent_name, f"  turn {turn_number}")

    return callback


def _make_drafting_callback(
    agent_name: str,
    on_progress: Callable[[str, str], None] | None,
) -> Callable[[int], None] | None:
    """Wrap on_progress to report once per turn when text generation begins.

    Reports "drafting" on the first text delta of each content block.
    Resets when a new content block starts, so multi-turn agents
    (tool call → draft → tool call → draft) get one report per turn.
    """
    if not on_progress:
        return None

    state = {"reported": False}

    def callback(chars: int):
        if chars == 0:
            # content_block_start resets chars to 0 — new turn
            state["reported"] = False
            return
        if state["reported"]:
            return
        state["reported"] = True
        on_progress(agent_name, "  \u270e drafting")

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
    on_drafting: Callable[[int], None] | None = None,
    on_turn: Callable[[int], None] | None = None,
    max_turns: int = 0,
) -> ClaudeResult:
    """Drive an already-created subprocess: send prompt, stream events, return result.

    Reads stdout line-by-line for stream-json events. Reports tool_use events
    via on_tool_use(tool_name, summary). Reports text generation progress via
    on_drafting(chars_so_far). Returns the final result event's JSON.

    When *max_turns* > 0, raises ``_MaxTurnsExceeded`` after the agent
    completes that many turns, allowing the caller to interrupt and resume.
    """
    proc.stdin.write(prompt.encode())
    await proc.stdin.drain()
    proc.stdin.close()

    result_json: dict[str, Any] | None = None
    all_lines: list[str] = []
    seen_tool_ids: set[str] = set()
    draft_chars = 0  # characters generated in current text block
    turns = 0
    tool_counts: dict[str, int] = {}

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

        if event_type == "assistant":
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
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                    if on_tool_use:
                        tool_input = item.get("input", {})
                        summary = _summarize_tool_call(tool_name, tool_input)
                        on_tool_use(tool_name, summary)

        elif event_type == "stream_event":
            inner = event.get("event", {})
            inner_type = inner.get("type", "")
            if inner_type == "message_start":
                # Report the turn before its content streams.
                turns += 1
                if on_turn:
                    on_turn(turns)
            elif inner_type == "message_stop":
                # Enforce the limit after the turn completes so we
                # don't interrupt mid-turn.
                if max_turns > 0 and turns >= max_turns:
                    raise _MaxTurnsExceeded()
            elif inner_type == "content_block_start":
                draft_chars = 0
                if on_drafting:
                    on_drafting(0)  # signal new block for per-turn reset
            elif inner_type == "content_block_delta":
                if on_drafting:
                    delta = inner.get("delta", {})
                    if delta.get("type") == "text_delta":
                        draft_chars += len(delta.get("text", ""))
                        on_drafting(draft_chars)

        elif event_type == "result":
            # The result event includes the CLI's own num_turns count.
            turns = event.get("num_turns", turns)
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
        turns_taken=turns,
        tool_usage=tool_counts,
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


# --- Graceful timeout: SIGINT → drain session_id → resume for results ---


async def _drain_after_interrupt(
    proc: asyncio.subprocess.Process,
) -> tuple[str | None, TokenUsage]:
    """After SIGINT, wait for process exit and extract session_id + usage.

    The Claude CLI emits a final ``result`` event on SIGINT that contains
    the ``session_id`` needed to resume the conversation.
    """
    try:
        await asyncio.wait_for(proc.wait(), timeout=_INTERRUPT_DRAIN_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return None, TokenUsage()

    # Read whatever remains in stdout after the process exited
    session_id = None
    usage = TokenUsage()
    try:
        remaining = await asyncio.wait_for(proc.stdout.read(), timeout=5)
        for line in remaining.decode().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "result":
                    session_id = event.get("session_id")
                    usage = extract_usage(event)
            except json.JSONDecodeError:
                continue
    except asyncio.TimeoutError:
        pass

    return session_id, usage


async def _resume_for_results(
    session_id: str,
    cwd: str,
) -> AgentResult | None:
    """Resume a timed-out session and ask it to return results immediately.

    Spawns ``claude --resume <session_id>`` with a prompt that tells the
    agent to stop doing tool calls and return whatever it has as JSON.
    """
    args = [
        "claude",
        "--resume", session_id,
        "--print",
        "--output-format", "stream-json",
        "--verbose",
        "--max-turns", "1",
        "--permission-mode", "bypassPermissions",
    ]

    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            limit=10 * 1024 * 1024,
        )

        result = await asyncio.wait_for(
            _drive_process(proc, _RESUME_PROMPT),
            timeout=_GRACE_TIMEOUT,
        )

        usage = extract_usage(result.result_json) if result.result_json else TokenUsage()

        if result.returncode != 0:
            return None

        output = unwrap_result(result.result_json)

        if output is not None:
            return AgentResult(output=output, usage=usage, duration_seconds=0)
        return None

    except asyncio.TimeoutError:
        if proc and proc.returncode is None:
            proc.kill()
            await proc.wait()
        return None
    except Exception:
        if proc and proc.returncode is None:
            proc.kill()
            await proc.wait()
        return None


# --- CLI argument building (Claude CLI specific) ---


KNOWN_TOOLS = {
    "Bash", "Read", "Glob", "Grep", "Edit", "Write",
    "Agent", "Skill", "ToolSearch", "WebSearch", "WebFetch",
}


def build_cli_args(
    agent_name: str,
    system_prompt: str,
    model: str | None,
    repo_root: str,
    allowed_tools: list[str] | None = None,
) -> list[str]:
    """Build command-line arguments for a ``claude`` CLI invocation.

    Turn limits (``max_turns``) are enforced by ``_drive_process`` at
    the stream level, not via a CLI flag.
    """
    args = [
        "claude",
        "--print",                       # Non-interactive, print result
        "--output-format", "stream-json", # Stream JSON events for tool visibility
        "--verbose",                     # Required for stream-json with --print
        "--include-partial-messages",    # Stream deltas for progress reporting
    ]

    args += [
        "--system-prompt", system_prompt,
        "--permission-mode", "bypassPermissions",
    ]

    # Read is always available so agents can explore reference files.
    # All other tools are disabled unless explicitly in allowed_tools.
    effective_tools = {"Read"} | set(allowed_tools or [])
    disallowed = KNOWN_TOOLS - effective_tools
    if disallowed:
        args.extend(["--disallowedTools", ",".join(sorted(disallowed))])
    args.extend(["--allowedTools", ",".join(sorted(effective_tools))])

    if model:
        args.extend(["--model", model])

    return args


# --- Main agent runner ---


async def _run_agent_impl(
    agent_name: str,
    prompt: str,
    cli_args: list[str],
    repo_root: str,
    timeout_seconds: int,
    on_progress: Callable[[str, str], None] | None = None,
    progress_label: str = "running",
    report_tool_use: bool = True,
    max_turns: int = 0,
) -> AgentResult:
    """Spawn a claude CLI agent and return the parsed result or error.

    On timeout or turn-limit, attempts a graceful recovery:
    1. SIGINT the process so the CLI exits cleanly with a session_id.
    2. Resume the session with a "return your results now" prompt.
    3. Parse the resumed output normally.

    Falls back to a hard error if resume fails or no session_id is available.
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
                on_drafting=_make_drafting_callback(agent_name, on_progress) if report_tool_use else None,
                on_turn=_make_turn_callback(agent_name, on_progress),
                max_turns=max_turns,
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
                turns_taken=result.turns_taken,
                tool_usage=result.tool_usage,
            )

        output = unwrap_result(result.result_json)
        if output is None:
            return AgentResult(
                output=None, usage=usage,
                duration_seconds=time.monotonic() - start,
                error=f"Unparseable output: {result.stdout[:300]}",
                turns_taken=result.turns_taken,
                tool_usage=result.tool_usage,
            )

        return AgentResult(
            output=output, usage=usage,
            duration_seconds=time.monotonic() - start,
            turns_taken=result.turns_taken,
            tool_usage=result.tool_usage,
        )

    except (asyncio.TimeoutError, _MaxTurnsExceeded) as limit_exc:
        is_turn_limit = isinstance(limit_exc, _MaxTurnsExceeded)
        reason = (
            f"turn limit ({max_turns})" if is_turn_limit
            else f"timeout ({timeout_seconds}s)"
        )

        # Phase 1: SIGINT for clean shutdown, drain session_id
        session_id = None
        interrupted_usage = TokenUsage()

        if proc and proc.returncode is None:
            proc.send_signal(signal.SIGINT)
            session_id, interrupted_usage = await _drain_after_interrupt(proc)

        # Phase 2: Resume the session for partial results
        if session_id:
            if on_progress:
                on_progress(agent_name, f"{reason} \u2014 resuming for partial results")

            resumed = await _resume_for_results(session_id, repo_root)
            if resumed and resumed.output:
                total_usage = interrupted_usage
                total_usage += resumed.usage
                return AgentResult(
                    output=resumed.output,
                    usage=total_usage,
                    duration_seconds=time.monotonic() - start,
                )

            if on_progress:
                on_progress(agent_name, "resume failed \u2014 no partial results")

        # Hard failure — no recovery possible
        if proc and proc.returncode is None:
            proc.kill()
            await proc.wait()
        return AgentResult(
            output=None, usage=interrupted_usage,
            duration_seconds=time.monotonic() - start,
            error=f"Exceeded {reason}",
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


class ClaudeCliBackend:
    """AgentBackend implementation using the Claude CLI subprocess."""

    async def run_agent(
        self,
        agent_name: str,
        prompt: str,
        system_prompt: str,
        repo_root: str,
        timeout_seconds: int,
        *,
        model: str | None = None,
        max_turns: int = 0,
        allowed_tools: list[str] | None = None,
        on_progress: Callable[[str, str], None] | None = None,
        progress_label: str = "running",
        report_tool_use: bool = True,
    ) -> AgentResult:
        cli_args = build_cli_args(
            agent_name, system_prompt, model, repo_root,
            allowed_tools=allowed_tools,
        )
        return await _run_agent_impl(
            agent_name, prompt, cli_args, repo_root,
            timeout_seconds, on_progress, progress_label, report_tool_use,
            max_turns=max_turns,
        )
