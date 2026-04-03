# CLAUDE.md

## Project overview

Multi-agent consensus review system. Agents are defined in `multi_agent.toml` (not in Python source). The system spawns Claude CLI subprocesses to run each agent.

## Commands

```bash
source .venv/bin/activate
python -m pytest tests/ -v          # run tests
python -m multi_agent check-config   # validate config (needs --repo or a multi_agent.toml above cwd)
```

## Architecture

- **`backend.py`** — `AgentBackend` protocol and `AgentResult` dataclass. Defines the abstract interface for running agents — any backend (CLI, SDK, HTTP) implements `run_agent()` and returns `AgentResult`.
- **`config.py`** — TOML loading, `AgentConfig`/`GeneralConfig`/`CommandConfig` dataclasses, validation. Agents are defined via `[agents.<name>]` blocks, commands via `[commands.<name>]` blocks. `DEFAULT_REVIEW_COMMAND` provides a fallback review command. `get_display_name()` derives display names. `GeneralConfig.backend` selects the LLM backend (`"claude-cli"` by default).
- **`agents.py`** — Internal mode suffixes (review/dissent), `build_command_mode_suffix()` for TOML-defined commands, JSON output schemas, `build_agent_system_prompt()`, `build_name_normalizer()`. No agent-specific content — all generic.
- **`claude_runner.py`** — `ClaudeCliBackend` (implements `AgentBackend`), Claude CLI subprocess management, `build_cli_args()`, `KNOWN_TOOLS`, stream-json parsing, timeout/resume recovery.
- **`consensus.py`** — Orchestration: propose phase, review phase, iteration loop, arbitration, dissent collection. Receives an `AgentBackend` instance and calls `backend.run_agent()` for all agent interactions.
- **`context.py`** — Builds user prompts (file contents, reference file listing, review round proposals). Separate from system prompts.
- **`models.py`** — Dataclasses (`FileEdit`, `AgentProposal`, `AgentReviewResponse`, etc.), JSON parsing, path sanitization.
- **`output.py`** — Rich terminal formatting. Call `init_agent_styles(config.agents)` after loading config to set up display names and colors.
- **`merge.py`** — N-way edit merging via diff-match-patch.
- **`cli.py`** — `ConfigGroup` auto-generates top-level Click commands from TOML `[commands]`. Built-in commands: `review`, `install-hook`, `uninstall-hook`, `check-config`. `_create_backend()` factory instantiates the backend from config.

## Key patterns

- **Backend adapter** — `AgentBackend` protocol in `backend.py` decouples orchestration from any specific LLM. `ClaudeCliBackend` in `claude_runner.py` is the only implementation today. The backend is instantiated in `cli.py` via `_create_backend()` and threaded through `consensus.py`. To add a new backend: implement `AgentBackend`, add the name to `KNOWN_BACKENDS` in `config.py`, and add a branch in `_create_backend()`.
- System prompt (agent specialty + mode suffix) and user prompt (file contents + instructions) are separate — system prompt is passed to `backend.run_agent()`, user prompt is the conversation input.
- `build_name_normalizer(config.agents)` returns a closure for mapping agent name variants (display name, lowercase, spaces) back to config keys. Built once per run, threaded to `parse_proposal_reviews()`.
- `init_agent_styles()` sets module-level dicts in `output.py` — avoids threading display names/colors through every output function.
- Agents never review their own proposals or edits they last modified.
- `build_cli_args()` and `KNOWN_TOOLS` live in `claude_runner.py` (Claude CLI concerns). Re-exported from `agents.py` for backward compatibility.

## Development practices

- **DRY** — Don't duplicate logic. Agent display names are derived in one place (`get_display_name()`), name normalization in one place (`build_name_normalizer()`), colors in one place (`init_agent_styles()`). If you need agent metadata, use the existing helpers rather than re-deriving it.
- **Config-driven, not code-driven** — Agent definitions, tool permissions, model choices, and command definitions live in TOML. Don't add agent-specific or command-specific logic to Python source. New agent capabilities or commands should come from config fields, not hardcoded branches.
- **Thread, don't globalize** — Runtime state (normalizer, display names) is built from config and passed explicitly through function parameters. `output.py` is the one exception (module-level dicts via `init_agent_styles()`) to avoid threading through ~15 output functions.
- **Keep modules focused** — Config loading in `config.py`, prompt assembly in `agents.py` + `context.py`, orchestration in `consensus.py`, display in `output.py`. Don't let orchestration logic leak into output or vice versa.
- **Validate at the boundary** — Config validation happens once in `load_config()`. Internal code trusts that config fields are valid after that point.

## Git workflow

- **Do not commit or push** unless explicitly instructed to.
- **Reverting changes** — Before using `git checkout -- <file>` or `git restore`, check `git log --oneline -1 <file>` and `git diff --stat <file>` to confirm the latest commit reflects the state you want to restore to. Restoring discards all uncommitted changes irreversibly — if work has been done since the last commit, it will be lost.

## Testing

Tests are in `tests/`. Merge tests (`test_merge.py`) use arbitrary agent names ("a", "b") and don't depend on any specific agent configuration. Consensus tests (`test_consensus.py`) inject a mock `AgentBackend` via dependency injection rather than patching module imports.
