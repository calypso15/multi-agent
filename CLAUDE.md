# CLAUDE.md

## Project overview

Multi-agent consensus review system for collaborative fiction. Agents are defined in `multi_agent.toml` (not in Python source). The system spawns Claude CLI subprocesses to run each agent.

## Commands

```bash
source .venv/bin/activate
python -m pytest tests/ -v          # run tests
python -m multi_agent check-config   # validate config (needs --repo or a multi_agent.toml above cwd)
```

## Architecture

- **`config.py`** — TOML loading, `AgentConfig`/`GeneralConfig`/`CommandConfig` dataclasses, validation. Agents are defined via `[agents.<name>]` blocks, commands via `[commands.<name>]` blocks. `DEFAULT_COMMANDS` provides fallback review/expand/contract. `get_display_name()` derives display names.
- **`agents.py`** — Internal mode suffixes (review/dissent), `build_command_mode_suffix()` for TOML-defined commands, JSON output schemas, `build_agent_system_prompt()`, `build_cli_args()`, `build_name_normalizer()`. No agent-specific content — all generic.
- **`consensus.py`** — Orchestration: propose phase, review phase, iteration loop, arbitration, dissent collection. Agents run sequentially via `claude_runner.run_agent()`.
- **`context.py`** — Builds user prompts (file contents, canon listing, review round proposals). Separate from system prompts.
- **`models.py`** — Dataclasses (`FileEdit`, `AgentProposal`, `AgentReviewResponse`, etc.), JSON parsing, path sanitization.
- **`output.py`** — Rich terminal formatting. Call `init_agent_styles(config.agents)` after loading config to set up display names and colors.
- **`merge.py`** — N-way edit merging via diff-match-patch.
- **`cli.py`** — `ConfigGroup` auto-generates top-level Click commands from TOML `[commands]`. Built-in commands: `review`, `install-hook`, `uninstall-hook`, `check-config`.

## Key patterns

- System prompt (agent specialty + mode suffix) and user prompt (file contents + instructions) are separate — system prompt goes to `--system-prompt` CLI arg, user prompt is the conversation input.
- `build_name_normalizer(config.agents)` returns a closure for mapping agent name variants (display name, lowercase, spaces) back to config keys. Built once per run, threaded to `parse_proposal_reviews()`.
- `init_agent_styles()` sets module-level dicts in `output.py` — avoids threading display names/colors through every output function.
- Agents never review their own proposals or edits they last modified.

## Development practices

- **DRY** — Don't duplicate logic. Agent display names are derived in one place (`get_display_name()`), name normalization in one place (`build_name_normalizer()`), colors in one place (`init_agent_styles()`). If you need agent metadata, use the existing helpers rather than re-deriving it.
- **Config-driven, not code-driven** — Agent definitions, tool permissions, model choices, and command definitions live in TOML. Don't add agent-specific or command-specific logic to Python source. New agent capabilities or commands should come from config fields, not hardcoded branches.
- **Thread, don't globalize** — Runtime state (normalizer, display names) is built from config and passed explicitly through function parameters. `output.py` is the one exception (module-level dicts via `init_agent_styles()`) to avoid threading through ~15 output functions.
- **Keep modules focused** — Config loading in `config.py`, prompt assembly in `agents.py` + `context.py`, orchestration in `consensus.py`, display in `output.py`. Don't let orchestration logic leak into output or vice versa.
- **Validate at the boundary** — Config validation happens once in `load_config()`. Internal code trusts that config fields are valid after that point.

## Testing

Tests are in `tests/test_merge.py`. They use arbitrary agent names ("a", "b") and don't depend on any specific agent configuration.
