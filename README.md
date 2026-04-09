# Multi-Agent Review

A consensus-based multi-agent system where specialized AI agents collaborate via propose-review-iterate cycles until they reach agreement. Use it to review and edit content, or ask questions and get a single consensus answer. Powered by the Claude Code CLI using your existing Claude Max subscription.

## How It Works

When you run a review (manually or via pre-commit hook), the system launches your configured specialist agents. Agents are defined entirely in `multi_agent.toml` — you can add, remove, or customize any number of them with different specialties (e.g., domain accuracy, style consistency, technical correctness).

The review runs in phases:

1. **Propose** — Each agent proposes concrete edits (search-and-replace) from their specialty perspective, each classified by severity (`critical`, `major`, `minor`, `suggestion`). Edits below `min_severity` are dropped.
2. **Review** — All agents review all proposals. Each either approves or suggests modifications. Edits below `min_blocking_severity` don't prevent consensus but modifications are still applied.
3. **Iterate** — If consensus isn't reached, modifications are merged and agents review again (up to `max_rounds`).
4. **Arbitrate** — If the loop stalls (no improvement for 2 rounds), contested edits are sent to an impartial arbitrator that picks the better version. Arbitrated regions are locked from further modification.
5. **Present** — The final set of changes is shown as a unified diff. If consensus wasn't reached, dissenting agents provide brief opinions. You choose whether to apply the changes.

## Prerequisites

- Python 3.11+
- [Claude Code CLI](https://claude.ai/code) installed and authenticated (uses your Max subscription)

## Setup

```bash
# Clone the tool
git clone https://github.com/calypso15/multi-agent.git
cd multi-agent

# Create a virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then set up your target repository:

```bash
# Copy the example config to your project
cp multi_agent.example.toml ~/git/my-project/multi_agent.toml

# Edit it to match your project structure
vim ~/git/my-project/multi_agent.toml

# Optionally install the pre-commit hook
python -m multi_agent --repo ~/git/my-project install-hook
```

Verify configuration:

```bash
python -m multi_agent --repo ~/git/my-project check-config
```

## Usage

### Reviewing files

```bash
# Review specific files
python -m multi_agent --repo ~/git/my-project review src/api.py

# Review a directory
python -m multi_agent --repo ~/git/my-project review src/

# Dry run — show proposed changes without applying
python -m multi_agent --repo ~/git/my-project review --dry-run src/api.py

# Review staged files (same as pre-commit hook)
python -m multi_agent --repo ~/git/my-project review
```

### Asking questions

The `ask` command lets you pose a question and get a consensus answer from all agents. Each agent answers from their specialty perspective, then they review and iterate until they converge on a single answer.

```bash
# Ask a question (quoting optional)
python -m multi_agent --repo ~/git/my-project ask "What are the main themes of this story?"
python -m multi_agent --repo ~/git/my-project ask How should we handle the timeline inconsistency

# Override max rounds
python -m multi_agent --repo ~/git/my-project ask --max-rounds 5 "What if we changed the setting?"
```

The answer is displayed in the terminal and saved in the structured run log (see below).

### Run logs

Every run writes a structured JSON log to `.multi_agent_runs/` in the repo root. Logs capture proposals, reviews, arbitration results, dissents, token usage, and config — everything needed to understand why agents agreed or disagreed. Add `.multi_agent_runs/` to `.gitignore`.

### Commands

Commands are defined in `[commands]` blocks in `multi_agent.toml`. They can be
invoked as top-level CLI commands or via `review --task`:

```bash
# Top-level command (auto-generated from TOML)
python -m multi_agent --repo ~/git/my-project expand docs/overview.md

# Equivalent via --task
python -m multi_agent --repo ~/git/my-project review --task expand docs/overview.md

# Any command defined in [commands] works the same way
python -m multi_agent --repo ~/git/my-project deepen-characters docs/overview.md
```

### Pre-commit hook

When installed, the hook runs automatically on `git commit`. If agents propose changes, they are applied to the working tree and the commit is aborted so you can review them before re-committing.

```bash
# Install
python -m multi_agent --repo ~/git/my-project install-hook

# Bypass when needed
git commit --no-verify -m "Fix typo"

# Uninstall
python -m multi_agent --repo ~/git/my-project uninstall-hook
```

### Example output

```
╭─ Multi-Agent Review ─────────────────────────────────────────╮
│  Reviewing 1 file(s): docs/overview.md                       │
│  Reference context: 4 file(s) (23 KB)                        │
╰──────────────────────────────────────────────────────────────╯
  Scientific Rigor: proposing
  Canon Continuity: proposing
  Sociopolitical: proposing
  Scientific Rigor: done — 1 edit(s)
  Canon Continuity: done — 2 edit(s)
  Sociopolitical: done — 0 edit(s)

──────────────────── Propose Phase ────────────────────────────
  Scientific Rigor    1 edit(s) [1 critical] (docs/overview.md)     8.2s
  Canon Continuity    2 edit(s) [1 major, 1 minor] (docs/overview.md)    12.1s
  Sociopolitical      no edits                            9.7s

  3 total edit(s) proposed

── Review Round 1  3/3 approved (need 2) ──────────────────────
  Scientific Rigor    APPROVED ALL                        2.1s
  Canon Continuity    APPROVED ALL                        1.8s
  Sociopolitical      APPROVED ALL                        2.3s

╭─ CONSENSUS ──────────────────────────────────────────────────╮
│  Consensus reached (3/3). All agents approve the proposed    │
│  changes.                                                    │
╰──────────────────────────────────────────────────────────────╯

──────────────────── Proposed Changes ─────────────────────────

--- a/docs/overview.md
+++ b/docs/overview.md
@@ -12,7 +12,7 @@
-The ship accelerated to 3c using conventional thrusters.
+The ship accelerated to 0.3c using conventional thrusters.

╭─ Usage ──────────────────────────────────────────────────────╮
│  Input tokens              45,230                             │
│    Cache read              38,000                             │
│  Output tokens              1,450                             │
│  Total cost               $0.0842                             │
│  Duration                  18.3s                              │
╰──────────────────────────────────────────────────────────────╯

Apply these changes? [y/N]
```

## Configuration

Place a `multi_agent.toml` in your repository. See `multi_agent.example.toml` for a fully commented example.

### General options

| Option | Default | Description |
|---|---|---|
| `file_patterns` | `["*.md", "*.txt"]` | Glob patterns for files to review |
| `consensus_threshold` | `2` | Minimum weighted approvals required for consensus |
| `timeout_seconds` | `600` | Per-agent timeout in seconds |
| `reference_directories` | `["reference"]` | Directories containing established reference files |
| `max_reference_size_kb` | `500` | Max total size of reference content loaded |
| `max_rounds` | `3` | Maximum propose-review iteration rounds |
| `min_severity` | `"minor"` | Minimum severity for proposed edits: `"critical"`, `"major"`, `"minor"`, or `"suggestion"`. Edits below this are dropped. |
| `min_blocking_severity` | `"major"` | Minimum severity for edits to block consensus. Edits below this are included but won't prevent consensus. Must be at least as severe as `min_severity`. |
| `propose_max_turns` | `0` (unlimited) | Max turns per agent in the propose phase |
| `review_max_turns` | `0` (unlimited) | Max turns per agent in review rounds |

### Agent options

Agents are fully defined in the TOML file. You can add, remove, or customize any number of agents. Each `[agents.<name>]` block supports:

| Option | Default | Description |
|---|---|---|
| `system_prompt` | **required** | The agent's specialty/focus prompt |
| `display_name` | titlecased key | Human-readable name for terminal output |
| `enabled` | `true` | Disable an agent to skip it |
| `weight` | `1` | How much this agent's approval counts toward `consensus_threshold` |
| `propose_model` | — | Claude model for the propose phase (e.g., `sonnet`) |
| `review_model` | same as `propose_model` | Model for review rounds (e.g., `haiku` for speed) |
| `allowed_tools` | `[]` | Tools available during the propose phase (e.g., `["WebSearch", "WebFetch"]`) |
| `propose_max_turns` | inherits from general | Override max turns for this agent's propose phase |
| `review_max_turns` | inherits from general | Override max turns for this agent's review phase |

### Commands

Define commands in `multi_agent.toml` under `[commands.<name>]`. Each command needs a `prompt` describing the task for agents. Commands become available both as top-level CLI commands and via `review --task`.

| Option | Default | Description |
|---|---|---|
| `prompt` | **required** | Task instructions for agents |
| `description` | — | Shown in `--help` output |
| `agents` | all enabled | Agents to use — simple list or table with per-agent overrides |
| `consensus_threshold` | inherits from general | Override consensus threshold for this command |
| `propose_model` | — | Override agent models for the propose phase |
| `review_model` | — | Override agent models for review/dissent phases |

Settings cascade four levels: **command-agent > command > agent > general**. Per-agent overrides within a command take highest priority.

```toml
[commands.review]
description = "Review files and propose changes via consensus"
prompt = "Review the submitted content from your specialty perspective and propose CONCRETE edits to improve it."

[commands.expand]
description = "Expand files with richer detail via consensus"
propose_model = "sonnet"
prompt = "Your goal is to enrich the submitted content. Add vivid descriptions, flesh out thin scenes..."

# Simple agent list
[commands.deepen-characters]
agents = ["canon_continuity", "sociopolitical"]
prompt = "Focus on deepening character voices, adding internal monologue, and making dialogue more distinctive."

# Per-agent overrides within a command
[commands.lead-review]
prompt = "Review with the lead agent having final say."
consensus_threshold = 3

[commands.lead-review.agents.scientific_rigor]
weight = 2     # lead agent's approval counts as 2

[commands.lead-review.agents.canon_continuity]
weight = 1

[commands.lead-review.agents.sociopolitical]
weight = 1
propose_model = "opus"   # use a different model just for this command
```

Commands can inherit from other commands to avoid duplicating prompts:

```toml
[commands.expand-aggressive]
inherits = "expand"
max_rounds = 8
```

Only the fields you specify are overridden; everything else comes from the base. Single-level inheritance only (a base command cannot itself use `inherits`).

The `review` and `ask` commands have built-in defaults and are always available even without a TOML file. All other commands must be defined in config. You can override the default `ask` prompt in TOML under `[commands.ask]`.

### Example config

```toml
[general]
file_patterns = ["*.md", "*.txt"]
consensus_threshold = 2
timeout_seconds = 600
reference_directories = ["reference"]
max_rounds = 5
min_severity = "minor"
min_blocking_severity = "major"
propose_max_turns = 5
review_max_turns = 2

[agents.accuracy]
display_name = "Accuracy"
weight = 2                    # lead reviewer — approval counts double
propose_model = "sonnet"
review_model = "haiku"
allowed_tools = ["WebSearch", "WebFetch"]
system_prompt = "You are the Accuracy Reviewer. Verify all claims, data, and references..."

[agents.consistency]
display_name = "Consistency"
propose_model = "sonnet"
review_model = "haiku"
system_prompt = "You are the Consistency Reviewer. Ensure alignment with reference files..."

[commands.simplify]
prompt = "Simplify complex passages while preserving technical accuracy."
```

## CLI Reference

```
multi-agent [--repo PATH] review [FILES] [--task NAME] [--dry-run] [--max-rounds N] [--prompt TEXT] [-d] [-v]
multi-agent [--repo PATH] ask QUESTION [--max-rounds N] [-d] [-v]
multi-agent [--repo PATH] <command> FILES [--dry-run] [--max-rounds N] [--prompt TEXT] [-d] [-v]
multi-agent [--repo PATH] install-hook
multi-agent [--repo PATH] uninstall-hook
multi-agent [--repo PATH] check-config
```

- `review` — review files on disk, or staged files when no FILES given
- `ask` — pose a question and get a consensus answer from all agents
- `<command>` — any command defined in `[commands]` (e.g. `expand`, `contract`); auto-generated from TOML
- `--repo PATH` — target a different git repository (defaults to current directory)
- `--config PATH` — use a specific config file
- `--task NAME` — run a command from `[commands]` config (e.g. `expand`, `contract`)
- `--dry-run` — show proposed changes without applying
- `--max-rounds N` — override the configured max iteration rounds
- `--prompt TEXT` — append additional instructions for the agents
- `--detail`, `-d` — show full edit content for each round (what each agent proposed, what reviewers modified)
- `--verbose`, `-v` — show operational telemetry (token usage, tool calls, turns, resolved config)

## Project Structure

```
multi-agent/
├── multi_agent.example.toml  # Example config (copy to your project repo)
└── src/multi_agent/
    ├── models.py             # Dataclasses, typed events, parsing utilities
    ├── claude_runner.py      # Claude CLI subprocess management
    ├── consensus.py          # Propose-review-iterate orchestration loop
    ├── arbitration.py        # Stall detection, arbitration, dissent collection
    ├── agents.py             # Agent system prompts, JSON schemas, prompt modes
    ├── context.py            # Git integration, file resolution, prompt builders
    ├── merge.py              # N-way edit merging via diff-match-patch
    ├── cli.py                # CLI commands
    ├── output.py             # Rich terminal output formatting
    ├── config.py             # TOML config loading
    └── hook.py               # Git pre-commit hook installer
```

## License

[MIT](LICENSE)
