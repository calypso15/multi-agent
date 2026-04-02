# Multi-Agent Fiction Review

A consensus-based review system for collaborative fiction worldbuilding. Three specialized AI agents propose changes to your writing, review each other's proposals, and iterate until they reach agreement — then present the final edits for your approval.

Built for a hard sci-fi universe set on Earth, post-First Contact. Powered by the Claude Code CLI using your existing Claude Max subscription.

## How It Works

When you run a review (manually or via pre-commit hook), the system launches your configured specialist agents. Agents are defined entirely in `multi_agent.toml` — you can add, remove, or customize any number of them. The example config ships with three:

| Agent | Focus |
|---|---|
| **Scientific Rigor** | Physics, biology, chemistry, technology plausibility. Ensures hard sci-fi standards. Can web-search to verify claims. |
| **Canon Continuity** | Cross-references against all existing canon files. Catches timeline contradictions, character inconsistencies, naming errors. |
| **Sociopolitical** | Evaluates government responses, cultural shifts, economic impacts, institutional reactions for realism. Can web-search to verify real-world references. |

The review runs in phases:

1. **Propose** — Each agent proposes concrete edits (search-and-replace) from their specialty perspective, running in parallel.
2. **Review** — All agents review all proposals. Each either approves or suggests modifications.
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

Then set up your fiction repository:

```bash
# Copy the example config to your fiction repo
cp multi_agent.example.toml ~/git/my-novel/multi_agent.toml

# Edit it to match your project structure
vim ~/git/my-novel/multi_agent.toml

# Optionally install the pre-commit hook
python -m multi_agent --repo ~/git/my-novel install-hook
```

Verify configuration:

```bash
python -m multi_agent --repo ~/git/my-novel check-config
```

## Usage

### Reviewing files

```bash
# Review specific files
python -m multi_agent --repo ~/git/my-novel review canon/chapter-01.md

# Review a directory
python -m multi_agent --repo ~/git/my-novel review canon/

# Dry run — show proposed changes without applying
python -m multi_agent --repo ~/git/my-novel review --dry-run canon/chapter-01.md

# Review staged files (same as pre-commit hook)
python -m multi_agent --repo ~/git/my-novel review
```

### Expanding and contracting

```bash
# Expand a file with richer detail
python -m multi_agent --repo ~/git/my-novel expand canon/chapter-03.md

# Tighten prose and cut filler
python -m multi_agent --repo ~/git/my-novel contract canon/chapter-03.md

# These are shortcuts for:
python -m multi_agent --repo ~/git/my-novel review --task expand canon/chapter-03.md
python -m multi_agent --repo ~/git/my-novel review --task contract canon/chapter-03.md

# Custom tasks defined in multi_agent.toml
python -m multi_agent --repo ~/git/my-novel review --task deepen-characters canon/chapter-03.md
```

### Pre-commit hook

When installed, the hook runs automatically on `git commit`. If agents propose changes, they are applied to the working tree and the commit is aborted so you can review them before re-committing.

```bash
# Install
python -m multi_agent --repo ~/git/my-novel install-hook

# Bypass when needed
git commit --no-verify -m "Fix typo"

# Uninstall
python -m multi_agent --repo ~/git/my-novel uninstall-hook
```

### Example output

```
╭─ Multi-Agent Fiction Review ─────────────────────────────────╮
│  Reviewing 1 file(s): canon/chapter-03.md                    │
│  Canon context: 4 file(s) (23 KB)                            │
╰──────────────────────────────────────────────────────────────╯
  Scientific Rigor: proposing
  Canon Continuity: proposing
  Sociopolitical: proposing
  Scientific Rigor: done — 1 edit(s)
  Canon Continuity: done — 2 edit(s)
  Sociopolitical: done — 0 edit(s)

──────────────────── Propose Phase ────────────────────────────
  Scientific Rigor    1 edit(s) (canon/chapter-03.md)     8.2s
  Canon Continuity    2 edit(s) (canon/chapter-03.md)    12.1s
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

--- a/canon/chapter-03.md
+++ b/canon/chapter-03.md
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

Place a `multi_agent.toml` in your fiction repository. See `multi_agent.example.toml` for a fully commented example.

### General options

| Option | Default | Description |
|---|---|---|
| `file_patterns` | `["*.md", "*.txt"]` | Glob patterns for files to review |
| `consensus_threshold` | `2` | Minimum approvals required for consensus |
| `timeout_seconds` | `600` | Per-agent timeout in seconds |
| `canon_directories` | `["canon"]` | Directories containing established fiction files |
| `max_canon_size_kb` | `500` | Max total size of canon context loaded |
| `max_rounds` | `3` | Maximum propose-review iteration rounds |
| `min_severity` | `"minor"` | Minimum severity for proposed edits: `"critical"`, `"major"`, `"minor"`, or `"suggestion"`. Set to `"major"` to skip nitpicks. |
| `propose_max_turns` | `0` (unlimited) | Max turns per agent in the propose phase |
| `review_max_turns` | `0` (unlimited) | Max turns per agent in review rounds |

### Agent options

Agents are fully defined in the TOML file. You can add, remove, or customize any number of agents. Each `[agents.<name>]` block supports:

| Option | Default | Description |
|---|---|---|
| `system_prompt` | **required** | The agent's specialty/focus prompt |
| `display_name` | titlecased key | Human-readable name for terminal output |
| `enabled` | `true` | Disable an agent to skip it |
| `propose_model` | — | Claude model for the propose phase (e.g., `claude-sonnet-4-6`) |
| `review_model` | same as `propose_model` | Model for review rounds (e.g., `claude-haiku-4-5-20251001` for speed) |
| `allowed_tools` | `[]` | Tools available during the propose phase (e.g., `["WebSearch", "WebFetch"]`) |
| `propose_max_turns` | inherits from general | Override max turns for this agent's propose phase |
| `review_max_turns` | inherits from general | Override max turns for this agent's review phase |

### Custom tasks

Define reusable tasks in `multi_agent.toml` under `[tasks.<name>]`:

```toml
[tasks.deepen-characters]
prompt = "Focus on deepening character voices, adding internal monologue, and making dialogue more distinctive."

[tasks.worldbuild]
prompt = "Enrich world-building details: sensory descriptions, environmental atmosphere, and cultural texture."
```

Use with `--task`:

```bash
python -m multi_agent review --task deepen-characters canon/chapter-03.md
```

### Example config

```toml
[general]
file_patterns = ["*.md", "*.txt"]
consensus_threshold = 2
timeout_seconds = 600
canon_directories = ["canon"]
max_rounds = 5
min_severity = "major"
propose_max_turns = 5
review_max_turns = 2

[agents.scientific_rigor]
enabled = true
display_name = "Scientific Rigor"
propose_model = "claude-sonnet-4-6"
review_model = "claude-haiku-4-5-20251001"
allowed_tools = ["WebSearch", "WebFetch"]
system_prompt = "You are the Scientific Rigor Reviewer..."

[agents.canon_continuity]
enabled = true
display_name = "Canon Continuity"
propose_model = "claude-sonnet-4-6"
review_model = "claude-haiku-4-5-20251001"
system_prompt = "You are the Canon Continuity Reviewer..."

[agents.sociopolitical]
enabled = true
propose_model = "claude-sonnet-4-6"
review_model = "claude-haiku-4-5-20251001"
allowed_tools = ["WebSearch", "WebFetch"]
system_prompt = "You are the Sociopolitical Plausibility Reviewer..."

[tasks.deepen-characters]
prompt = "Focus on deepening character voices and making dialogue more distinctive."
```

## CLI Reference

```
multi-agent [--repo PATH] review [FILES] [--task NAME] [--dry-run] [--max-rounds N]
multi-agent [--repo PATH] expand FILES [--dry-run] [--max-rounds N]
multi-agent [--repo PATH] contract FILES [--dry-run] [--max-rounds N]
multi-agent [--repo PATH] install-hook
multi-agent [--repo PATH] uninstall-hook
multi-agent [--repo PATH] check-config
```

- `review` — review files on disk, or staged files when no FILES given
- `expand` — shortcut for `review --task expand`
- `contract` — shortcut for `review --task contract`
- `--repo PATH` — target a different git repository (defaults to current directory)
- `--config PATH` — use a specific config file
- `--task NAME` — task mode: `expand`, `contract`, or a custom task from config
- `--dry-run` — show proposed changes without applying
- `--max-rounds N` — override the configured max iteration rounds
- `--prompt TEXT` — append additional instructions for the agents

## Project Structure

```
multi-agent/
├── multi_agent.example.toml  # Example config (copy to your fiction repo)
└── src/multi_agent/
    ├── models.py             # Dataclasses, typed events, parsing utilities
    ├── claude_runner.py      # Claude CLI subprocess management
    ├── consensus.py          # Propose-review-iterate orchestration loop
    ├── agents.py             # Agent system prompts, JSON schemas, prompt modes
    ├── context.py            # Git integration, prompt builders
    ├── merge.py              # N-way edit merging via diff-match-patch
    ├── cli.py                # CLI commands
    ├── output.py             # Rich terminal output formatting
    ├── config.py             # TOML config loading
    └── hook.py               # Git pre-commit hook installer
```
