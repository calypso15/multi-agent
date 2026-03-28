# Multi-Agent Fiction Review

A consensus-based review system for collaborative fiction worldbuilding. Three specialized AI agents review your writing in parallel and must reach majority agreement before changes are committed.

Built for a hard sci-fi universe set on Earth, post-First Contact. Powered by the Claude Code CLI using your existing Claude Max subscription.

## How It Works

When you commit changes to fiction files (`.md`, `.txt`), a pre-commit hook launches three reviewer agents in parallel:

| Agent | Focus |
|---|---|
| **Scientific Rigor** | Physics, biology, chemistry, technology plausibility. Ensures hard sci-fi standards. |
| **Canon Continuity** | Cross-references against all existing files. Catches timeline contradictions, character inconsistencies, naming errors. |
| **Sociopolitical** | Evaluates government responses, cultural shifts, economic impacts, institutional reactions for realism. |

Each agent returns **APPROVE** or **REQUEST_CHANGES**. The commit proceeds only if at least 2 out of 3 agents approve (majority consensus).

## Prerequisites

- Python 3.11+
- [Claude Code CLI](https://claude.ai/code) installed and authenticated (uses your Max subscription)

## Setup

```bash
# Clone and enter the repo
cd multi-agent

# Create a virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Install the git pre-commit hook
python -m multi_agent install-hook
```

Verify your configuration:

```bash
python -m multi_agent check-config
```

## Usage

### Writing and committing fiction

Place your fiction files in the `canon/` directory (or any directory listed in `canon_directories` in the config). When you commit, the review runs automatically:

```bash
# Write your content
vim canon/chapter-01.md

# Stage and commit — the review triggers automatically
git add canon/chapter-01.md
git commit -m "Add chapter 1"
```

If consensus is reached, the commit proceeds. If not, you'll see specific issues with quotes and suggestions:

```
╭─ Multi-Agent Fiction Review ─────────────────────────────────╮
│  Reviewing 1 file(s): canon/chapter-03.md                    │
│  Canon context: 4 files (23 KB)                              │
╰──────────────────────────────────────────────────────────────╯

  Scientific Rigor        APPROVE           8.2s
  Canon Continuity        REQUEST_CHANGES  12.1s
  Sociopolitical          APPROVE           9.7s

╭─ APPROVED (2/3) ────────────────────────────────────────────╮
│  Consensus reached (2/3). Commit may proceed.               │
╰─────────────────────────────────────────────────────────────╯

Canon Continuity Issues:
  [major] canon/chapter-03.md
    "Ambassador Chen arrived in Geneva on March 15th"
    In chapter-01.md, Chen is established as being in Beijing until March 20th.
    Suggestion: Change the date to March 21st or later.

Duration: 12.3s
```

To bypass the review (e.g., for non-fiction files or quick fixes):

```bash
git commit --no-verify -m "Fix typo"
```

### Reviewing existing files

Review files that are already committed or on disk without making a new commit:

```bash
# Review specific files
python -m multi_agent review-files canon/chapter-01.md canon/chapter-02.md

# Review all canon files
python -m multi_agent review-files canon/*.md
```

### Reviewing staged changes manually

Run the same review the pre-commit hook would, without actually committing:

```bash
git add canon/chapter-04.md
python -m multi_agent review
```

### Working with a separate repository

If your fiction lives in a different repo, use the global `--repo` flag (before the subcommand):

```bash
# Review files in another repo
python -m multi_agent --repo ~/git/my-novel review-files canon/*.md

# Review staged changes in another repo
python -m multi_agent --repo ~/git/my-novel review

# Install the pre-commit hook into another repo
python -m multi_agent --repo ~/git/my-novel install-hook
```

You can also place a `multi_agent.toml` in the target repo to customize its settings independently.

## Configuration

Settings are in `multi_agent.toml` at the project root:

```toml
[general]
file_patterns = ["*.md", "*.txt"]   # Which files to review
consensus_threshold = 2              # Approvals needed (out of enabled agents)
timeout_seconds = 120                # Max time per agent
canon_directories = ["canon"]        # Where fiction files live
max_canon_size_kb = 500              # Max canon context sent to agents

[agents.scientific_rigor]
enabled = true
model = "claude-sonnet-4-6"          # Model for this agent

[agents.canon_continuity]
enabled = true
model = "claude-sonnet-4-6"

[agents.sociopolitical]
enabled = true
model = "claude-sonnet-4-6"
```

### Configuration options

| Option | Default | Description |
|---|---|---|
| `file_patterns` | `["*.md", "*.txt"]` | Glob patterns for files to review |
| `consensus_threshold` | `2` | Minimum approvals required |
| `timeout_seconds` | `120` | Per-agent timeout |
| `canon_directories` | `["canon"]` | Directories containing fiction files |
| `max_canon_size_kb` | `500` | Max total size of canon context |

### Agent options

Each agent under `[agents.<name>]` supports:

| Option | Description |
|---|---|
| `enabled` | `true`/`false` — disable an agent to skip it |
| `model` | Claude model to use (e.g., `claude-sonnet-4-6`, `claude-opus-4-6`) |
| `system_prompt_override` | Replace the built-in system prompt entirely |

## CLI Reference

```
multi-agent [--repo PATH] review              Review staged files (same as pre-commit hook)
multi-agent [--repo PATH] review-files FILES  Review existing files on disk
multi-agent [--repo PATH] install-hook        Install the git pre-commit hook
multi-agent [--repo PATH] uninstall-hook      Remove the git pre-commit hook
multi-agent check-config                      Show current configuration
```

`--repo PATH` targets a different git repository (defaults to current directory).
`--config PATH` is accepted by `review`, `review-files`, and `check-config` to use a specific config file.

## Project Structure

```
multi-agent/
├── multi_agent.toml          # Configuration
├── canon/                    # Your fiction files go here
│   ├── chapter-01.md
│   ├── chapter-02.md
│   └── ...
└── src/multi_agent/
    ├── agents.py             # Agent system prompts and definitions
    ├── consensus.py          # Parallel execution and vote tallying
    ├── context.py            # Git diff extraction and canon loading
    ├── cli.py                # CLI commands
    ├── output.py             # Terminal output formatting
    ├── hook.py               # Git hook installer
    └── config.py             # TOML config loading
```
