"""Agent system prompt assembly and CLI argument building."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from multi_agent.config import AgentConfig


def build_name_normalizer(
    agents: dict[str, AgentConfig],
) -> Callable[[str], str]:
    """Build a name-normalizer from the current agent config.

    Returns a closure that maps any variant of an agent name (display name,
    lowercased, underscores-as-spaces) back to its canonical config key.
    """
    from multi_agent.config import get_display_name

    lookup: dict[str, str] = {}
    for key, cfg in agents.items():
        display = get_display_name(key, cfg)
        lookup[key] = key
        lookup[display] = key
        lookup[display.lower()] = key
        lookup[key.replace("_", " ")] = key

    def normalize(name: str) -> str:
        # Exact match first
        result = lookup.get(name) or lookup.get(name.lower())
        if result:
            return result
        # Strip parenthetical suffixes LLMs sometimes add, e.g.
        # "Sociopolitical (politics)" → "Sociopolitical"
        stripped = name.split("(")[0].strip() if "(" in name else name
        if stripped != name:
            result = lookup.get(stripped) or lookup.get(stripped.lower())
            if result:
                return result
        return name

    return normalize


# --- Proposal / Review schemas and prompts for the iterate loop ---

PROPOSAL_OUTPUT_FORMAT = json.dumps({
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "One-paragraph summary of proposed changes and rationale.",
            },
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {
                            "type": "string",
                            "description": "Relative file path to edit.",
                        },
                        "original_text": {
                            "type": "string",
                            "description": (
                                "Exact verbatim text from the file to be replaced. "
                                "Must be a unique substring that appears exactly once."
                            ),
                        },
                        "replacement_text": {
                            "type": "string",
                            "description": "The replacement text.",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Why this edit is necessary from your specialty perspective.",
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["critical", "major", "minor", "suggestion"],
                            "description": "Severity of the issue this edit addresses.",
                        },
                    },
                    "required": ["file", "original_text", "replacement_text", "rationale", "severity"],
                },
            },
        },
        "required": ["summary", "edits"],
    },
})

EDIT_REVIEW_RESPONSE_FORMAT = json.dumps({
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["APPROVE", "MODIFY"],
            },
            "modified_replacement": {
                "type": "string",
                "description": "Your revised replacement_text. Required if verdict is MODIFY.",
            },
            "rationale": {
                "type": "string",
                "description": "Why you approve or are modifying this edit.",
            },
        },
        "required": ["verdict", "rationale"],
    },
})


_PROPOSE_JSON_INSTRUCTIONS = """\

For each change, propose an exact text replacement:
- "original_text" must be an EXACT verbatim substring copied from the file \
(whitespace, punctuation, and all). It must appear exactly once in the file.
- "replacement_text" is your proposed replacement for that text.
- Explain WHY from your domain expertise in "rationale".
- Classify the severity of the issue the edit addresses:
  - "critical" — correctness, safety, or logical errors
  - "major" — significant quality or consistency issues
  - "minor" — small improvements, clarity
  - "suggestion" — optional, stylistic

IMPORTANT — keep edits small and focused:
- Prefer MANY small edits over FEW large ones. Target one paragraph or \
logical unit per edit.
- Do NOT combine multiple unrelated changes into a single large edit.
- Keep each "replacement_text" under ~500 words. If a larger change is \
needed, split it into multiple sequential edits.
- This constraint exists because all agents generate output in parallel — \
one agent producing a very long response blocks the others.

If you have no changes to propose, return an empty edits array.

Return your response as JSON matching this schema:
- "summary": one-paragraph summary of your proposed changes
- "edits": array of edit objects, each with:
  - "file": relative file path
  - "original_text": exact text to replace (verbatim from file)
  - "replacement_text": your proposed replacement
  - "rationale": why this edit is needed from your specialty
  - "severity": "critical", "major", "minor", or "suggestion"
"""

EDIT_REVIEW_MODE_SUFFIX = """\


# YOUR TASK MODE: REVIEW A PROPOSED EDIT

You are in REVIEW mode. Another specialist agent has proposed an edit to the \
content. Decide whether the edit is acceptable or needs modification.

IMPORTANT: The file under review is a draft being improved — it is NOT \
authoritative. Only files in the reference directories are authoritative. \
Do not reject the edit solely because it changes terminology, facts, or \
descriptions established in the file under review. Changing those is often \
the point of the edit. Judge the edit against the REFERENCE FILES, not against \
the current draft text.

Consider from YOUR specialty perspective:
- Does the proposed edit maintain correctness in your domain?
- Does the proposed edit introduce new problems?
- Is the edit consistent with established reference files?
- Can the edit be improved while preserving the original intent?

If the edit is acceptable as-is, set "verdict" to "APPROVE".

If the edit needs modification, set "verdict" to "MODIFY" and provide a \
"modified_replacement" that preserves the original proposer's intent while \
fixing the issue you identified.

Return your response as JSON:
- "verdict": "APPROVE" or "MODIFY"
- "modified_replacement": your revised replacement text (required for MODIFY)
- "rationale": why you approve or are modifying
"""


DISSENT_MODE_SUFFIX = """\


# YOUR TASK MODE: DISSENTING OPINION

Consensus was NOT reached. The proposed changes are being presented to the \
user despite your objections. You have one chance to provide a brief \
dissenting opinion.

Be concise (2-4 sentences). State your most important concern and why \
the user should be cautious about accepting the proposed changes. Focus on \
the single most critical issue from your specialty perspective.

Return your response as JSON with a single "opinion" field containing your dissent.
"""

DISSENT_OUTPUT_FORMAT = json.dumps({
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "opinion": {
                "type": "string",
                "description": "Brief dissenting opinion (2-4 sentences).",
            },
        },
        "required": ["opinion"],
    },
})

ARBITRATOR_PROMPT = """\
You are an impartial arbitrator for a multi-agent review system. Two specialist \
agents have proposed conflicting versions of the same text and cannot agree.

Your job is to pick the better version OR merge them into a single version \
that resolves both agents' concerns. You have no specialty bias — judge \
purely on quality, consistency, and which version better serves the content.

Be decisive. Return your response as JSON with:
- "replacement_text": the final text (pick one version or merge)
- "rationale": one sentence explaining your decision
"""

ARBITRATOR_OUTPUT_FORMAT = json.dumps({
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "replacement_text": {
                "type": "string",
                "description": "The final replacement text.",
            },
            "rationale": {
                "type": "string",
                "description": "One-sentence explanation of the decision.",
            },
        },
        "required": ["replacement_text", "rationale"],
    },
})

_INTERNAL_MODE_SUFFIXES: dict[str, str] = {
    "review": EDIT_REVIEW_MODE_SUFFIX,
    "dissent": DISSENT_MODE_SUFFIX,
}


def build_command_mode_suffix(command_name: str, prompt: str) -> str:
    """Build a propose-mode suffix from a TOML command's prompt."""
    label = command_name.upper().replace("-", " ").replace("_", " ")
    return f"""\


# YOUR TASK MODE: {label}

{prompt}
""" + _PROPOSE_JSON_INSTRUCTIONS


def build_agent_system_prompt(
    agent_name: str,
    mode: str,
    system_prompt: str,
    *,
    command_name: str | None = None,
    command_prompt: str | None = None,
    max_turns: int = 0,
) -> str:
    """Build a complete system prompt for an agent in the given mode.

    system_prompt: the agent's base prompt from config.
    mode: "command" (user-facing task) or "review"/"dissent" (internal phase).
    For "command" mode, command_name and command_prompt must be provided.
    max_turns: if > 0, appends a turn budget notice so the agent can plan.
    """
    if mode == "command":
        suffix = build_command_mode_suffix(
            command_name or "task", command_prompt or "",
        )
    else:
        suffix = _INTERNAL_MODE_SUFFIXES.get(mode, "")
    prompt = system_prompt + suffix
    if max_turns > 0:
        prompt += (
            f"\n\nIMPORTANT: You have a budget of {max_turns} turn(s). "
            "Plan your tool usage carefully and return your JSON response "
            "before running out of turns."
        )
    return prompt


# Re-exports — these now live in claude_runner but are kept here for
# backward compatibility with external importers and existing tests.
from multi_agent.claude_runner import KNOWN_TOOLS, build_cli_args  # noqa: F401
