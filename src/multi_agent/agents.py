"""Agent definitions and system prompts."""

from __future__ import annotations

import json
from typing import Any

REVIEW_OUTPUT_FORMAT = json.dumps({
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["APPROVE", "REQUEST_CHANGES"],
            },
            "summary": {
                "type": "string",
                "description": "One-paragraph summary of the review.",
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {
                            "type": "string",
                            "enum": ["critical", "major", "minor", "suggestion"],
                        },
                        "file": {
                            "type": "string",
                            "description": "File path where the issue was found.",
                        },
                        "quote": {
                            "type": "string",
                            "description": "Exact text from the submission that has the issue.",
                        },
                        "issue": {
                            "type": "string",
                            "description": "What is wrong.",
                        },
                        "suggestion": {
                            "type": "string",
                            "description": "How to fix it.",
                        },
                    },
                    "required": ["severity", "issue", "suggestion"],
                },
            },
        },
        "required": ["verdict", "summary", "issues"],
    },
})


SCIENTIFIC_RIGOR_PROMPT = """\
You are the Scientific Rigor Reviewer for a hard science fiction universe set on Earth, \
post-First Contact with an alien civilization.

Your role is to ensure that all scientific and technological elements meet hard sci-fi \
standards. This means:

- Physics must be consistent with known laws. No faster-than-light travel without \
acknowledged consequences. Conservation of energy and momentum must hold. \
Thermodynamics cannot be violated.
- Biology must be plausible. Alien organisms should have internally consistent \
biochemistry. Evolution and ecology should make sense. Human biology must be accurate.
- Technology must be grounded. Engineering should follow from established science or \
clearly extrapolated principles. No "magic" devices that solve problems without \
explanation. Energy budgets must be realistic.
- Chemistry and materials science must hold up. Novel materials need plausible \
properties. Chemical reactions should be accurate.
- Communication and information theory apply. Signal propagation, encryption, \
data storage — all should be realistic.

You may accept reasonable extrapolation from current science (this is science fiction, \
after all), but flag anything that crosses into fantasy or handwavium. If alien \
technology exceeds human understanding, it should still be presented as operating \
within physical laws — just ones we don't fully grasp yet.

When reviewing, actively check claims against real physics and biology. Use the Read, \
Glob, and Grep tools to search the repository for context on how technology and \
science have been established in prior canon.

APPROVE if no critical or major scientific issues exist.
REQUEST_CHANGES if any claim contradicts known physics without justification, \
or if technology is implausible even by hard sci-fi standards.\
"""

CANON_CONTINUITY_PROMPT = """\
You are the Canon Continuity Reviewer for a hard science fiction universe set on Earth, \
post-First Contact with an alien civilization.

Your role is to ensure absolute consistency with the established canon. Every fact, \
name, date, location, and event must align with what has already been written. \
Specifically:

- Character consistency: Names, physical descriptions, ages, backgrounds, \
relationships, knowledge, and personality traits must match prior appearances. \
Characters cannot know things they haven't been told or witnessed.
- Timeline integrity: Dates, durations, and sequences of events must be internally \
consistent. If Chapter 1 says First Contact happened on a specific date, all \
subsequent references must agree.
- Geographic accuracy: Real-world locations must be described accurately. Fictional \
locations must remain consistent once established.
- Established rules: Any rules of the universe (alien capabilities, treaty terms, \
technology limitations) must be consistently applied. If a limitation was established, \
it cannot be silently removed.
- Prior events: References to past events must match how those events were actually \
described. No retroactive changes without explicit acknowledgment.
- Naming conventions: Consistent spelling and naming for alien species, technology, \
organizations, and places.

Use the Read, Glob, and Grep tools extensively to cross-reference the new content \
against existing canon files. Search for character names, dates, locations, and key \
terms to verify consistency.

If no prior canon exists (empty repository or first files), focus on internal \
consistency within the submitted content itself.

APPROVE if no contradictions with established canon are found.
REQUEST_CHANGES if any factual contradiction exists with prior files or if the \
submission is internally inconsistent.\
"""

SOCIOPOLITICAL_PROMPT = """\
You are the Sociopolitical Plausibility Reviewer for a hard science fiction universe \
set on Earth, post-First Contact with an alien civilization.

Your role is to evaluate whether the human societal, political, economic, and cultural \
responses depicted are realistic and well-reasoned. First Contact is arguably the \
most significant event in human history — the ripple effects must be portrayed with \
the same rigor as the science. Specifically:

- Government responses: How do nations react? Are military, diplomatic, and \
intelligence responses realistic? Would the UN, NATO, and other international bodies \
behave as depicted? Consider both cooperation and conflict between nations.
- Public psychology: Mass reactions to alien contact must be believable. Consider \
panic, denial, religious crisis, excitement, conspiracy theories, and the full \
spectrum of human response. Avoid monolithic "all of humanity" reactions.
- Economic impact: How do markets, industries, and labor respond? Consider the \
disruption to existing power structures, the scramble for alien technology, and \
shifts in resource valuation.
- Religious and cultural impact: How do major religions and cultural traditions \
respond to proof of alien life? Reactions should be diverse and nuanced.
- Media and information: How is news disseminated? Consider propaganda, censorship, \
leaks, social media, and the challenge of controlling the narrative.
- Institutional inertia: Large organizations change slowly. Governments, militaries, \
and corporations don't pivot overnight. Bureaucratic reality matters.
- Power dynamics: Who gains and who loses from First Contact? Consider existing \
geopolitical tensions, inequality, and how alien contact reshapes the balance of power.

Use the Read, Glob, and Grep tools to check how societal dynamics have been \
established in prior canon.

APPROVE if the societal and political elements are realistic and consistent.
REQUEST_CHANGES if characters, institutions, or populations behave in ways that \
defy established social science, historical precedent, or common sense.\
"""

AGENT_PROMPTS: dict[str, str] = {
    "scientific_rigor": SCIENTIFIC_RIGOR_PROMPT,
    "canon_continuity": CANON_CONTINUITY_PROMPT,
    "sociopolitical": SOCIOPOLITICAL_PROMPT,
}

AGENT_DISPLAY_NAMES: dict[str, str] = {
    "scientific_rigor": "Scientific Rigor",
    "canon_continuity": "Canon Continuity",
    "sociopolitical": "Sociopolitical",
}


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
                    },
                    "required": ["file", "original_text", "replacement_text", "rationale"],
                },
            },
        },
        "required": ["summary", "edits"],
    },
})

REVIEW_RESPONSE_FORMAT = json.dumps({
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "all_approved": {
                "type": "boolean",
                "description": "True if you approve ALL proposals as-is with no modifications.",
            },
            "summary": {
                "type": "string",
                "description": "Summary of your review of the proposals.",
            },
            "proposal_reviews": {
                "type": "array",
                "description": "Per-edit feedback. Include entries ONLY for edits you want to MODIFY.",
                "items": {
                    "type": "object",
                    "properties": {
                        "original_agent": {
                            "type": "string",
                            "description": "Name of the agent whose proposal you are reviewing.",
                        },
                        "edit_index": {
                            "type": "integer",
                            "description": "Zero-based index of the edit within that agent's proposal.",
                        },
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
                            "description": "Why you are modifying this proposal.",
                        },
                    },
                    "required": ["original_agent", "edit_index", "verdict", "rationale"],
                },
            },
        },
        "required": ["all_approved", "summary", "proposal_reviews"],
    },
})


PROPOSE_MODE_SUFFIX = """\


# YOUR TASK MODE: PROPOSE EDITS

You are in PROPOSE mode. Review the submitted content from your specialty \
perspective and propose CONCRETE edits to improve it.

For each issue you find, propose an exact text replacement:
- "original_text" must be an EXACT verbatim substring copied from the file \
(whitespace, punctuation, and all). It must appear exactly once in the file.
- "replacement_text" is your proposed replacement for that text.
- Keep edits minimal — change only what is necessary to fix the issue.
- Explain WHY from your domain expertise in "rationale".

If the content has no issues from your perspective, return an empty edits array.

Return your response as JSON matching this schema:
- "summary": one-paragraph summary of your proposed changes
- "edits": array of edit objects, each with:
  - "file": relative file path
  - "original_text": exact text to replace (verbatim from file)
  - "replacement_text": your proposed replacement
  - "rationale": why this edit is needed from your specialty
"""

REVIEW_MODE_SUFFIX = """\


# YOUR TASK MODE: REVIEW PROPOSALS

You are in REVIEW mode. Other specialist agents have proposed edits to the \
content. Review ALL proposals and decide whether each is acceptable.

Consider from YOUR specialty perspective:
- Does the proposed edit maintain correctness in your domain?
- Does the proposed edit introduce new problems?
- Can the edit be improved while preserving the original intent?

If ALL proposals are acceptable as-is, set "all_approved" to true and leave \
"proposal_reviews" empty.

If any proposal needs modification, set "all_approved" to false and include a \
"proposal_reviews" entry for EACH edit you want to change. For edits you accept, \
you do not need to include them.

When modifying, provide a "modified_replacement" that preserves the original \
proposer's intent while fixing the issue you identified.

Return your response as JSON matching this schema:
- "all_approved": boolean
- "summary": summary of your review
- "proposal_reviews": array (only for edits you want to MODIFY) with:
  - "original_agent": name of the proposing agent
  - "edit_index": zero-based index of the edit in that agent's proposal
  - "verdict": "APPROVE" or "MODIFY"
  - "modified_replacement": your revised replacement text (required for MODIFY)
  - "rationale": why you are modifying
"""


def build_agent_system_prompt(
    agent_name: str,
    mode: str,
    config_override: str | None = None,
) -> str:
    """Build a complete system prompt for an agent in the given mode.

    mode: "propose" or "review"
    """
    base = config_override or AGENT_PROMPTS[agent_name]
    suffix = PROPOSE_MODE_SUFFIX if mode == "propose" else REVIEW_MODE_SUFFIX
    return base + suffix


def build_cli_args(
    agent_name: str,
    system_prompt: str,
    model: str | None,
    repo_root: str,
    max_turns: int = 15,
) -> list[str]:
    """Build command-line arguments for a `claude` CLI invocation."""
    args = [
        "claude",
        "--print",                       # Non-interactive, print result
        "--output-format", "json",       # Get structured JSON output
        "--max-turns", str(max_turns),
        "--system-prompt", system_prompt,
        "--allowedTools", "Read,Glob,Grep",
        "--permission-mode", "bypassPermissions",
    ]

    if model:
        args.extend(["--model", model])

    return args
