"""Agent definitions and system prompts."""

from __future__ import annotations

import json
from typing import Any

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

When reviewing, actively check claims against real physics and biology. Use the Read \
tool to examine canon files relevant to your review. If web search is available, use \
it to verify specific scientific claims, physical constants, or technical feasibility.

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

Cross-reference the new content against established canon files. Use the Read tool \
to examine canon files, searching for character names, dates, locations, and key \
terms to verify consistency.

If no prior canon exists, focus on internal \
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

Use the Read tool to examine canon files for how societal dynamics have been \
established. If web search is available, use it to verify claims about real-world \
institutions, geopolitics, historical precedents, or cultural practices.

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

# Reverse lookup: display name → key name, plus identity mappings
AGENT_NAME_LOOKUP: dict[str, str] = {}
for _key, _display in AGENT_DISPLAY_NAMES.items():
    AGENT_NAME_LOOKUP[_key] = _key
    AGENT_NAME_LOOKUP[_display] = _key
    AGENT_NAME_LOOKUP[_display.lower()] = _key
    AGENT_NAME_LOOKUP[_key.replace("_", " ")] = _key


def normalize_agent_name(name: str) -> str:
    """Map any variant of an agent name back to its key name."""
    return AGENT_NAME_LOOKUP.get(name, AGENT_NAME_LOOKUP.get(name.lower(), name))


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


_PROPOSE_JSON_INSTRUCTIONS = """\

For each change, propose an exact text replacement:
- "original_text" must be an EXACT verbatim substring copied from the file \
(whitespace, punctuation, and all). It must appear exactly once in the file.
- "replacement_text" is your proposed replacement for that text.
- Explain WHY from your domain expertise in "rationale".

If you have no changes to propose, return an empty edits array.

Return your response as JSON matching this schema:
- "summary": one-paragraph summary of your proposed changes
- "edits": array of edit objects, each with:
  - "file": relative file path
  - "original_text": exact text to replace (verbatim from file)
  - "replacement_text": your proposed replacement
  - "rationale": why this edit is needed from your specialty
"""

PROPOSE_MODE_SUFFIX = """\


# YOUR TASK MODE: PROPOSE EDITS

You are in PROPOSE mode. Review the submitted content from your specialty \
perspective and propose CONCRETE edits to improve it. \
Keep edits minimal — change only what is necessary to fix the issue.
""" + _PROPOSE_JSON_INSTRUCTIONS

EXPAND_MODE_SUFFIX = """\


# YOUR TASK MODE: EXPAND CONTENT

You are in EXPAND mode. Your goal is to enrich the submitted content from \
your specialty perspective. Add vivid descriptions, flesh out thin scenes, \
deepen character moments, and develop world-building elements. Preserve the \
existing narrative arc and voice. Propose concrete edits that ADD detail \
where the content would benefit from it.
""" + _PROPOSE_JSON_INSTRUCTIONS

CONTRACT_MODE_SUFFIX = """\


# YOUR TASK MODE: CONTRACT CONTENT

You are in CONTRACT mode. Your goal is to tighten the prose from your \
specialty perspective. Remove redundant words and phrases, consolidate \
repetitive passages, cut filler, and streamline sentences. Preserve all \
meaning and narrative beats with fewer words. Propose concrete edits that \
make the writing more concise.
""" + _PROPOSE_JSON_INSTRUCTIONS

REVIEW_MODE_SUFFIX = """\


# YOUR TASK MODE: REVIEW PROPOSALS

You are in REVIEW mode. Other specialist agents have proposed edits to the \
content. Review ALL proposals and decide whether each is acceptable.

IMPORTANT: The files under review are drafts being improved — they are NOT \
authoritative canon. Only files in the canon directory are authoritative. \
Do not reject edits solely because they change terminology, facts, or \
descriptions established in the file under review. Changing those is often \
the point of the edit. Judge edits against the CANON FILES, not against \
the current draft text.

Consider from YOUR specialty perspective:
- Does the proposed edit maintain correctness in your domain?
- Does the proposed edit introduce new problems?
- Is the edit consistent with established canon files?
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
You are an impartial arbitrator for a fiction review system. Two specialist \
agents have proposed conflicting versions of the same text and cannot agree.

Your job is to pick the better version OR merge them into a single version \
that resolves both agents' concerns. You have no specialty bias — judge \
purely on quality, consistency, and which version better serves the story.

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

_MODE_SUFFIXES: dict[str, str] = {
    "propose": PROPOSE_MODE_SUFFIX,
    "expand": EXPAND_MODE_SUFFIX,
    "contract": CONTRACT_MODE_SUFFIX,
    "review": REVIEW_MODE_SUFFIX,
    "dissent": DISSENT_MODE_SUFFIX,
}


def build_custom_mode_suffix(task_prompt: str) -> str:
    """Build a propose-mode suffix from a custom task prompt."""
    return f"""\


# YOUR TASK MODE: CUSTOM TASK

{task_prompt}

Propose concrete edits from your specialty perspective.
""" + _PROPOSE_JSON_INSTRUCTIONS


def build_agent_system_prompt(
    agent_name: str,
    mode: str,
    config_override: str | None = None,
    custom_task_prompt: str | None = None,
) -> str:
    """Build a complete system prompt for an agent in the given mode.

    mode: "propose", "expand", "contract", "custom", or "review"
    """
    base = config_override or AGENT_PROMPTS[agent_name]
    if mode == "custom" and custom_task_prompt:
        suffix = build_custom_mode_suffix(custom_task_prompt)
    else:
        suffix = _MODE_SUFFIXES.get(mode, PROPOSE_MODE_SUFFIX)
    return base + suffix


KNOWN_TOOLS = {
    "Bash", "Read", "Glob", "Grep", "Edit", "Write",
    "Agent", "Skill", "ToolSearch", "WebSearch", "WebFetch",
}


def build_cli_args(
    agent_name: str,
    system_prompt: str,
    model: str | None,
    repo_root: str,
    max_turns: int = 0,
    allowed_tools: list[str] | None = None,
) -> list[str]:
    """Build command-line arguments for a `claude` CLI invocation."""
    args = [
        "claude",
        "--print",                       # Non-interactive, print result
        "--output-format", "stream-json", # Stream JSON events for tool visibility
        "--verbose",                     # Required for stream-json with --print
    ]

    if max_turns > 0:
        args.extend(["--max-turns", str(max_turns)])

    args += [
        "--system-prompt", system_prompt,
        "--permission-mode", "bypassPermissions",
    ]

    # Read is always available so agents can explore canon files.
    # All other tools are disabled unless explicitly in allowed_tools.
    effective_tools = {"Read"} | set(allowed_tools or [])
    disallowed = KNOWN_TOOLS - effective_tools
    if disallowed:
        args.extend(["--disallowedTools", ",".join(sorted(disallowed))])
    args.extend(["--allowedTools", ",".join(sorted(effective_tools))])

    if model:
        args.extend(["--model", model])

    return args
