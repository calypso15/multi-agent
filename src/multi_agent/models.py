"""Dataclasses, parsing utilities, and typed events for the review loop."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Union


# --- Token usage ---


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float = 0.0

    def __iadd__(self, other: TokenUsage) -> TokenUsage:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cost_usd += other.cost_usd
        return self


def extract_usage(outer: dict[str, Any]) -> TokenUsage:
    """Extract token usage from the claude CLI JSON envelope."""
    usage = outer.get("usage", {})
    return TokenUsage(
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_read_input_tokens=usage.get("cache_read_input_tokens", 0),
        cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0),
        cost_usd=outer.get("total_cost_usd", 0.0),
    )


# --- Core domain objects ---


@dataclass
class FileEdit:
    file: str
    original_text: str
    replacement_text: str
    rationale: str


@dataclass
class AgentProposal:
    agent_name: str
    edits: list[FileEdit]
    summary: str
    duration_seconds: float = 0.0
    error: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class ProposalReview:
    original_agent: str
    edit_index: int
    verdict: str  # "APPROVE" or "MODIFY"
    modified_replacement: str | None = None
    rationale: str = ""


@dataclass
class AgentReviewResponse:
    agent_name: str
    all_approved: bool
    proposal_reviews: list[ProposalReview]
    summary: str
    duration_seconds: float = 0.0
    error: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class IterationRound:
    round_number: int
    reviews: list[AgentReviewResponse]
    consensus_reached: bool


@dataclass
class Dissent:
    agent_name: str
    opinion: str
    duration_seconds: float = 0.0
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class ContestedEdit:
    """An edit that two agents keep modifying back and forth."""
    file: str
    original_text: str
    versions: dict[str, str]  # agent_name -> replacement_text
    rationales: dict[str, str]  # agent_name -> rationale


@dataclass
class ArbitrationResult:
    file: str
    original_text: str
    replacement_text: str
    rationale: str
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class IterationResult:
    consensus_reached: bool
    final_edits: list[FileEdit]
    proposals: list[AgentProposal]
    rounds: list[IterationRound]
    total_duration_seconds: float
    files_reviewed: list[str]
    merged_texts: dict[str, str] = field(default_factory=dict)
    total_usage: TokenUsage = field(default_factory=TokenUsage)
    dissents: list[Dissent] = field(default_factory=list)
    best_round: int = -1
    best_approvals: int = 0
    stalled: bool = False


# --- Typed phase events (replacing stringly-typed on_phase callbacks) ---


@dataclass
class ProposeDone:
    proposals: list[AgentProposal]


@dataclass
class ReviewDone:
    round_number: int
    reviews: list[AgentReviewResponse]
    consensus_threshold: int


@dataclass
class ArbitrationStart:
    contested: list[ContestedEdit]


@dataclass
class ArbitrationDone:
    results: list[ArbitrationResult]


@dataclass
class DissentsDone:
    dissents: list[Dissent]


PhaseEvent = Union[ProposeDone, ReviewDone, ArbitrationStart, ArbitrationDone, DissentsDone]


# --- Helpers ---


def count_approvals(reviews: list[AgentReviewResponse]) -> int:
    """Count reviews that approved with no error."""
    return sum(1 for r in reviews if r.all_approved and r.error is None)


def sanitize_edit_path(filepath: str) -> str:
    """Validate and normalize a file path from agent output.

    Rejects absolute paths, path traversal (``..``), and paths that would
    escape the working directory after normalization.

    Returns the normalized relative path.
    """
    if os.path.isabs(filepath):
        raise ValueError(f"Absolute path not allowed in edit: {filepath}")
    normalized = os.path.normpath(filepath)
    if normalized.startswith("..") or normalized.startswith(os.sep):
        raise ValueError(f"Path traversal not allowed in edit: {filepath}")
    return normalized


# --- JSON parsing ---


def extract_json(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from text, handling markdown code fences."""
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    for marker in ("```json", "```"):
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass

    # Try finding first { ... } block, respecting JSON string boundaries
    brace_start = text.find("{")
    if brace_start >= 0:
        depth = 0
        in_string = False
        escape = False
        for i in range(brace_start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == '\\' and in_string:
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i + 1])
                    except json.JSONDecodeError:
                        break

    return None


def unwrap_result(result_json: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract the agent's JSON response from the claude CLI result envelope.

    The stream-json result event has a "result" field containing the agent's
    text output, which itself should be JSON.
    """
    if result_json is None:
        return None

    inner = result_json.get("result")
    if inner is None:
        return None

    if isinstance(inner, dict):
        return inner

    if isinstance(inner, str):
        return extract_json(inner)

    return None


def parse_edits(raw_edits: list[dict[str, Any]]) -> list[FileEdit]:
    """Parse edit dicts into FileEdit objects with path sanitization."""
    edits = []
    for item in raw_edits:
        filepath = item.get("file", "")
        try:
            filepath = sanitize_edit_path(filepath)
        except ValueError:
            continue  # skip edits with invalid paths
        edits.append(FileEdit(
            file=filepath,
            original_text=item.get("original_text", ""),
            replacement_text=item.get("replacement_text", ""),
            rationale=item.get("rationale", ""),
        ))
    return edits


def parse_proposal_reviews(
    raw: list[dict[str, Any]],
    normalizer: Callable[[str], str] = lambda x: x,
) -> list[ProposalReview]:
    """Parse proposal review dicts into ProposalReview objects."""
    reviews = []
    for item in raw:
        reviews.append(ProposalReview(
            original_agent=normalizer(item.get("original_agent", "")),
            edit_index=item.get("edit_index", 0),
            verdict=item.get("verdict", "APPROVE"),
            modified_replacement=item.get("modified_replacement"),
            rationale=item.get("rationale", ""),
        ))
    return reviews
