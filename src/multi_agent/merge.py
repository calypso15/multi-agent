"""N-way merge of agent edits using diff-match-patch word-level diffing.

Uses dmp for high-quality word-level change detection, then merges all
changes position-based against the original text (like git), avoiding
the context corruption that sequential patch_apply causes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from diff_match_patch import diff_match_patch

if TYPE_CHECKING:
    from multi_agent.models import AgentProposal, FileEdit


@dataclass
class FailedPatch:
    """A change that conflicted with another agent's change."""
    file: str
    agent_name: str
    original_text: str      # region in the original file
    agent_replacement: str  # what this agent wanted instead


@dataclass
class MergeResult:
    """Result of merging all agents' edits for all files."""
    merged_texts: dict[str, str] = field(default_factory=dict)
    failed_patches: list[FailedPatch] = field(default_factory=list)


@dataclass
class _ChangeOp:
    """A single change operation positioned in the original text."""
    start: int          # start position in original
    end: int            # end position in original (range to delete)
    replacement: str    # text to insert
    agent_name: str


def _words_to_chars(
    text1: str, text2: str,
) -> tuple[str, str, list[str]]:
    """Map each word-token to a unique character for word-level diffing.

    Tokens are runs of non-whitespace followed by any trailing whitespace.
    Leading whitespace (before the first word) is its own token.

    Returns (encoded_text1, encoded_text2, token_array) where
    token_array[i] is the token that chr(i) represents.
    """
    token_array: list[str] = [""]  # index 0 unused; diff_charsToLines expects 1-based
    token_to_char: dict[str, str] = {}

    def _encode(text: str) -> str:
        tokens: list[str] = []
        leading = re.match(r"^\s+", text)
        if leading:
            tokens.append(leading.group())
        tokens.extend(re.findall(r"\S+\s*", text[leading.end() if leading else 0:]))

        chars: list[str] = []
        for token in tokens:
            if token not in token_to_char:
                next_index = len(token_array)
                if next_index > 0x10FFFF:
                    raise ValueError(
                        f"Too many unique word tokens ({next_index}) for "
                        "character encoding. Maximum supported is 1,114,111."
                    )
                token_array.append(token)
                token_to_char[token] = chr(next_index)
            chars.append(token_to_char[token])
        return "".join(chars)

    return _encode(text1), _encode(text2), token_array


def _word_diff(original: str, modified: str) -> list[tuple[int, str]]:
    """Compute a word-level diff between original and modified text.

    Returns a list of (operation, text) tuples where operation is
    0 (EQUAL), -1 (DELETE), or 1 (INSERT).
    """
    dmp = diff_match_patch()
    enc_orig, enc_mod, token_array = _words_to_chars(original, modified)

    diffs = dmp.diff_main(enc_orig, enc_mod, checklines=False)
    dmp.diff_cleanupSemantic(diffs)

    # Map encoded characters back to word tokens
    dmp.diff_charsToLines(diffs, token_array)

    return diffs


def _diffs_to_ops(diffs: list[tuple[int, str]], agent_name: str) -> list[_ChangeOp]:
    """Convert word-level diffs to position-based change operations.

    Walks through the diffs tracking position in the original text.
    Groups adjacent DELETE+INSERT into a single replacement operation.
    """
    ops: list[_ChangeOp] = []
    pos = 0  # current position in original text

    i = 0
    while i < len(diffs):
        op, text = diffs[i]
        if op == 0:  # EQUAL — advance position
            pos += len(text)
            i += 1
        elif op == -1:  # DELETE
            start = pos
            deleted_len = len(text)
            pos += deleted_len
            # Check if followed by INSERT (making it a replacement)
            inserted = ""
            if i + 1 < len(diffs) and diffs[i + 1][0] == 1:
                inserted = diffs[i + 1][1]
                i += 1
            ops.append(_ChangeOp(start, start + deleted_len, inserted, agent_name))
            i += 1
        elif op == 1:  # INSERT (without preceding DELETE)
            ops.append(_ChangeOp(pos, pos, text, agent_name))
            i += 1

    return ops


def _ops_overlap(a: _ChangeOp, b: _ChangeOp) -> bool:
    """Check if two operations overlap in the original text.

    Pure insertions at the same point are considered overlapping only
    if they insert different text.
    """
    if a.start == a.end and b.start == b.end:
        # Both are pure insertions at the same position
        return a.start == b.start and a.replacement != b.replacement
    if a.start == a.end:
        # a is pure insertion — overlaps if inside b's delete range
        return b.start < a.start < b.end
    if b.start == b.end:
        return a.start < b.start < a.end
    return a.start < b.end and b.start < a.end


def _apply_edits_to_text(content: str, edits: list[FileEdit]) -> str:
    """Apply a single agent's edits to text.

    Locates each edit by position, sorts, and applies back-to-front so
    earlier replacements don't shift later positions.
    """
    located: list[tuple[int, int, str]] = []
    for edit in edits:
        pos = content.find(edit.original_text)
        if pos < 0:
            continue
        located.append((pos, pos + len(edit.original_text), edit.replacement_text))

    located.sort(key=lambda t: t[0])

    for start, end, replacement in reversed(located):
        content = content[:start] + replacement + content[end:]

    return content


def merge_agent_edits(
    file_contents: dict[str, str],
    proposals: list[AgentProposal],
) -> MergeResult:
    """Merge all agents' edits using word-level diffing and position-based merge.

    For each file:
    1. Apply each agent's edits independently to get per-agent versions
    2. Compute word-level diffs from original to each agent's version
    3. Convert diffs to position-based change operations
    4. Merge all operations: non-overlapping apply cleanly, overlapping
       operations are conflicts sent to arbitration
    5. Apply merged operations back-to-front to the original

    Returns MergeResult with merged texts and any conflicts.
    """
    merged_texts: dict[str, str] = {}
    failed_patches: list[FailedPatch] = []

    # Collect all files that have edits
    all_edited_files: set[str] = set()
    for proposal in proposals:
        for edit in proposal.edits:
            all_edited_files.add(edit.file)

    for filepath in sorted(all_edited_files):
        original = file_contents.get(filepath, "")

        # Collect change operations from all agents
        all_ops: list[_ChangeOp] = []

        for proposal in proposals:
            agent_edits = [e for e in proposal.edits if e.file == filepath]
            if not agent_edits:
                continue

            agent_version = _apply_edits_to_text(original, agent_edits)
            if agent_version == original:
                continue

            diffs = _word_diff(original, agent_version)
            all_ops.extend(_diffs_to_ops(diffs, proposal.agent_name))

        if not all_ops:
            continue

        # Sort by start position, then by end (larger ranges first for overlap detection)
        all_ops.sort(key=lambda o: (o.start, -o.end))

        # Deduplicate identical operations from different agents
        seen: set[tuple[int, int, str]] = set()
        unique_ops: list[_ChangeOp] = []
        for op in all_ops:
            key = (op.start, op.end, op.replacement)
            if key not in seen:
                seen.add(key)
                unique_ops.append(op)

        # Partition into non-conflicting and conflicting operations
        accepted: list[_ChangeOp] = []
        for op in unique_ops:
            conflict_found = False
            for existing in accepted:
                if _ops_overlap(op, existing):
                    # Conflict — record both sides for arbitration
                    failed_patches.append(FailedPatch(
                        file=filepath,
                        agent_name=op.agent_name,
                        original_text=original[op.start:op.end],
                        agent_replacement=op.replacement,
                    ))
                    conflict_found = True
                    break
            if not conflict_found:
                accepted.append(op)

        # Apply accepted operations back-to-front to preserve positions
        merged = original
        for op in sorted(accepted, key=lambda o: o.start, reverse=True):
            merged = merged[:op.start] + op.replacement + merged[op.end:]

        if merged != original:
            merged_texts[filepath] = merged

    return MergeResult(
        merged_texts=merged_texts,
        failed_patches=failed_patches,
    )


def apply_arbitration_to_merged(
    merged_texts: dict[str, str],
    filepath: str,
    text_to_replace: str,
    replacement_text: str,
) -> None:
    """Apply an arbitration result to the merged text.

    Finds text_to_replace (the winning agent's version) in the merged
    text and replaces it with the arbitrated replacement_text.
    """
    merged = merged_texts.get(filepath, "")
    merged_texts[filepath] = merged.replace(text_to_replace, replacement_text, 1)
