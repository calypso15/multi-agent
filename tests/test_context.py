"""Tests for prompt building and git context extraction."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from multi_agent.context import (
    _apply_edits_to_text,
    _build_unified_diff,
    _propose_instructions,
    _reference_section,
    apply_merged_texts,
    build_diff_preview_from_merged,
    build_propose_prompt,
    build_review_round_prompt,
    find_git_root,
    get_staged_content,
    get_staged_files,
    load_reference,
)
from multi_agent.models import AgentProposal, FileEdit


# --- _reference_section ---


class TestReferenceSection:
    def test_non_empty_reference(self):
        ref = {"lore/world.md": "World building content here."}
        result = _reference_section(ref)
        assert "REFERENCE FILES" in result
        assert "lore/world.md" in result
        assert "KB" in result

    def test_empty_reference(self):
        result = _reference_section({})
        assert "No reference files" in result

    def test_files_sorted_by_path(self):
        ref = {"z.md": "zzz", "a.md": "aaa"}
        result = _reference_section(ref)
        assert result.index("a.md") < result.index("z.md")


# --- _propose_instructions ---


class TestProposeInstructions:
    def test_includes_severity_classification(self):
        result = _propose_instructions("minor")
        assert "severity" in result.lower()
        assert "YOUR TASK" in result

    def test_minor_severity_filter_note(self):
        result = _propose_instructions("minor")
        assert "minor" in result

    def test_major_severity_filter_note(self):
        result = _propose_instructions("major")
        assert "critical" in result
        assert "major" in result

    def test_suggestion_no_filter_note(self):
        result = _propose_instructions("suggestion")
        assert "filtered out" not in result


# --- build_propose_prompt ---


class TestBuildProposePrompt:
    def test_includes_file_contents(self):
        result = build_propose_prompt(
            {"story.md": "Once upon a time."}, {},
        )
        assert "story.md" in result
        assert "Once upon a time." in result

    def test_includes_reference_section(self):
        result = build_propose_prompt(
            {"f.md": "text"}, {"ref.md": "reference"},
        )
        assert "REFERENCE FILES" in result

    def test_staged_diff_included(self):
        result = build_propose_prompt(
            {"f.md": "text"}, {},
            staged_diff="@@ -1 +1 @@\n-old\n+new",
        )
        assert "DIFF" in result
        assert "-old" in result

    def test_no_staged_diff_omits_section(self):
        result = build_propose_prompt({"f.md": "text"}, {})
        assert "DIFF" not in result


# --- build_review_round_prompt ---


class TestBuildReviewRoundPrompt:
    def test_includes_proposals(self):
        proposals = [
            AgentProposal(
                agent_name="alpha",
                edits=[FileEdit("f.md", "old", "new", "fix", "minor")],
                summary="Summary.",
            ),
        ]
        result = build_review_round_prompt(proposals, {"f.md": "old text"}, {}, 0)
        assert "old" in result
        assert "new" in result

    def test_display_names_used(self):
        proposals = [
            AgentProposal(
                agent_name="alpha",
                edits=[FileEdit("f.md", "old", "new", "fix", "minor")],
                summary="Summary.",
            ),
        ]
        result = build_review_round_prompt(
            proposals, {"f.md": "old text"}, {}, 0,
            display_names={"alpha": "Dr. Alpha"},
        )
        assert "Dr. Alpha" in result

    def test_round_number_one_indexed(self):
        proposals = [
            AgentProposal(
                agent_name="alpha",
                edits=[FileEdit("f.md", "old", "new", "fix", "minor")],
                summary="Summary.",
            ),
        ]
        result = build_review_round_prompt(proposals, {"f.md": "text"}, {}, 2)
        assert "Round 3" in result

    def test_severity_shown_in_edit_headers(self):
        proposals = [
            AgentProposal(
                agent_name="alpha",
                edits=[FileEdit("f.md", "old", "new", "fix", "critical")],
                summary="Summary.",
            ),
        ]
        result = build_review_round_prompt(proposals, {"f.md": "old text"}, {}, 0)
        assert "(critical)" in result


# --- _apply_edits_to_text ---


class TestApplyEditsToText:
    def test_single_edit(self):
        result = _apply_edits_to_text(
            "The quick brown fox.",
            [FileEdit("f.md", "quick brown", "slow red", "", "minor")],
        )
        assert result == "The slow red fox."

    def test_multiple_non_overlapping_edits(self):
        result = _apply_edits_to_text(
            "The quick brown fox jumps over the lazy dog.",
            [
                FileEdit("f.md", "quick brown", "slow red", "", "minor"),
                FileEdit("f.md", "lazy dog", "sleepy cat", "", "minor"),
            ],
        )
        assert "slow red" in result
        assert "sleepy cat" in result

    def test_missing_original_skipped(self):
        result = _apply_edits_to_text(
            "The quick brown fox.",
            [FileEdit("f.md", "not found", "replacement", "", "minor")],
        )
        assert result == "The quick brown fox."


# --- _build_unified_diff ---


class TestBuildUnifiedDiff:
    def test_produces_diff_with_file_paths(self):
        result = _build_unified_diff("story.md", "old line\n", "new line\n")
        assert "a/story.md" in result
        assert "b/story.md" in result
        assert "-old line" in result
        assert "+new line" in result


# --- build_diff_preview_from_merged ---


class TestBuildDiffPreviewFromMerged:
    def test_changed_files_produce_diff(self):
        result = build_diff_preview_from_merged(
            {"f.md": "new content\n"},
            {"f.md": "old content\n"},
        )
        assert "-old content" in result
        assert "+new content" in result

    def test_unchanged_files_omitted(self):
        result = build_diff_preview_from_merged(
            {"f.md": "same"},
            {"f.md": "same"},
        )
        assert result == ""


# --- Mocked git/subprocess tests ---


class TestFindGitRoot:
    @patch("multi_agent.context.subprocess.run")
    def test_returns_path_from_stdout(self, mock_run):
        mock_run.return_value = MagicMock(stdout="/home/user/repo\n")
        result = find_git_root()
        assert result == Path("/home/user/repo")

    @patch("multi_agent.context.subprocess.run")
    def test_passes_cwd_when_start_given(self, mock_run):
        mock_run.return_value = MagicMock(stdout="/repo\n")
        find_git_root(start=Path("/some/dir"))
        call_kwargs = mock_run.call_args
        assert call_kwargs.kwargs.get("cwd") == "/some/dir" or \
               call_kwargs[1].get("cwd") == "/some/dir"


class TestGetStagedFiles:
    @patch("multi_agent.context.subprocess.run")
    def test_filters_by_patterns(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="chapter.md\nscript.py\nnotes.txt\n",
        )
        result = get_staged_files(Path("/repo"), ["*.md", "*.txt"])
        filenames = [p.name for p in result]
        assert "chapter.md" in filenames
        assert "notes.txt" in filenames
        assert "script.py" not in filenames

    @patch("multi_agent.context.subprocess.run")
    def test_empty_diff_returns_empty(self, mock_run):
        mock_run.return_value = MagicMock(stdout="")
        result = get_staged_files(Path("/repo"), ["*.md"])
        assert result == []


class TestGetStagedContent:
    @patch("multi_agent.context.subprocess.run")
    def test_returns_file_content(self, mock_run):
        mock_run.return_value = MagicMock(stdout="file content here")
        result = get_staged_content(Path("/repo"), Path("chapter.md"))
        assert result == "file content here"


class TestLoadReference:
    @patch("multi_agent.context.subprocess.run")
    def test_loads_matching_files(self, mock_run):
        def side_effect(args, **kwargs):
            if "rev-parse" in args and "HEAD" in args:
                return MagicMock(returncode=0)
            if "ls-tree" in args:
                return MagicMock(stdout="reference/lore.md\nsrc/main.py\n")
            if "show" in args:
                return MagicMock(returncode=0, stdout="lore content")
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect
        result = load_reference(Path("/repo"), ["reference"], ["*.md"])
        assert "reference/lore.md" in result
        assert result["reference/lore.md"] == "lore content"

    @patch("multi_agent.context.subprocess.run")
    def test_respects_max_size(self, mock_run):
        # Each file is 600 bytes; first fits in 1 KB, second pushes over
        content = "x" * 600

        def side_effect(args, **kwargs):
            if "rev-parse" in args and "HEAD" in args:
                return MagicMock(returncode=0)
            if "ls-tree" in args:
                return MagicMock(stdout="reference/a.md\nreference/b.md\n")
            if "show" in args:
                return MagicMock(returncode=0, stdout=content)
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect
        result = load_reference(Path("/repo"), ["reference"], ["*.md"], max_size_kb=1)
        assert len(result) == 1


class TestApplyMergedTexts:
    def test_writes_files(self, tmp_path):
        (tmp_path / "f.md").write_text("old")
        modified = apply_merged_texts(tmp_path, {"f.md": "new"})
        assert modified == ["f.md"]
        assert (tmp_path / "f.md").read_text() == "new"

    def test_path_traversal_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="escapes repo root"):
            apply_merged_texts(tmp_path, {"../escape.md": "bad"})
