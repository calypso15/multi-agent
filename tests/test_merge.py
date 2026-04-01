"""Tests for the word-level merge system and edit locking."""

from multi_agent.models import (
    AgentProposal,
    AgentReviewResponse,
    FileEdit,
    ProposalReview,
)
from multi_agent.consensus import (
    _edit_overlaps_locked,
    _filter_self_modified_edits,
    deduplicate_edits,
    merge_proposals,
)
from multi_agent.merge import apply_arbitration_to_merged, merge_agent_edits


# --- merge_agent_edits ---


class TestMergeNonOverlapping:
    def test_edits_from_different_agents_merge(self):
        r = merge_agent_edits(
            {"f.md": "The quick brown fox jumps over the lazy dog."},
            [
                AgentProposal(agent_name="a", edits=[
                    FileEdit("f.md", "quick brown", "swift auburn", ""),
                ], summary=""),
                AgentProposal(agent_name="b", edits=[
                    FileEdit("f.md", "lazy dog", "sleeping hound", ""),
                ], summary=""),
            ],
        )
        assert "swift auburn" in r.merged_texts["f.md"]
        assert "sleeping hound" in r.merged_texts["f.md"]
        assert not r.failed_patches

    def test_nearby_edits_in_same_sentence(self):
        r = merge_agent_edits(
            {"f.md": "The ship accelerated to 3c using conventional thrusters."},
            [
                AgentProposal(agent_name="a", edits=[
                    FileEdit("f.md", "accelerated to 3c", "accelerated to 0.3c", ""),
                ], summary=""),
                AgentProposal(agent_name="b", edits=[
                    FileEdit("f.md", "conventional thrusters", "ion drive engines", ""),
                ], summary=""),
            ],
        )
        assert "0.3c" in r.merged_texts["f.md"]
        assert "ion drive engines" in r.merged_texts["f.md"]
        assert not r.failed_patches

    def test_multi_file_multi_agent(self):
        r = merge_agent_edits(
            {
                "ch1.md": "Chapter 1. The aliens arrived Tuesday.",
                "ch2.md": "Chapter 2. The president spoke.",
            },
            [
                AgentProposal(agent_name="a", edits=[
                    FileEdit("ch1.md", "Tuesday", "Wednesday", ""),
                    FileEdit("ch2.md", "president", "prime minister", ""),
                ], summary=""),
                AgentProposal(agent_name="b", edits=[
                    FileEdit("ch1.md", "aliens arrived", "visitors landed", ""),
                ], summary=""),
            ],
        )
        assert "Wednesday" in r.merged_texts["ch1.md"]
        assert "visitors landed" in r.merged_texts["ch1.md"]
        assert "prime minister" in r.merged_texts["ch2.md"]
        assert not r.failed_patches


class TestMergeIdentical:
    def test_identical_edits_applied_once(self):
        r = merge_agent_edits(
            {"f.md": "The date was January 5th, 2045."},
            [
                AgentProposal(agent_name="a", edits=[
                    FileEdit("f.md", "January 5th, 2045", "March 12th, 2046", ""),
                ], summary=""),
                AgentProposal(agent_name="b", edits=[
                    FileEdit("f.md", "January 5th, 2045", "March 12th, 2046", ""),
                ], summary=""),
            ],
        )
        assert r.merged_texts["f.md"].count("March 12th, 2046") == 1
        assert not r.failed_patches


class TestMergeConflicts:
    def test_same_text_different_replacements(self):
        r = merge_agent_edits(
            {"f.md": "The radiation levels were dangerously high."},
            [
                AgentProposal(agent_name="sci", edits=[
                    FileEdit("f.md", "dangerously high", "within safe limits", ""),
                ], summary=""),
                AgentProposal(agent_name="pol", edits=[
                    FileEdit("f.md", "dangerously high", "catastrophically elevated", ""),
                ], summary=""),
            ],
        )
        assert len(r.failed_patches) == 1
        assert r.failed_patches[0].agent_name == "pol"
        assert "within safe limits" in r.merged_texts["f.md"]

    def test_overlapping_different_regions(self):
        r = merge_agent_edits(
            {"f.md": "The quick brown fox jumps over the lazy dog."},
            [
                AgentProposal(agent_name="a", edits=[
                    FileEdit("f.md", "quick brown fox", "swift auburn fox", ""),
                ], summary=""),
                AgentProposal(agent_name="b", edits=[
                    FileEdit("f.md", "brown fox jumps", "red fox leaps", ""),
                ], summary=""),
            ],
        )
        assert len(r.failed_patches) == 1


class TestMergeGarbledOutput:
    """Regression test for the garbled merge bug where old content survived."""

    def test_heading_change_plus_bullet_replacement(self):
        original = (
            "### Winter 2036-2037 - Ice Reformation\n"
            "\n"
            "- Melt zone begins refreezing within weeks\n"
            "- Multi-year ice fails to reform\n"
            "- New ice remains thinner and more mobile\n"
        )
        r = merge_agent_edits(
            {"f.md": original},
            [
                AgentProposal(agent_name="a", edits=[
                    FileEdit("f.md", "Ice Reformation", "Failed Ice Reformation", ""),
                ], summary=""),
                AgentProposal(agent_name="b", edits=[
                    FileEdit(
                        "f.md",
                        "- Melt zone begins refreezing within weeks\n"
                        "- Multi-year ice fails to reform\n"
                        "- New ice remains thinner and more mobile",
                        "- Blast-disrupted zone fails to refreeze\n"
                        "- Persistent open-water zone forms\n"
                        "- Surrounding ice patterns permanently altered",
                        "",
                    ),
                ], summary=""),
            ],
        )
        merged = r.merged_texts["f.md"]
        assert "Failed Ice Reformation" in merged
        assert "Blast-disrupted" in merged
        assert "Melt zone" not in merged, "Old bullet survived in merge"
        assert "Multi-year ice fails" not in merged, "Old bullet survived in merge"
        assert not r.failed_patches


# --- apply_arbitration_to_merged ---


class TestApplyArbitration:
    def test_replaces_winning_text(self):
        merged_texts = {"f.md": "The radiation levels were within safe limits."}
        apply_arbitration_to_merged(
            merged_texts, "f.md",
            "within safe limits", "elevated but manageable",
        )
        assert "elevated but manageable" in merged_texts["f.md"]


# --- _edit_overlaps_locked ---


class TestEditOverlapsLocked:
    FILE_CONTENTS = {"f.md": "The quick brown fox jumps over the lazy dog in the park."}
    # "quick brown fox jumps" is at positions 4-28
    LOCKED = {"f.md": [(4, 28)]}

    def test_exact_match(self):
        edit = FileEdit("f.md", "quick brown fox jumps", "X", "")
        assert _edit_overlaps_locked(edit, self.FILE_CONTENTS, self.LOCKED)

    def test_partial_overlap_start(self):
        edit = FileEdit("f.md", "The quick brown", "X", "")
        assert _edit_overlaps_locked(edit, self.FILE_CONTENTS, self.LOCKED)

    def test_partial_overlap_end(self):
        edit = FileEdit("f.md", "fox jumps over", "X", "")
        assert _edit_overlaps_locked(edit, self.FILE_CONTENTS, self.LOCKED)

    def test_no_overlap(self):
        edit = FileEdit("f.md", "lazy dog", "X", "")
        assert not _edit_overlaps_locked(edit, self.FILE_CONTENTS, self.LOCKED)

    def test_different_file(self):
        edit = FileEdit("g.md", "quick brown fox jumps", "X", "")
        assert not _edit_overlaps_locked(edit, self.FILE_CONTENTS, self.LOCKED)

    def test_no_locked_regions(self):
        edit = FileEdit("f.md", "quick brown", "X", "")
        assert not _edit_overlaps_locked(edit, self.FILE_CONTENTS, {})


# --- merge_proposals with locked regions ---


class TestMergeProposalsLocked:
    FILE_CONTENTS = {"f.md": "The quick brown fox jumps over the lazy dog."}
    # Lock "quick brown fox jumps" (positions 4-28)
    LOCKED = {"f.md": [(4, 28)]}

    def _proposals(self):
        return [
            AgentProposal(agent_name="sci", edits=[
                FileEdit("f.md", "quick brown fox jumps", "arbitrated text", ""),
                FileEdit("f.md", "lazy dog", "sleeping hound", ""),
            ], summary=""),
        ]

    def test_locked_edit_protected(self):
        reviews = [
            AgentReviewResponse(
                agent_name="pol", all_approved=False,
                proposal_reviews=[
                    ProposalReview(original_agent="sci", edit_index=0, verdict="MODIFY",
                                   modified_replacement="pol version", rationale=""),
                    ProposalReview(original_agent="sci", edit_index=1, verdict="MODIFY",
                                   modified_replacement="happy cat", rationale=""),
                ],
                summary="",
            ),
        ]
        result = merge_proposals(
            self._proposals(), reviews, self.LOCKED, self.FILE_CONTENTS,
        )
        assert result[0].edits[0].replacement_text == "arbitrated text"
        assert result[0].edits[1].replacement_text == "happy cat"

    def test_partial_overlap_also_locked(self):
        proposals = [
            AgentProposal(agent_name="sci", edits=[
                FileEdit("f.md", "brown fox jumps over", "new text", ""),
            ], summary=""),
        ]
        reviews = [
            AgentReviewResponse(
                agent_name="pol", all_approved=False,
                proposal_reviews=[
                    ProposalReview(original_agent="sci", edit_index=0, verdict="MODIFY",
                                   modified_replacement="pol text", rationale=""),
                ],
                summary="",
            ),
        ]
        result = merge_proposals(proposals, reviews, self.LOCKED, self.FILE_CONTENTS)
        assert result[0].edits[0].replacement_text == "new text"

    def test_no_locks_normal_behavior(self):
        reviews = [
            AgentReviewResponse(
                agent_name="pol", all_approved=False,
                proposal_reviews=[
                    ProposalReview(original_agent="sci", edit_index=0, verdict="MODIFY",
                                   modified_replacement="pol version", rationale=""),
                ],
                summary="",
            ),
        ]
        result = merge_proposals(self._proposals(), reviews)
        assert result[0].edits[0].replacement_text == "pol version"


# --- count_approvals ---


class TestCountApprovals:
    def test_all_approved(self):
        from multi_agent.models import count_approvals
        reviews = [
            AgentReviewResponse(agent_name="a", all_approved=True, proposal_reviews=[], summary=""),
            AgentReviewResponse(agent_name="b", all_approved=True, proposal_reviews=[], summary=""),
        ]
        assert count_approvals(reviews) == 2

    def test_error_not_counted(self):
        from multi_agent.models import count_approvals
        reviews = [
            AgentReviewResponse(agent_name="a", all_approved=True, proposal_reviews=[], summary="", error="fail"),
            AgentReviewResponse(agent_name="b", all_approved=True, proposal_reviews=[], summary=""),
        ]
        assert count_approvals(reviews) == 1

    def test_rejected_not_counted(self):
        from multi_agent.models import count_approvals
        reviews = [
            AgentReviewResponse(agent_name="a", all_approved=False, proposal_reviews=[], summary=""),
            AgentReviewResponse(agent_name="b", all_approved=True, proposal_reviews=[], summary=""),
        ]
        assert count_approvals(reviews) == 1

    def test_empty_list(self):
        from multi_agent.models import count_approvals
        assert count_approvals([]) == 0


# --- sanitize_edit_path ---


class TestSanitizeEditPath:
    def test_normal_path(self):
        from multi_agent.models import sanitize_edit_path
        assert sanitize_edit_path("canon/chapter-01.md") == "canon/chapter-01.md"

    def test_absolute_path_rejected(self):
        from multi_agent.models import sanitize_edit_path
        import pytest
        with pytest.raises(ValueError, match="Absolute path"):
            sanitize_edit_path("/etc/passwd")

    def test_traversal_rejected(self):
        from multi_agent.models import sanitize_edit_path
        import pytest
        with pytest.raises(ValueError, match="Path traversal"):
            sanitize_edit_path("../../etc/passwd")

    def test_sneaky_traversal_rejected(self):
        from multi_agent.models import sanitize_edit_path
        import pytest
        with pytest.raises(ValueError, match="Path traversal"):
            sanitize_edit_path("canon/../../etc/passwd")

    def test_simple_filename(self):
        from multi_agent.models import sanitize_edit_path
        assert sanitize_edit_path("readme.md") == "readme.md"


# --- deduplicate_edits with 3-tuple key ---


class TestDeduplicateEdits:
    def test_exact_duplicates_removed(self):
        edits = [
            FileEdit("f.md", "foo", "bar", "r1"),
            FileEdit("f.md", "foo", "bar", "r2"),
        ]
        kept, dropped = deduplicate_edits(edits)
        assert len(kept) == 1
        assert len(dropped) == 1

    def test_different_replacements_kept(self):
        edits = [
            FileEdit("f.md", "foo", "bar", "r1"),
            FileEdit("f.md", "foo", "baz", "r2"),
        ]
        kept, dropped = deduplicate_edits(edits)
        assert len(kept) == 2
        assert len(dropped) == 0

    def test_different_files_kept(self):
        edits = [
            FileEdit("a.md", "foo", "bar", "r1"),
            FileEdit("b.md", "foo", "bar", "r2"),
        ]
        kept, dropped = deduplicate_edits(edits)
        assert len(kept) == 2
        assert len(dropped) == 0


# --- _filter_self_modified_edits ---


class TestFilterSelfModifiedEdits:
    """Agents should not re-review edits they last modified."""

    def _proposals(self):
        return [
            AgentProposal(agent_name="socio", edits=[
                FileEdit("f.md", "aaa", "bbb", "edit 0"),
                FileEdit("f.md", "ccc", "ddd", "edit 1"),
                FileEdit("f.md", "eee", "fff", "edit 2"),
            ], summary=""),
            AgentProposal(agent_name="canon", edits=[
                FileEdit("f.md", "ggg", "hhh", "edit 0"),
            ], summary=""),
        ]

    def test_self_modified_edit_excluded(self):
        """sci modified socio's edit 1 → sci should not see it next round."""
        prev_reviews = [
            AgentReviewResponse(
                agent_name="sci", all_approved=False,
                proposal_reviews=[
                    ProposalReview(original_agent="socio", edit_index=1,
                                   verdict="MODIFY", modified_replacement="new", rationale=""),
                ],
                summary="",
            ),
        ]
        filtered = _filter_self_modified_edits(
            self._proposals(), "sci", prev_reviews,
        )
        socio = next(p for p in filtered if p.agent_name == "socio")
        # edit 0 and edit 2 survive, edit 1 is removed
        assert len(socio.edits) == 2
        assert socio.edits[0].original_text == "aaa"
        assert socio.edits[1].original_text == "eee"

    def test_other_agents_still_see_edit(self):
        """canon should still see the edit sci modified."""
        prev_reviews = [
            AgentReviewResponse(
                agent_name="sci", all_approved=False,
                proposal_reviews=[
                    ProposalReview(original_agent="socio", edit_index=1,
                                   verdict="MODIFY", modified_replacement="new", rationale=""),
                ],
                summary="",
            ),
        ]
        filtered = _filter_self_modified_edits(
            self._proposals(), "canon", prev_reviews,
        )
        socio = next(p for p in filtered if p.agent_name == "socio")
        assert len(socio.edits) == 3  # all edits visible

    def test_later_modifier_wins(self):
        """If canon modified AFTER sci, sci should still see the edit."""
        prev_reviews = [
            AgentReviewResponse(
                agent_name="sci", all_approved=False,
                proposal_reviews=[
                    ProposalReview(original_agent="socio", edit_index=1,
                                   verdict="MODIFY", modified_replacement="sci ver", rationale=""),
                ],
                summary="",
            ),
            AgentReviewResponse(
                agent_name="canon", all_approved=False,
                proposal_reviews=[
                    ProposalReview(original_agent="socio", edit_index=1,
                                   verdict="MODIFY", modified_replacement="canon ver", rationale=""),
                ],
                summary="",
            ),
        ]
        filtered = _filter_self_modified_edits(
            self._proposals(), "sci", prev_reviews,
        )
        socio = next(p for p in filtered if p.agent_name == "socio")
        # canon was last modifier → sci should review it
        assert len(socio.edits) == 3

    def test_no_previous_reviews_no_filter(self):
        """First round — no filtering needed."""
        filtered = _filter_self_modified_edits(
            self._proposals(), "sci", [],
        )
        socio = next(p for p in filtered if p.agent_name == "socio")
        assert len(socio.edits) == 3

    def test_does_not_mutate_originals(self):
        """Filtering should return copies, not mutate the input proposals."""
        proposals = self._proposals()
        prev_reviews = [
            AgentReviewResponse(
                agent_name="sci", all_approved=False,
                proposal_reviews=[
                    ProposalReview(original_agent="socio", edit_index=1,
                                   verdict="MODIFY", modified_replacement="new", rationale=""),
                ],
                summary="",
            ),
        ]
        _filter_self_modified_edits(proposals, "sci", prev_reviews)
        # Original should be untouched
        assert len(proposals[0].edits) == 3
