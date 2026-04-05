"""Tests for orchestration helpers and mocked phase runners.

Functions already tested in test_merge.py are NOT duplicated here:
_edit_overlaps_locked, deduplicate_edits, merge_proposals.
"""

from unittest.mock import AsyncMock

import pytest

from multi_agent.backend import AgentResult
from multi_agent.config import (
    AgentConfig,
    GeneralConfig,
    MultiAgentConfig,
    ResolvedAgentSettings,
    ResolvedRunConfig,
    resolve_run_config,
)
from multi_agent.arbitration import (
    build_arbitration_prompt,
    build_dissent_prompt,
    detect_stall,
    find_contested_edits,
)
from multi_agent.consensus import (
    _last_modifiers,
    _run_single_edit_review,
    _run_single_proposer,
    run_iteration_loop,
    run_propose_phase,
    run_review_phase,
    validate_edits,
)
from multi_agent.models import (
    AgentProposal,
    AgentReviewResponse,
    ContestedEdit,
    FileEdit,
    IterationRound,
    ProposalReview,
    TokenUsage,
)


# --- validate_edits ---


class TestValidateEdits:
    def test_keeps_matching_edits(self):
        edits = [FileEdit("f.md", "hello", "hi", "", "minor")]
        result = validate_edits(edits, {"f.md": "say hello world"})
        assert len(result) == 1

    def test_drops_missing_file(self):
        edits = [FileEdit("missing.md", "hello", "hi", "", "minor")]
        result = validate_edits(edits, {"f.md": "hello"})
        assert len(result) == 0

    def test_drops_unmatched_text(self):
        edits = [FileEdit("f.md", "not here", "replacement", "", "minor")]
        result = validate_edits(edits, {"f.md": "actual content"})
        assert len(result) == 0


# --- _last_modifiers ---


class TestLastModifiers:
    def test_tracks_modify_verdicts(self):
        reviews = [
            AgentReviewResponse(
                agent_name="beta", all_approved=False,
                proposal_reviews=[
                    ProposalReview("alpha", 0, "MODIFY", "new text", "fix"),
                ],
                summary="",
            ),
        ]
        result = _last_modifiers(reviews)
        assert result == {("alpha", 0): "beta"}

    def test_later_reviewer_overrides(self):
        reviews = [
            AgentReviewResponse(
                agent_name="beta", all_approved=False,
                proposal_reviews=[
                    ProposalReview("alpha", 0, "MODIFY", "v1", ""),
                ],
                summary="",
            ),
            AgentReviewResponse(
                agent_name="gamma", all_approved=False,
                proposal_reviews=[
                    ProposalReview("alpha", 0, "MODIFY", "v2", ""),
                ],
                summary="",
            ),
        ]
        result = _last_modifiers(reviews)
        assert result[("alpha", 0)] == "gamma"

    def test_approve_not_included(self):
        reviews = [
            AgentReviewResponse(
                agent_name="beta", all_approved=True,
                proposal_reviews=[
                    ProposalReview("alpha", 0, "APPROVE", None, ""),
                ],
                summary="",
            ),
        ]
        result = _last_modifiers(reviews)
        assert len(result) == 0


# --- detect_stall ---


class TestDetectStall:
    def _round(self, approvals, total=3):
        reviews = []
        for i in range(total):
            reviews.append(AgentReviewResponse(
                agent_name=f"agent_{i}", all_approved=i < approvals,
                proposal_reviews=[], summary="",
            ))
        return IterationRound(
            round_number=0, reviews=reviews, consensus_reached=False,
            approvals=approvals,
        )

    def test_fewer_than_two_rounds(self):
        assert detect_stall([self._round(1)]) is False

    def test_same_approvals_is_stall(self):
        assert detect_stall([self._round(1), self._round(1)]) is True

    def test_fewer_approvals_is_stall(self):
        assert detect_stall([self._round(2), self._round(1)]) is True

    def test_more_approvals_not_stall(self):
        assert detect_stall([self._round(1), self._round(2)]) is False


# --- find_contested_edits ---


class TestFindContestedEdits:
    def test_fewer_than_two_rounds_empty(self):
        assert find_contested_edits([], []) == []

    def test_edit_modified_in_two_rounds(self):
        proposals = [
            AgentProposal("alpha", [FileEdit("f.md", "old", "new", "", "minor")], ""),
        ]
        rounds = [
            IterationRound(0, [
                AgentReviewResponse("beta", False, [
                    ProposalReview("alpha", 0, "MODIFY", "v1", "reason1"),
                ], ""),
            ], False),
            IterationRound(1, [
                AgentReviewResponse("beta", False, [
                    ProposalReview("alpha", 0, "MODIFY", "v2", "reason2"),
                ], ""),
            ], False),
        ]
        contested = find_contested_edits(rounds, proposals)
        assert len(contested) == 1
        assert "alpha" in contested[0].versions
        assert "beta" in contested[0].versions

    def test_edit_modified_only_once_not_contested(self):
        proposals = [
            AgentProposal("alpha", [FileEdit("f.md", "old", "new", "", "minor")], ""),
        ]
        rounds = [
            IterationRound(0, [
                AgentReviewResponse("beta", True, [], ""),
            ], False),
            IterationRound(1, [
                AgentReviewResponse("beta", False, [
                    ProposalReview("alpha", 0, "MODIFY", "v1", ""),
                ], ""),
            ], False),
        ]
        contested = find_contested_edits(rounds, proposals)
        assert len(contested) == 0

    def test_approved_agent_excluded(self):
        proposals = [
            AgentProposal("alpha", [FileEdit("f.md", "old", "new", "", "minor")], ""),
        ]
        rounds = [
            IterationRound(0, [
                AgentReviewResponse("beta", False, [
                    ProposalReview("alpha", 0, "MODIFY", "v1", "r1"),
                ], ""),
                AgentReviewResponse("gamma", False, [
                    ProposalReview("alpha", 0, "MODIFY", "v1g", "r1g"),
                ], ""),
            ], False),
            IterationRound(1, [
                AgentReviewResponse("beta", True, [], ""),  # beta approved
                AgentReviewResponse("gamma", False, [
                    ProposalReview("alpha", 0, "MODIFY", "v2g", "r2g"),
                ], ""),
            ], False),
        ]
        contested = find_contested_edits(rounds, proposals)
        # beta approved in latest round, so only gamma + alpha versions
        if contested:
            assert "beta" not in contested[0].versions


# --- build_dissent_prompt ---


class TestBuildDissentPrompt:
    def test_includes_edits(self):
        proposals = [
            AgentProposal("alpha", [FileEdit("f.md", "old", "new", "fix", "minor")], ""),
        ]
        result = build_dissent_prompt(proposals, {"f.md": "old"})
        assert "old" in result
        assert "new" in result

    def test_skips_empty_proposals(self):
        proposals = [
            AgentProposal("alpha", [], "No edits."),
            AgentProposal("beta", [FileEdit("f.md", "old", "new", "", "minor")], ""),
        ]
        result = build_dissent_prompt(proposals, {"f.md": "old"})
        assert "alpha" not in result
        assert "beta" in result


# --- build_arbitration_prompt ---


class TestBuildArbitrationPrompt:
    def test_includes_contested_edit_details(self):
        contested = ContestedEdit(
            file="f.md",
            original_text="original",
            versions={"alpha": "version_a", "beta": "version_b"},
            rationales={"alpha": "reason_a", "beta": "reason_b"},
        )
        result = build_arbitration_prompt(contested, {"f.md": "original"})
        assert "f.md" in result
        assert "original" in result
        assert "version_a" in result
        assert "version_b" in result
        assert "reason_a" in result
        assert "reason_b" in result


# --- Mocked orchestration tests ---


def _make_agent_result(output=None, error=None):
    return AgentResult(
        output=output or {},
        usage=TokenUsage(),
        duration_seconds=1.0,
        error=error,
    )


def _make_mock_backend(return_value=None, side_effect=None):
    """Create a mock AgentBackend with a mocked run_agent method."""
    backend = AsyncMock()
    backend.run_agent = AsyncMock(return_value=return_value, side_effect=side_effect)
    return backend


class TestRunSingleProposer:
    async def test_successful_run(self):
        backend = _make_mock_backend(return_value=_make_agent_result(
            output={"summary": "Changes.", "edits": [
                {"file": "f.md", "original_text": "old", "replacement_text": "new", "rationale": "fix"},
            ]},
        ))
        result = await _run_single_proposer(
            "alpha", "prompt", backend, "system prompt", "/repo", 60,
        )
        assert result.agent_name == "alpha"
        assert len(result.edits) == 1
        assert result.error is None

    async def test_error_returns_empty_proposal(self):
        backend = _make_mock_backend(return_value=_make_agent_result(error="timeout"))
        result = await _run_single_proposer(
            "alpha", "prompt", backend, "system prompt", "/repo", 60,
        )
        assert result.error == "timeout"
        assert result.edits == []


class TestRunSingleEditReview:
    async def test_approve(self):
        backend = _make_mock_backend(return_value=_make_agent_result(
            output={"verdict": "APPROVE", "rationale": "Looks good."},
        ))
        pr, raw = await _run_single_edit_review(
            "beta", "prompt", backend, "system prompt", "/repo", 60,
            original_agent="alpha", edit_index=0,
        )
        assert pr is not None
        assert pr.verdict == "APPROVE"
        assert pr.original_agent == "alpha"
        assert pr.edit_index == 0

    async def test_modify(self):
        backend = _make_mock_backend(return_value=_make_agent_result(
            output={"verdict": "MODIFY", "modified_replacement": "better",
                     "rationale": "improvement"},
        ))
        pr, raw = await _run_single_edit_review(
            "beta", "prompt", backend, "system prompt", "/repo", 60,
            original_agent="alpha", edit_index=2,
        )
        assert pr is not None
        assert pr.verdict == "MODIFY"
        assert pr.modified_replacement == "better"
        assert pr.original_agent == "alpha"
        assert pr.edit_index == 2

    async def test_error_returns_none(self):
        backend = _make_mock_backend(return_value=_make_agent_result(error="crash"))
        pr, raw = await _run_single_edit_review(
            "beta", "prompt", backend, "system prompt", "/repo", 60,
            original_agent="alpha", edit_index=0,
        )
        assert pr is None
        assert raw.error == "crash"


def _make_config(agents=None):
    """Build a ResolvedRunConfig for testing."""
    if agents is None:
        agents = {
            "alpha": AgentConfig(system_prompt="Alpha."),
            "beta": AgentConfig(system_prompt="Beta."),
        }
    config = MultiAgentConfig(general=GeneralConfig(), agents=agents)
    return resolve_run_config(config)


class TestRunProposePhase:
    async def test_calls_each_enabled_agent(self, tmp_path):
        backend = _make_mock_backend(return_value=_make_agent_result(
            output={"summary": "Done.", "edits": []},
        ))
        resolved = _make_config()
        proposals = await run_propose_phase(
            resolved, {"f.md": "content"}, {}, None, str(tmp_path), backend,
        )
        assert len(proposals) == 2
        assert backend.run_agent.call_count == 2

    async def test_skips_disabled_agents(self, tmp_path):
        backend = _make_mock_backend(return_value=_make_agent_result(
            output={"summary": "Done.", "edits": []},
        ))
        resolved = _make_config({
            "alpha": AgentConfig(system_prompt="A."),
            "beta": AgentConfig(system_prompt="B.", enabled=False),
            "gamma": AgentConfig(system_prompt="G."),
        })
        proposals = await run_propose_phase(
            resolved, {"f.md": "content"}, {}, None, str(tmp_path), backend,
        )
        assert len(proposals) == 2
        names = [p.agent_name for p in proposals]
        assert "beta" not in names


class TestRunReviewPhase:
    async def test_agents_dont_review_own_proposals(self):
        backend = _make_mock_backend(return_value=_make_agent_result(
            output={"all_approved": True, "summary": "OK", "proposal_reviews": []},
        ))
        resolved = _make_config()
        proposals = [
            AgentProposal("alpha", [FileEdit("f.md", "old", "new", "", "minor")], ""),
            AgentProposal("beta", [FileEdit("f.md", "x", "y", "", "minor")], ""),
        ]
        reviews = await run_review_phase(
            resolved, proposals, {"f.md": "old x"}, {}, "/repo", 0, backend,
        )
        assert len(reviews) == 2

    async def test_no_other_proposals_auto_approves(self):
        backend = _make_mock_backend(return_value=_make_agent_result(
            output={"all_approved": True, "summary": "OK", "proposal_reviews": []},
        ))
        resolved = _make_config()
        # Only alpha has edits; beta has none
        proposals = [
            AgentProposal("alpha", [FileEdit("f.md", "old", "new", "", "minor")], ""),
            AgentProposal("beta", [], ""),
        ]
        reviews = await run_review_phase(
            resolved, proposals, {"f.md": "old"}, {}, "/repo", 0, backend,
        )
        # Alpha reviews beta (no edits -> auto-approve), beta reviews alpha
        alpha_review = next(r for r in reviews if r.agent_name == "alpha")
        assert alpha_review.all_approved is True
        assert "No proposals" in alpha_review.summary


class TestRunIterationLoop:
    async def test_no_edits_returns_consensus(self, tmp_path):
        backend = _make_mock_backend(return_value=_make_agent_result(
            output={"summary": "All good.", "edits": []},
        ))
        resolved = _make_config()
        (tmp_path / "f.md").write_text("content")
        result = await run_iteration_loop(
            resolved, str(tmp_path), backend, target_files=["f.md"],
        )
        assert result.consensus_reached is True
        assert result.final_edits == []

    async def test_consensus_in_one_round(self, tmp_path):
        call_count = 0

        async def side_effect(agent_name, prompt, system_prompt, repo_root,
                              timeout, **kwargs):
            nonlocal call_count
            call_count += 1
            # First 2 calls are propose phase
            if call_count <= 2:
                if call_count == 1:
                    return _make_agent_result(output={
                        "summary": "Fix.",
                        "edits": [{"file": "f.md", "original_text": "old",
                                   "replacement_text": "new", "rationale": "fix"}],
                    })
                return _make_agent_result(output={"summary": "None.", "edits": []})
            # Review phase: all approve
            return _make_agent_result(output={
                "all_approved": True, "summary": "Approved.",
                "proposal_reviews": [],
            })

        backend = _make_mock_backend(side_effect=side_effect)
        resolved = _make_config()
        (tmp_path / "f.md").write_text("old content here")
        result = await run_iteration_loop(
            resolved, str(tmp_path), backend, target_files=["f.md"],
        )
        assert result.consensus_reached is True
        assert len(result.rounds) == 1
