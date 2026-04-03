"""Tests for dataclasses, JSON parsing, and edit/review parsing.

count_approvals and sanitize_edit_path are already tested in test_merge.py.
"""

from multi_agent.models import (
    AgentProposal,
    AgentReviewResponse,
    FileEdit,
    ProposalReview,
    TokenUsage,
    count_blocking_approvals,
    extract_json,
    extract_usage,
    filter_edits_by_severity,
    is_blocking_severity,
    parse_edits,
    parse_proposal_reviews,
    severity_index,
    unwrap_result,
)


# --- extract_usage ---


class TestExtractUsage:
    def test_well_formed(self):
        data = {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_read_input_tokens": 10,
                "cache_creation_input_tokens": 5,
            },
            "total_cost_usd": 0.42,
        }
        usage = extract_usage(data)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_read_input_tokens == 10
        assert usage.cache_creation_input_tokens == 5
        assert usage.cost_usd == 0.42

    def test_missing_usage_key(self):
        usage = extract_usage({})
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cost_usd == 0.0

    def test_partial_fields(self):
        usage = extract_usage({"usage": {"input_tokens": 99}})
        assert usage.input_tokens == 99
        assert usage.output_tokens == 0


# --- extract_json ---


class TestExtractJson:
    def test_direct_json(self):
        result = extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_code_fence(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_plain_code_fence(self):
        text = 'Before\n```\n{"key": "value"}\n```\nAfter'
        result = extract_json(text)
        assert result == {"key": "value"}

    def test_embedded_braces(self):
        text = 'The result is {"answer": 42} as expected.'
        result = extract_json(text)
        assert result == {"answer": 42}

    def test_non_json_returns_none(self):
        assert extract_json("no json here") is None

    def test_nested_objects(self):
        text = '{"outer": {"inner": "value"}}'
        result = extract_json(text)
        assert result == {"outer": {"inner": "value"}}

    def test_escaped_quotes_in_strings(self):
        text = '{"msg": "he said \\"hello\\""}'
        result = extract_json(text)
        assert result == {"msg": 'he said "hello"'}


# --- unwrap_result ---


class TestUnwrapResult:
    def test_none_input(self):
        assert unwrap_result(None) is None

    def test_missing_result_key(self):
        assert unwrap_result({"other": "data"}) is None

    def test_dict_result(self):
        result = unwrap_result({"result": {"summary": "done"}})
        assert result == {"summary": "done"}

    def test_json_string_result(self):
        result = unwrap_result({"result": '{"summary": "done"}'})
        assert result == {"summary": "done"}

    def test_non_dict_non_string_result(self):
        assert unwrap_result({"result": 42}) is None


# --- parse_edits ---


class TestParseEdits:
    def test_well_formed(self):
        raw = [{"file": "foo.md", "original_text": "old", "replacement_text": "new",
                "rationale": "fix", "severity": "major"}]
        edits = parse_edits(raw)
        assert len(edits) == 1
        assert edits[0].file == "foo.md"
        assert edits[0].original_text == "old"
        assert edits[0].replacement_text == "new"
        assert edits[0].severity == "major"

    def test_invalid_paths_dropped(self):
        raw = [
            {"file": "/etc/passwd", "original_text": "a", "replacement_text": "b", "rationale": ""},
            {"file": "../escape.md", "original_text": "a", "replacement_text": "b", "rationale": ""},
            {"file": "valid.md", "original_text": "a", "replacement_text": "b", "rationale": ""},
        ]
        edits = parse_edits(raw)
        assert len(edits) == 1
        assert edits[0].file == "valid.md"

    def test_missing_fields_default(self):
        raw = [{"file": "f.md"}]
        edits = parse_edits(raw)
        assert len(edits) == 1
        assert edits[0].original_text == ""
        assert edits[0].replacement_text == ""
        assert edits[0].rationale == ""
        assert edits[0].severity == "minor"

    def test_empty_list(self):
        assert parse_edits([]) == []

    def test_severity_defaults_to_minor(self):
        raw = [{"file": "f.md", "original_text": "a", "replacement_text": "b", "rationale": ""}]
        edits = parse_edits(raw)
        assert edits[0].severity == "minor"

    def test_invalid_severity_clamped(self):
        raw = [{"file": "f.md", "original_text": "a", "replacement_text": "b",
                "rationale": "", "severity": "extreme"}]
        edits = parse_edits(raw)
        assert edits[0].severity == "minor"


# --- parse_proposal_reviews ---


class TestParseProposalReviews:
    def test_well_formed(self):
        raw = [{
            "original_agent": "alpha",
            "edit_index": 1,
            "verdict": "MODIFY",
            "modified_replacement": "new text",
            "rationale": "improvement",
        }]
        reviews = parse_proposal_reviews(raw)
        assert len(reviews) == 1
        assert reviews[0].original_agent == "alpha"
        assert reviews[0].edit_index == 1
        assert reviews[0].verdict == "MODIFY"
        assert reviews[0].modified_replacement == "new text"

    def test_normalizer_applied(self):
        raw = [{"original_agent": "Alpha", "edit_index": 0, "verdict": "APPROVE", "rationale": ""}]
        reviews = parse_proposal_reviews(raw, normalizer=str.lower)
        assert reviews[0].original_agent == "alpha"

    def test_defaults(self):
        raw = [{}]
        reviews = parse_proposal_reviews(raw)
        assert reviews[0].original_agent == ""
        assert reviews[0].edit_index == 0
        assert reviews[0].verdict == "APPROVE"
        assert reviews[0].modified_replacement is None
        assert reviews[0].rationale == ""


# --- severity helpers ---


class TestSeverityIndex:
    def test_known_values(self):
        assert severity_index("critical") == 0
        assert severity_index("major") == 1
        assert severity_index("minor") == 2
        assert severity_index("suggestion") == 3

    def test_unknown_defaults_to_minor(self):
        assert severity_index("extreme") == 2


class TestFilterEditsBySeverity:
    def test_filters_below_threshold(self):
        edits = [
            FileEdit("f.md", "a", "b", "", "critical"),
            FileEdit("f.md", "c", "d", "", "minor"),
            FileEdit("f.md", "e", "f", "", "suggestion"),
        ]
        result = filter_edits_by_severity(edits, "major")
        assert len(result) == 1
        assert result[0].severity == "critical"

    def test_suggestion_keeps_all(self):
        edits = [
            FileEdit("f.md", "a", "b", "", "minor"),
            FileEdit("f.md", "c", "d", "", "suggestion"),
        ]
        result = filter_edits_by_severity(edits, "suggestion")
        assert len(result) == 2


class TestIsBlockingSeverity:
    def test_critical_always_blocking(self):
        assert is_blocking_severity("critical", "suggestion") is True

    def test_suggestion_not_blocking_at_major(self):
        assert is_blocking_severity("suggestion", "major") is False

    def test_equal_to_threshold(self):
        assert is_blocking_severity("major", "major") is True

    def test_minor_not_blocking_at_major(self):
        assert is_blocking_severity("minor", "major") is False


class TestCountBlockingApprovals:
    def _edit(self, severity="major"):
        return FileEdit("f.md", "old", "new", "fix", severity)

    def test_all_approved(self):
        reviews = [
            AgentReviewResponse("beta", True, [], ""),
        ]
        proposals = [AgentProposal("alpha", [self._edit()], "")]
        assert count_blocking_approvals(reviews, proposals, "major") == 1

    def test_modify_blocking_edit_not_counted(self):
        reviews = [
            AgentReviewResponse("beta", False, [
                ProposalReview("alpha", 0, "MODIFY", "new", "fix"),
            ], ""),
        ]
        proposals = [AgentProposal("alpha", [self._edit("critical")], "")]
        assert count_blocking_approvals(reviews, proposals, "major") == 0

    def test_modify_non_blocking_edit_counted(self):
        reviews = [
            AgentReviewResponse("beta", False, [
                ProposalReview("alpha", 0, "MODIFY", "new", "fix"),
            ], ""),
        ]
        proposals = [AgentProposal("alpha", [self._edit("suggestion")], "")]
        assert count_blocking_approvals(reviews, proposals, "major") == 1

    def test_error_not_counted(self):
        reviews = [
            AgentReviewResponse("beta", True, [], "", error="crash"),
        ]
        proposals = [AgentProposal("alpha", [self._edit()], "")]
        assert count_blocking_approvals(reviews, proposals, "major") == 0

    def test_mixed_blocking_and_non_blocking(self):
        reviews = [
            AgentReviewResponse("beta", False, [
                ProposalReview("alpha", 0, "MODIFY", "x", ""),
                ProposalReview("alpha", 1, "MODIFY", "y", ""),
            ], ""),
        ]
        proposals = [AgentProposal("alpha", [
            self._edit("critical"),
            self._edit("suggestion"),
        ], "")]
        # One MODIFY targets blocking, so not counted
        assert count_blocking_approvals(reviews, proposals, "major") == 0
