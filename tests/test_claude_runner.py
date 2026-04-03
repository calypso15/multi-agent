"""Tests for claude_runner pure helpers."""

from multi_agent.claude_runner import (
    _make_drafting_callback,
    _make_tool_callback,
    _summarize_tool_call,
)


# --- _summarize_tool_call ---


class TestSummarizeToolCall:
    def test_read(self):
        result = _summarize_tool_call("Read", {"file_path": "/repo/src/main.py"})
        assert "main.py" in result

    def test_read_with_offset(self):
        result = _summarize_tool_call("Read", {"file_path": "f.py", "offset": 10, "limit": 20})
        assert "offset=10" in result
        assert "limit=20" in result

    def test_grep(self):
        result = _summarize_tool_call("Grep", {"pattern": "TODO"})
        assert "TODO" in result

    def test_glob(self):
        result = _summarize_tool_call("Glob", {"pattern": "*.py"})
        assert "*.py" in result

    def test_web_search(self):
        result = _summarize_tool_call("WebSearch", {"query": "python async"})
        assert "python async" in result

    def test_web_fetch(self):
        result = _summarize_tool_call("WebFetch", {"url": "https://example.com/page"})
        assert "example.com" in result

    def test_fallback(self):
        result = _summarize_tool_call("CustomTool", {"data": "some value"})
        assert "some value" in result

    def test_empty_input(self):
        result = _summarize_tool_call("CustomTool", {})
        assert result == ""


# --- _make_tool_callback ---


class TestMakeToolCallback:
    def test_none_progress_returns_none(self):
        assert _make_tool_callback("alpha", None) is None

    def test_returns_callable(self):
        calls = []
        cb = _make_tool_callback("alpha", lambda name, msg: calls.append((name, msg)))
        assert callable(cb)
        cb("Read", "main.py")
        assert len(calls) == 1
        assert calls[0][0] == "alpha"
        assert "Read" in calls[0][1]
        assert "main.py" in calls[0][1]


# --- _make_drafting_callback ---


class TestMakeDraftingCallback:
    def test_none_progress_returns_none(self):
        assert _make_drafting_callback("alpha", None) is None

    def test_reports_on_first_delta(self):
        calls = []
        cb = _make_drafting_callback("alpha", lambda name, msg: calls.append((name, msg)))
        cb(10)  # first text delta
        assert len(calls) == 1
        assert "drafting" in calls[0][1]

    def test_does_not_repeat(self):
        calls = []
        cb = _make_drafting_callback("alpha", lambda name, msg: calls.append((name, msg)))
        cb(10)
        cb(20)
        cb(30)
        assert len(calls) == 1

    def test_resets_on_new_block(self):
        calls = []
        cb = _make_drafting_callback("alpha", lambda name, msg: calls.append((name, msg)))
        cb(10)  # first block
        cb(0)   # content_block_start — new turn
        cb(5)   # second block
        assert len(calls) == 2
