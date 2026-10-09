import pytest
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolCallLimitMiddleware
from agents import (
    CHECKER_PROMPT,
    FINALIZER_PATH,
    LEAD_PROMPT,
    REPORT_PATH,
    RESEARCHER_PROMPT,
    SOURCES_PATH,
    VALIDATOR_PATH,
    build_subagents,
)
from tools import SOURCE_TOOLS


def test_build_subagents_specs():
    specs = build_subagents()
    assert len(specs) == 2
    names = {s["name"] for s in specs}
    assert names == {"researcher", "citation-checker"}

    by_name = {s["name"]: s for s in specs}
    researcher = by_name["researcher"]
    checker = by_name["citation-checker"]

    # Check tools
    researcher_tools = {t.name for t in researcher["tools"]}
    expected_tools = {t.name for t in SOURCE_TOOLS}
    assert researcher_tools == expected_tools

    checker_tools = [t.name for t in checker["tools"]]
    assert checker_tools == ["web_fetch"]

    # Check middleware
    for spec in (researcher, checker):
        mw = spec.get("middleware", [])
        types = [type(m) for m in mw]
        assert ModelCallLimitMiddleware in types
        assert ToolCallLimitMiddleware in types


def test_prompts_content():
    assert "TODO" not in LEAD_PROMPT
    assert REPORT_PATH in LEAD_PROMPT
    assert SOURCES_PATH in LEAD_PROMPT
    assert FINALIZER_PATH in LEAD_PROMPT
    assert VALIDATOR_PATH in LEAD_PROMPT
    assert "write_todos" in LEAD_PROMPT
    assert "citation-checker" in LEAD_PROMPT

    assert "untrusted" in RESEARCHER_PROMPT.lower()
    assert "## [S1]" in RESEARCHER_PROMPT
    assert "untrusted" in CHECKER_PROMPT.lower()
