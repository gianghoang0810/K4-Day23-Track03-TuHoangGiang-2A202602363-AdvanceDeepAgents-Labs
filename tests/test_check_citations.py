import json
import pytest
from check_citations import check
from finalize_citations import finalize


def test_valid_report_and_sources():
    raw_body = "This is a body discussing topic [1] and another aspect [2].\n"
    raw_sources = [
        {"n": 1, "url": "https://arxiv.org/abs/2401.00001", "title": "Paper 1", "source": "arxiv", "date": "2024-01-01"},
        {"n": 2, "url": "https://huggingface.co/papers/2402.00002", "title": "Paper 2", "source": "hf-search", "date": "2024-02-02"},
    ]
    report, new_sources, problems = finalize(raw_body, raw_sources)
    assert not problems
    assert check(report, new_sources) == []


def test_empty_sources():
    assert check("Some body [1].\n\n## References\n[1] Title. web. https://example.com (2025)", []) == ["no sources in sources.json"]
    assert check("Some body", None) == ["no sources in sources.json"]


def test_invalid_source_schema():
    body = "Body [1].\n\n## References\n[1] Title. web. https://example.com (2025)\n"
    # n is string
    probs1 = check(body, [{"n": "1", "url": "https://example.com"}])
    assert any("n is not an integer" in p for p in probs1)

    # n is bool
    probs2 = check(body, [{"n": True, "url": "https://example.com"}])
    assert any("n is not an integer" in p for p in probs2)

    # invalid url scheme
    probs3 = check(body, [{"n": 1, "url": "ftp://example.com"}])
    assert any("url must start with http" in p for p in probs3)

    # duplicate url
    probs4 = check(body, [{"n": 1, "url": "https://example.com"}, {"n": 2, "url": "https://example.com"}])
    assert any("duplicate url" in p for p in probs4)


def test_missing_references_heading():
    body = "Body citing [1] without references heading."
    probs = check(body, [{"n": 1, "url": "https://example.com"}])
    assert any("missing '## References' heading" in p for p in probs)


def test_citation_mismatches():
    # [5] cited but missing from sources
    body1 = "Body citing [5].\n\n## References\n[1] Title. web. https://example.com (2025)\n"
    probs1 = check(body1, [{"n": 1, "url": "https://example.com"}])
    assert any("[5] cited but missing from sources.json" in p for p in probs1)
    assert any("source [1] never cited" in p for p in probs1)


def test_reference_line_problems():
    # Reference line has no source
    body1 = "Body [1].\n\n## References\n[1] T1. web. https://a.com (2025)\n[2] T2. web. https://b.com (2025)\n"
    probs1 = check(body1, [{"n": 1, "url": "https://a.com"}])
    assert any("reference line [2] is not a source" in p for p in probs1)

    # Source has no reference line
    body2 = "Body [1][2].\n\n## References\n[1] T1. web. https://a.com (2025)\n"
    probs2 = check(body2, [{"n": 1, "url": "https://a.com"}, {"n": 2, "url": "https://b.com"}])
    assert any("source [2] has no reference line" in p for p in probs2)

    # Duplicate reference line
    body3 = "Body [1].\n\n## References\n[1] T1. web. https://a.com (2025)\n[1] T1. web. https://a.com (2025)\n"
    probs3 = check(body3, [{"n": 1, "url": "https://a.com"}])
    assert any("reference [1] listed more than once" in p for p in probs3)

    # Line has 2 URLs
    body4 = "Body [1].\n\n## References\n[1] T1. web. https://a.com and https://b.com (2025)\n"
    probs4 = check(body4, [{"n": 1, "url": "https://a.com"}])
    assert any("must contain exactly one URL" in p for p in probs4)

    # URL does not match sources.json
    body5 = "Body [1].\n\n## References\n[1] T1. web. https://different.com (2025)\n"
    probs5 = check(body5, [{"n": 1, "url": "https://a.com"}])
    assert any("url does not match sources.json" in p for p in probs5)


def test_accepted_cases():
    # Grouped citations [1, 2] and [1-3]
    body = (
        "Discussing [1, 2] and range [2-4].\n"
        "Here is code: `[9]` and:\n"
        "```\n[10] in code block\n```\n"
        "And a markdown link [11](https://link.com).\n\n"
        "## References\n"
        "[1] T1. web. https://a.com (2025)\n"
        "[2] T2. web. https://b.com (2025)\n"
        "[3] T3. web. https://c.com (2025)\n"
        "[4] T4. web. https://d.com (2025)\n"
    )
    sources = [
        {"n": 1, "url": "https://a.com"},
        {"n": 2, "url": "https://b.com"},
        {"n": 3, "url": "https://c.com"},
        {"n": 4, "url": "https://d.com"},
    ]
    probs = check(body, sources)
    assert probs == []
