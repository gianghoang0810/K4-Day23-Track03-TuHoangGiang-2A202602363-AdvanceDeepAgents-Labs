import json
import httpx
import pytest
import tools
from tools import RetryableError, arxiv_search, hf_daily_papers, hf_search_papers, web_fetch, web_search, with_retry


def test_retry_success_after_errors(monkeypatch):
    sleeps = []
    monkeypatch.setattr(tools.time, "sleep", lambda s: sleeps.append(s))
    calls = [0]

    def fn():
        calls[0] += 1
        if calls[0] < 3:
            raise RetryableError("temporary fail")
        return "success"

    res = with_retry(fn, attempts=5, base=1.0, cap=30.0)
    assert res == "success"
    assert calls[0] == 3
    assert len(sleeps) == 2
    assert all(s <= 30.0 for s in sleeps)


def test_retry_after_header_and_cap(monkeypatch):
    sleeps = []
    monkeypatch.setattr(tools.time, "sleep", lambda s: sleeps.append(s))
    calls = [0]

    def fn():
        calls[0] += 1
        if calls[0] == 1:
            raise RetryableError("wait 7", retry_after=7.0)
        if calls[0] == 2:
            raise RetryableError("wait 100", retry_after=100.0)
        return "ok"

    res = with_retry(fn, attempts=4, cap=30.0)
    assert res == "ok"
    assert sleeps[0] == 7.0
    assert sleeps[1] == 30.0


def test_retry_exhaustion_no_sleep_on_last(monkeypatch):
    sleeps = []
    monkeypatch.setattr(tools.time, "sleep", lambda s: sleeps.append(s))
    calls = [0]

    def fn():
        calls[0] += 1
        raise RetryableError("always fail")

    with pytest.raises(RetryableError):
        with_retry(fn, attempts=4)

    assert calls[0] == 4
    assert len(sleeps) == 3


def test_retry_non_retryable_error(monkeypatch):
    sleeps = []
    monkeypatch.setattr(tools.time, "sleep", lambda s: sleeps.append(s))

    def fn():
        raise ValueError("programming error")

    with pytest.raises(ValueError):
        with_retry(fn, attempts=4)

    assert len(sleeps) == 0


def test_retry_transport_error(monkeypatch):
    sleeps = []
    monkeypatch.setattr(tools.time, "sleep", lambda s: sleeps.append(s))
    calls = [0]

    def fn():
        calls[0] += 1
        if calls[0] == 1:
            raise httpx.ConnectError("cannot connect")
        return "connected"

    assert with_retry(fn, attempts=3) == "connected"
    assert len(sleeps) == 1


def test_arxiv_empty_query(monkeypatch):
    def fail_get(*args, **kwargs):
        raise AssertionError("Network should not be called")

    monkeypatch.setattr(tools.httpx, "get", fail_get)
    res = arxiv_search.invoke({"query": '"" : AND OR'})
    assert res == "NO RESULTS"


def test_arxiv_success_parse(monkeypatch):
    atom_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2501.00001v2</id>
        <published>2025-01-02T10:00:00Z</published>
        <title>  Title with   \n whitespace  </title>
        <summary>  Summary line 1 \n line 2  </summary>
      </entry>
    </feed>
    """
    called_url = []

    def fake_get(url, **kwargs):
        called_url.append(url)
        return httpx.Response(200, text=atom_xml, request=httpx.Request("GET", url))

    monkeypatch.setattr(tools.httpx, "get", fake_get)
    res = arxiv_search.invoke({"query": "world model", "max_results": 2})
    assert called_url[0].startswith("https://")
    data = json.loads(res)
    assert len(data) == 1
    assert data[0]["id"] == "2501.00001"
    assert data[0]["url"] == "https://arxiv.org/abs/2501.00001"
    assert data[0]["published"] == "2025-01-02"
    assert data[0]["title"] == "Title with whitespace"
    assert data[0]["summary"] == "Summary line 1 line 2"


def test_arxiv_empty_feed(monkeypatch):
    empty_feed = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"></feed>"""
    monkeypatch.setattr(tools.httpx, "get", lambda url, **kwargs: httpx.Response(200, text=empty_feed, request=httpx.Request("GET", url)))
    res = arxiv_search.invoke({"query": "something"})
    assert res == "NO RESULTS"


def test_arxiv_retry_429(monkeypatch):
    sleeps = []
    monkeypatch.setattr(tools.time, "sleep", lambda s: sleeps.append(s))
    calls = [0]
    atom_xml = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2501.00001v1</id><title>T</title></entry></feed>"""

    def fake_get(url, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, request=httpx.Request("GET", url))
        return httpx.Response(200, text=atom_xml, request=httpx.Request("GET", url))

    monkeypatch.setattr(tools.httpx, "get", fake_get)
    res = arxiv_search.invoke({"query": "test"})
    assert "2501.00001" in res
    assert 2.0 in sleeps


def test_arxiv_always_503(monkeypatch):
    monkeypatch.setattr(tools.time, "sleep", lambda s: None)
    monkeypatch.setattr(tools.httpx, "get", lambda url, **kwargs: httpx.Response(503, request=httpx.Request("GET", url)))
    res = arxiv_search.invoke({"query": "test"})
    assert res.startswith("ERROR")


def test_hf_daily_papers(monkeypatch):
    items = [
        {"paper": {"id": "p1", "title": "Alpha Model", "summary": "About world models", "upvotes": 10, "publishedAt": "2025-01-01"}},
        {"paper": None, "title": "Missing paper id"},
        {"paper": {"id": "p2", "title": "Beta Vision", "summary": "Vision stuff", "upvotes": 50, "publishedAt": "2025-01-02"}},
    ]
    monkeypatch.setattr(tools.httpx, "get", lambda url, **kwargs: httpx.Response(200, json=items, request=httpx.Request("GET", url)))

    # Test keyword filter and sorting by upvotes
    res = hf_daily_papers.invoke({"keyword": "world"})
    data = json.loads(res)
    assert len(data) == 1
    assert data[0]["id"] == "p1"
    assert data[0]["url"] == "https://huggingface.co/papers/p1"

    # Test all without keyword -> sorted desc by upvotes
    res_all = hf_daily_papers.invoke({})
    data_all = json.loads(res_all)
    assert len(data_all) == 2
    assert data_all[0]["id"] == "p2"  # 50 upvotes first
    assert data_all[1]["id"] == "p1"  # 10 upvotes second


def test_hf_search_papers(monkeypatch):
    items = [
        {"paper": {"id": "p1", "title": "Paper 1", "summary": "long summary", "ai_summary": "short ai summary", "upvotes": 5}},
    ]
    monkeypatch.setattr(tools.httpx, "get", lambda url, **kwargs: httpx.Response(200, json=items, request=httpx.Request("GET", url)))

    res = hf_search_papers.invoke({"query": "test"})
    data = json.loads(res)
    assert len(data) == 1
    assert data[0]["summary"] == "short ai summary"

    # Empty query
    assert hf_search_papers.invoke({"query": ""}) == "NO RESULTS"

    # Empty response
    monkeypatch.setattr(tools.httpx, "get", lambda url, **kwargs: httpx.Response(200, json=[], request=httpx.Request("GET", url)))
    assert hf_search_papers.invoke({"query": "empty"}) == "NO RESULTS"


def test_exa_sse_and_search(monkeypatch):
    sse_text = "event: message\ndata: " + json.dumps({
        "result": {
            "content": [{"type": "text", "text": "Found some search results here https://example.com"}]
        }
    }) + "\n\n"
    monkeypatch.setattr(tools.httpx, "post", lambda url, **kwargs: httpx.Response(200, text=sse_text, request=httpx.Request("POST", url)))

    res = web_search.invoke({"query": "world models"})
    assert "Found some search results" in res


def test_exa_rate_limit_and_retry(monkeypatch):
    sleeps = []
    monkeypatch.setattr(tools.time, "sleep", lambda s: sleeps.append(s))
    calls = [0]

    def fake_post(url, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            # 429 rate limit
            return httpx.Response(429, headers={"Retry-After": "1"}, request=httpx.Request("POST", url))
        return httpx.Response(200, json={"result": {"content": [{"type": "text", "text": "Ok text"}]}}, request=httpx.Request("POST", url))

    monkeypatch.setattr(tools.httpx, "post", fake_post)
    res = web_search.invoke({"query": "world models"})
    assert res == "Ok text"
    assert len(sleeps) >= 1


def test_exa_rate_limit_as_jsonrpc_error_is_retried(monkeypatch):
    sleeps = []
    monkeypatch.setattr(tools.time, "sleep", lambda s: sleeps.append(s))
    calls = [0]
    limited = {"jsonrpc": "2.0", "error": {"code": -32000, "message": "You've hit Exa's free MCP rate limit."}}

    def fake_post(url, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            return httpx.Response(200, text="data: " + json.dumps(limited) + "\n\n", request=httpx.Request("POST", url))
        return httpx.Response(200, json={"result": {"content": [{"type": "text", "text": "Ok text"}]}}, request=httpx.Request("POST", url))

    monkeypatch.setattr(tools.httpx, "post", fake_post)
    assert web_search.invoke({"query": "world models"}) == "Ok text"
    assert calls[0] == 2 and len(sleeps) == 1


def test_exa_long_retry_after_fails_fast(monkeypatch):
    sleeps = []
    monkeypatch.setattr(tools.time, "sleep", lambda s: sleeps.append(s))
    calls = [0]

    def fake_post(url, **kwargs):
        calls[0] += 1
        return httpx.Response(429, headers={"Retry-After": "72534"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(tools.httpx, "post", fake_post)
    res = web_search.invoke({"query": "world models"})
    assert res.startswith("ERROR") and "72534" in res
    assert calls[0] == 1 and sleeps == []


def test_exa_error_and_key_redaction(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "FAKEKEY123")

    def fake_post(url, **kwargs):
        raise httpx.ConnectError("boom https://mcp.exa.ai/mcp?exaApiKey=FAKEKEY123")

    monkeypatch.setattr(tools.httpx, "post", fake_post)
    monkeypatch.setattr(tools.time, "sleep", lambda s: None)

    res = web_search.invoke({"query": "world models"})
    assert res.startswith("ERROR")
    assert "FAKEKEY123" not in res
    assert "***" in res


def test_web_fetch_truncation(monkeypatch):
    long_text = "a" * 20000
    monkeypatch.setattr(tools.httpx, "post", lambda url, **kwargs: httpx.Response(200, json={"result": {"content": [{"type": "text", "text": long_text}]}}, request=httpx.Request("POST", url)))

    res = web_fetch.invoke({"url": "https://example.com"})
    assert len(res) <= 12050
    assert res.endswith("[truncated]")

    # Invalid URL
    res_bad = web_fetch.invoke({"url": "ftp://example.com"})
    assert res_bad.startswith("ERROR")


def test_invalid_parameters_do_not_crash(monkeypatch):
    monkeypatch.setattr(tools.httpx, "get", lambda url, **kwargs: httpx.Response(200, text="<feed xmlns='http://www.w3.org/2005/Atom'></feed>", request=httpx.Request("GET", url)))
    res1 = arxiv_search.func(query="test", max_results="invalid")
    assert isinstance(res1, str)
    res2 = arxiv_search.invoke({"query": "test", "max_results": -5})
    assert isinstance(res2, str)
    res3 = hf_daily_papers.func(limit="abc", date=None, keyword=None)
    assert isinstance(res3, str)
