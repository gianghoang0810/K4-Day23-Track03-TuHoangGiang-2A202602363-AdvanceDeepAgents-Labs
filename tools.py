"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET

from dotenv import load_dotenv
import httpx
from langchain_core.tools import tool

load_dotenv()

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


RETRY_STATUS = {429, 500, 502, 503, 504}


def _retry_after(response):
    val = response.headers.get("Retry-After")
    if val:
        try:
            return float(val)
        except ValueError:
            return None
    return None


def _check_status(response, label="HTTP"):
    if response.status_code in RETRY_STATUS:
        retry_after = _retry_after(response)
        raise RetryableError(f"{label}: HTTP {response.status_code}", retry_after=retry_after)
    response.raise_for_status()


def _clamp(val, lo, hi, default):
    try:
        v = int(val)
    except (TypeError, ValueError):
        return default
    return max(lo, min(v, hi))


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError or httpx.TransportError, wait and call it again."""
    for attempt in range(attempts):
        try:
            return fn()
        except (RetryableError, httpx.TransportError) as exc:
            if attempt == attempts - 1:
                raise
            retry_after = getattr(exc, "retry_after", None)
            if retry_after is not None:
                delay = float(retry_after)
            else:
                delay = base * (2 ** attempt) + random.uniform(0, base)
            time.sleep(max(0.0, min(delay, cap)))


# ---- TODO 2: arXiv ----
_ARXIV_LOCK = threading.Lock()
_ARXIV_GAP = 3.0
_arxiv_last = [0.0]


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Use 2-5 keywords. Returns a JSON list of {id, url, published, title, summary}."""
    try:
        q = re.sub(r"\b(?:all|ti|abs|au|cat|co|jr|rn|id):", " ", str(query))
        terms = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", q) if t.upper() not in {"AND", "OR", "NOT", "ANDNOT"}][:8]
        if not terms:
            return "NO RESULTS"
        search_query = " AND ".join(f"all:{t}" for t in terms)
        max_res = _clamp(max_results, 1, 30, 10)

        def fn():
            with _ARXIV_LOCK:
                wait = _ARXIV_GAP - (time.monotonic() - _arxiv_last[0])
                if wait > 0:
                    time.sleep(wait)
                try:
                    resp = httpx.get(
                        ARXIV_URL,
                        params={
                            "search_query": search_query,
                            "sortBy": "submittedDate",
                            "sortOrder": "descending",
                            "max_results": max_res,
                        },
                        timeout=60,
                    )
                finally:
                    _arxiv_last[0] = time.monotonic()
            _check_status(resp, "arXiv")
            return resp.text

        text = with_retry(fn, attempts=6, base=3.0, cap=60.0)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(text)
        records = []
        for entry in root.findall("a:entry", ns):
            raw_id = entry.findtext("a:id", default="", namespaces=ns) or ""
            if "/abs/" not in raw_id:
                continue
            paper_id = re.sub(r"v\d+$", "", raw_id.rsplit("/abs/", 1)[1])
            url = f"https://arxiv.org/abs/{paper_id}"
            published = (entry.findtext("a:published", default="", namespaces=ns) or "")[:10]
            title = " ".join((entry.findtext("a:title", default="", namespaces=ns) or "").split())
            summary = " ".join((entry.findtext("a:summary", default="", namespaces=ns) or "").split())[:600]
            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
            })
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 3: Hugging Face ----
def _hf_record(item, prefer_ai_summary=False):
    p = item.get("paper") or {}
    if not p.get("id"):
        return None
    summary = (p.get("ai_summary") if prefer_ai_summary else None) or p.get("summary") or item.get("summary") or ""
    summary = " ".join(str(summary).split())[:600]
    title = " ".join(str(p.get("title") or item.get("title") or "").split())
    published = str(p.get("publishedAt") or item.get("publishedAt") or "")[:10]
    return {
        "id": p["id"],
        "url": f"https://huggingface.co/papers/{p['id']}",
        "published": published,
        "title": title,
        "summary": summary,
        "upvotes": p.get("upvotes") or 0,
        "github": p.get("githubRepo"),
        "stars": p.get("githubStars"),
    }


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        params = {"limit": _clamp(limit, 1, 100, 30)}
        if date and str(date).strip():
            params["date"] = str(date).strip()

        def fn():
            resp = httpx.get(HF_DAILY_URL, params=params, timeout=30)
            _check_status(resp, "Hugging Face daily")
            return resp.json()

        items = with_retry(fn, attempts=5, base=1.0, cap=30.0)
        if not isinstance(items, list):
            return "NO RESULTS"

        records = []
        kw = str(keyword).strip().lower() if keyword else ""
        for item in items:
            rec = _hf_record(item, prefer_ai_summary=False)
            if not rec:
                continue
            if kw and kw not in (rec["title"] + " " + rec["summary"]).lower():
                continue
            records.append(rec)

        records.sort(key=lambda r: r.get("upvotes") or 0, reverse=True)
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        q = str(query).strip() if query else ""
        if not q:
            return "NO RESULTS"
        lim = _clamp(limit, 1, 50, 10)
        params = {"q": q, "limit": lim}

        def fn():
            resp = httpx.get(HF_SEARCH_URL, params=params, timeout=30)
            _check_status(resp, "Hugging Face search")
            return resp.json()

        items = with_retry(fn, attempts=5, base=1.0, cap=30.0)
        if not isinstance(items, list):
            return "NO RESULTS"

        records = []
        for item in items:
            rec = _hf_record(item, prefer_ai_summary=True)
            if rec:
                records.append(rec)
        records = records[:lim]
        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _redact(text: str) -> str:
    key = os.getenv("EXA_API_KEY", "").strip()
    out = text
    if key:
        out = out.replace(key, "***")
    out = re.sub(r"(exaApiKey=)[^&\s'\"]+", r"\1***", out)
    return out


def _has_rate_flag(d):
    if isinstance(d, dict):
        for k, v in d.items():
            if k == "ai.exa/usage":
                continue
            if "rate" in k.lower() and v:
                return True
            if _has_rate_flag(v):
                return True
    elif isinstance(d, list):
        for item in d:
            if _has_rate_flag(item):
                return True
    return False


def _exa_rate_limited(result, text):
    meta = result.get("_meta") or {}
    if _has_rate_flag(meta):
        return True
    if len(text) < 600 and "rate limit" in text.lower():
        return True
    return False


EXA_MAX_WAIT = 300.0  # a Retry-After longer than this (free tier: ~20 h) cannot be waited out: fail fast


def _exa_rate_limit_error(message, retry_after=None):
    """Exa signals its rate limit in several ways (HTTP 200 + _meta flag, HTTP 429, JSON-RPC error). A short wait is
    retried; a wait longer than EXA_MAX_WAIT is reported at once so the agent switches source instead of stalling."""
    if retry_after is not None and retry_after > EXA_MAX_WAIT:
        return RuntimeError(f"{message}; retry after {retry_after:.0f}s, set EXA_API_KEY to avoid the free-tier limit")
    return RetryableError(message, retry_after=retry_after)


def _exa_call(tool_name: str, arguments: dict) -> str:
    key = os.getenv("EXA_API_KEY", "").strip()
    params = {"exaApiKey": key} if key else None
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    def fn():
        resp = httpx.post(EXA_URL, params=params, json=body, headers=headers, timeout=60)
        if resp.status_code == 429:
            raise _exa_rate_limit_error("Exa rate limited (HTTP 429)", _retry_after(resp))
        _check_status(resp, "Exa")
        data = None
        for line in resp.text.splitlines():
            if line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if data is None:
            data = resp.json()
        if "error" in data:
            if "rate limit" in str(data["error"]).lower():
                raise _exa_rate_limit_error("Exa rate limited (JSON-RPC error)", _retry_after(resp))
            raise RuntimeError(f"Exa JSON-RPC error: {data['error']}")
        result = data.get("result") or {}
        text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        if _exa_rate_limited(result, text):
            raise RetryableError("Exa rate limited (HTTP 200 + _meta flag)")
        if result.get("isError"):
            raise RuntimeError(text[:300])
        return text

    return with_retry(fn, attempts=8, base=2.0, cap=60.0)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        q = str(query).strip() if query else ""
        if not q:
            return "NO RESULTS"
        obj = str(objective).strip() if objective else ""
        if not obj:
            obj = f"Find pages that best answer: {q}"
        num = _clamp(num_results, 1, 10, 5)
        text = _exa_call("web_search_exa", {"query": q, "objective": obj, "numResults": num})
        if not text or not text.strip():
            return "NO RESULTS"
        return _redact(text[:10000])
    except Exception as exc:
        return _redact(f"ERROR: {type(exc).__name__}: {exc}")


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        u = str(url).strip() if url else ""
        if not (u.startswith("http://") or u.startswith("https://")):
            return "ERROR: url must start with http:// or https://"
        text = _exa_call("web_fetch_exa", {"urls": [u]})
        if not text or not text.strip():
            return "NO RESULTS"
        if len(text) > 12000:
            return _redact(text[:12000] + "\n[truncated]")
        return _redact(text)
    except Exception as exc:
        return _redact(f"ERROR: {type(exc).__name__}: {exc}")


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n", flush=True)
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n", flush=True)
