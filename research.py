"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    s = re.sub(r"\W+", "-", str(topic).lower()).strip("-")[:60].strip("-")
    return s or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    today = time.strftime("%Y-%m-%d")
    return (
        f"Topic: {topic}\n"
        f"Today's date: {today}\n\n"
        "Please conduct a comprehensive deep research survey on this topic and deliver a fully cited survey report.\n"
        "Follow your required workflow in order:\n"
        "1. PLAN: Call write_todos to outline your steps and split the topic into 3-5 sub-questions.\n"
        "2. DELEGATE IN PARALLEL: Delegate each sub-question to a `researcher` subagent with full context (topic, sub-question, notes path, source families, format).\n"
        "3. CHECK RESULTS: read_file every notes file before using it. If empty or insufficient sources, redelegate.\n"
        "4. MERGE SOURCES: Combine sources into sources.json with fields (n, id, url, title, date, source). Ensure at least 3 source families are covered.\n"
        "5. WRITE REPORT BODY: Write report.md following the required headings: # <Title>, ## TL;DR, ## Background, 3-6 theme headings, and ## Trends and open problems (exact casing: lowercase 'open problems', never 'Open Problems'). Do NOT write ## References.\n"
        "6. FINALIZE: Execute finalize_citations.py in the sandbox to renumber citations and generate ## References.\n"
        "7. VALIDATE: Execute check_citations.py in the sandbox and resolve any problems until it prints OK.\n"
        "8. SPOT-CHECK: Send 3 key claims and source URLs to citation-checker to verify support.\n"
        "Write the report in English."
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}."""
    counts = Counter()
    input_tokens = 0
    output_tokens = 0

    for m in messages:
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls is None and isinstance(m, dict):
            tool_calls = m.get("tool_calls")
        if tool_calls:
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if name:
                    counts[name] += 1

        usage = getattr(m, "usage_metadata", None)
        if usage is None and isinstance(m, dict):
            usage = m.get("usage_metadata")
        if isinstance(usage, dict):
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)

    return {
        "model": model_name,
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": counts.get("task", 0),
        "tool_calls": dict(counts),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
        },
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report = files.get(REPORT_PATH)
    raw = files.get(SOURCES_PATH)

    if report is None or not report.strip():
        raise RuntimeError("report.md missing or empty in the sandbox")
    if raw is None or not raw.strip():
        raise RuntimeError("sources.json missing or empty in the sandbox")

    try:
        sources = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"sources.json is not valid JSON: {exc}")

    if not isinstance(sources, list) or len(sources) == 0:
        raise RuntimeError("sources.json must be a non-empty JSON list")

    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)

    families = sorted({s["source"] for s in sources if isinstance(s, dict) and s.get("source")})
    meta = {
        "topic": topic,
        **summarize(messages, elapsed, model_name),
        "n_sources": len(sources),
        "source_families": families,
    }

    (out_dir / f"{slug}.sources.json").write_bytes(raw)
    (out_dir / f"{slug}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / f"{slug}.md").write_bytes(report)

    return str(out_dir / f"{slug}.md")


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic)."""
    topic_str = (topic or "").strip()
    if not topic_str:
        print('usage: python research.py "<topic>"', file=sys.stderr)
        return 2

    model = make_model()
    model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or os.getenv("LAB_MODEL", "unknown")
    start = time.monotonic()

    with open_sandbox() as backend:
        backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
        upload(backend, {
            VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
            FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
        })
        agent = build_lead_agent(backend, model)
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic_str)}]},
                config={"recursion_limit": 1000},
            )
        except Exception as exc:
            print(f"FAILED: agent run: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

        try:
            path = save_outputs(backend, topic_str, result.get("messages", []), time.monotonic() - start, model_name)
        except RuntimeError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1

    print(f"Report saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
