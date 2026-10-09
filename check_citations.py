"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import sys

import re

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")   # [3], [1, 2], [1-3]; not [3](link)
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")


def _group_numbers(group):
    numbers = []
    for part in re.split(r"\s*,\s*", group):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            numbers.extend(range(a, b + 1) if 0 <= b - a <= 200 else [a, b])
        else:
            numbers.append(int(part))
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK)."""
    problems = []
    if not isinstance(sources, list) or len(sources) == 0:
        return ["no sources in sources.json"]

    sources_by_n = {}
    seen_urls = {}

    for i, s in enumerate(sources):
        if not isinstance(s, dict):
            problems.append(f"source #{i}: not an object")
            continue

        n = s.get("n")
        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"source #{i}: n is not an integer")
            continue

        if n in sources_by_n:
            problems.append(f"duplicate source number [{n}]")
        else:
            sources_by_n[n] = s

        url = s.get("url")
        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n}]: url must start with http:// or https://")
        elif url in seen_urls:
            problems.append(f"source [{n}]: duplicate url (same as [{seen_urls[url]}])")
        else:
            seen_urls[url] = n

    matches = list(_REF_HEADING.finditer(report_text))
    if not matches:
        problems.append("missing '## References' heading")
        body = report_text
        refs = ""
    else:
        m = matches[-1]
        body = report_text[:m.start()]
        refs = report_text[m.end():]

    segments = _CODE.split(body)
    cited = set()
    for i, segment in enumerate(segments):
        if i % 2 == 1:
            continue
        for match in _GROUP.finditer(segment):
            for num in _group_numbers(match.group(1)):
                cited.add(num)

    if not cited:
        problems.append("no citations found in the report body")

    for num in sorted(cited):
        if num not in sources_by_n:
            problems.append(f"[{num}] cited but missing from sources.json")

    for n in sorted(sources_by_n.keys()):
        if n not in cited:
            problems.append(f"source [{n}] never cited")

    if matches:
        ref_lines = []
        for line in refs.splitlines():
            line_str = line.strip()
            match = re.match(r"^\[(\d+)\](.*)$", line_str)
            if match:
                ref_lines.append((int(match.group(1)), line_str))

        seen_refs = {}
        for num, line_str in ref_lines:
            if num not in sources_by_n:
                problems.append(f"reference line [{num}] is not a source")
            if num in seen_refs:
                problems.append(f"reference [{num}] listed more than once")
            else:
                seen_refs[num] = line_str

            urls = re.findall(r"https?://[^\s<>\"']+", line_str)
            if len(urls) != 1:
                problems.append(f"reference [{num}] must contain exactly one URL (found {len(urls)})")
            elif num in sources_by_n:
                expected_url = sources_by_n[num].get("url")
                found_url = urls[0]
                if found_url != expected_url and found_url.rstrip(".,;)") != expected_url:
                    problems.append(f"reference [{num}] url does not match sources.json")

        for n in sorted(sources_by_n.keys()):
            if n not in seen_refs:
                problems.append(f"source [{n}] has no reference line")

    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
