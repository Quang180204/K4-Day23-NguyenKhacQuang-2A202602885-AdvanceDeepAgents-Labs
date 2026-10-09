"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_GROUP = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()")   # [3]  [1, 2]  [1-3]  [2-3]; not [3](link)
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_REF_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")


def _group_numbers(group_str):
    numbers = []
    for part in re.split(r"\s*,\s*", group_str.strip()):
        span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
        if span:
            a, b = int(span.group(1)), int(span.group(2))
            if 0 <= b - a <= 200:
                numbers.extend(range(a, b + 1))
            else:
                numbers.extend([a, b])
        elif part.isdigit():
            numbers.append(int(part))
    return numbers


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    Validates:
      1. sources is not empty, is a list of dicts.
      2. Every source: n is int, url starts with http:// or https://, no duplicate URLs or duplicate n.
      3. report_text contains a ## References heading.
      4. Every [n] in the body (excluding code blocks and markdown links) exists in sources;
         every source in sources is cited at least once in the body.
      5. The ## References section has exactly ONE line starting with [n] for each source.
      6. Each reference line contains exactly ONE URL, matching that source's url.
    """
    problems = []

    if not isinstance(sources, list) or len(sources) == 0:
        return ["no sources in sources.json"]

    sources_by_n = {}
    seen_urls = {}
    for i, entry in enumerate(sources):
        if not isinstance(entry, dict):
            problems.append(f"sources[{i}] must be a JSON object")
            continue

        n = entry.get("n")
        if not isinstance(n, int):
            problems.append(f"source entry at index {i} 'n' must be an int, got {type(n).__name__}: {n!r}")
        elif n in sources_by_n:
            problems.append(f"duplicate source number [{n}] in sources.json")
        else:
            sources_by_n[n] = entry

        url = entry.get("url")
        if not isinstance(url, str) or not (url.startswith("http://") or url.startswith("https://")):
            problems.append(f"source [{n}] url must start with http:// or https://: {url!r}")
        elif url in seen_urls:
            problems.append(f"duplicate url in sources.json: {url} (used by [{seen_urls[url]}] and [{n}])")
        else:
            seen_urls[url] = n

    matches = list(_REF_HEADING.finditer(report_text))
    if not matches:
        problems.append("missing ## References section heading")
        return problems

    last_match = matches[-1]
    body = report_text[:last_match.start()]
    ref_section = report_text[last_match.end():]

    # Extract citations in body (ignoring code blocks and markdown links)
    segments = _CODE.split(body)
    cited = set()
    for i, segment in enumerate(segments):
        if i % 2 == 1:
            continue  # inside code block or inline code
        for match in _GROUP.finditer(segment):
            for num in _group_numbers(match.group(1)):
                cited.add(num)

    for num in sorted(cited):
        if num not in sources_by_n:
            problems.append(f"[{num}] cited in body but missing from sources.json")

    for n in sorted(sources_by_n.keys()):
        if n not in cited:
            problems.append(f"source [{n}] never cited in body")

    # Parse ## References section lines starting with [n]
    ref_lines = {}
    for line in ref_section.splitlines():
        line_stripped = line.strip()
        m = re.match(r"^\[(\d+)\]", line_stripped)
        if m:
            n = int(m.group(1))
            ref_lines.setdefault(n, []).append(line_stripped)

    for n in sorted(ref_lines.keys()):
        if n not in sources_by_n:
            problems.append(f"reference line [{n}] is not in sources.json")

    for n in sorted(sources_by_n.keys()):
        if n not in ref_lines:
            problems.append(f"source [{n}] missing reference line in ## References")
        elif len(ref_lines[n]) > 1:
            problems.append(f"reference line [{n}] appears {len(ref_lines[n])} times (expected 1)")

    for n, lines in ref_lines.items():
        if n in sources_by_n and len(lines) == 1:
            line_text = lines[0]
            target_url = sources_by_n[n].get("url")
            raw_urls = re.findall(r"https?://[^\s<>\"'`]+", line_text)
            urls = []
            for u in raw_urls:
                if u != target_url and u.rstrip(".,;)") == target_url:
                    urls.append(u.rstrip(".,;)"))
                else:
                    urls.append(u)

            if len(urls) != 1:
                problems.append(f"reference line [{n}] must contain exactly one URL (found {len(urls)})")
            elif urls[0] != target_url:
                problems.append(f"reference line [{n}] URL does not match sources.json ({urls[0]} != {target_url})")

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
