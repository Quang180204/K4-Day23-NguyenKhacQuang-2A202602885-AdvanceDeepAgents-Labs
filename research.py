"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
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
    s = topic.strip().lower()
    s = re.sub(r"[^\w]+", "-", s)
    s = s.strip("-")
    s = s[:60].rstrip("-")
    return s if s else "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (
        f"Please conduct an in-depth academic survey and write a comprehensive research report on the topic:\n"
        f"'{topic}'\n\n"
        f"Follow the research and writing guidelines in your system prompt:\n"
        f"1. Plan your steps with `write_todos` and split the topic into 3 sub-questions.\n"
        f"2. Delegate each sub-question to the `researcher` subagent (at least 3 subagent calls):\n"
        f"   - Delegation 1: assign sub-question 1 focusing on 'arxiv' papers.\n"
        f"   - Delegation 2: assign sub-question 2 focusing on 'hf-search' / 'hf-daily' papers.\n"
        f"   - Delegation 3: assign sub-question 3 focusing on 'web' sources (GitHub repositories, official project pages).\n"
        f"3. Aggregate sources into {SOURCES_PATH}. CRITICAL: You MUST include sources from AT LEAST 3 distinct families ('arxiv', 'hf-search' or 'hf-daily', and 'web') in {SOURCES_PATH}.\n"
        f"4. Write the report body (following REPORT_TEMPLATE.md, without ## References) into {REPORT_PATH}. Ensure the text cites sources from all 3 source families inline.\n"
        f"5. Run the finalizer: `execute(command='python3 {FINALIZER_PATH}')` to generate references and renumber citations.\n"
        f"6. Run the validator: `execute(command='python3 {VALIDATOR_PATH} {REPORT_PATH} {SOURCES_PATH}')` until citations are completely verified (prints OK).\n"
        f"   CRITICAL: Never run interactive commands that wait for user input or open an interactive shell.\n"
        f"7. Spot-check 2-3 key claims with `citation-checker`.\n"
    )


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    tool_calls = Counter()
    input_tokens = 0
    output_tokens = 0

    for msg in messages:
        # Extract tool calls
        calls = None
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            calls = msg.tool_calls
        elif isinstance(msg, dict) and msg.get("tool_calls"):
            calls = msg.get("tool_calls")

        if calls:
            for call in calls:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if name:
                    tool_calls[name] += 1

        # Extract token usage metadata
        usage = None
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            usage = msg.usage_metadata
        elif hasattr(msg, "response_metadata") and isinstance(msg.response_metadata, dict):
            usage = msg.response_metadata.get("token_usage") or msg.response_metadata.get("usage")
        elif isinstance(msg, dict):
            usage = msg.get("usage_metadata") or msg.get("response_metadata", {}).get("token_usage")

        if isinstance(usage, dict):
            input_tokens += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)

    subagent_calls = tool_calls.get("task", 0)

    return {
        "model": str(model_name),
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(tool_calls),
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
        },
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)

    if not report_bytes or not report_bytes.strip():
        raise RuntimeError(f"Report at {REPORT_PATH} is missing or empty")

    if not sources_bytes or not sources_bytes.strip():
        raise RuntimeError(f"Sources at {SOURCES_PATH} is missing or empty")

    try:
        sources_data = json.loads(sources_bytes.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Sources at {SOURCES_PATH} is invalid JSON: {exc}")

    if not isinstance(sources_data, list) or len(sources_data) == 0:
        raise RuntimeError(f"Sources at {SOURCES_PATH} must be a non-empty JSON list")

    report_text = report_bytes.decode("utf-8")
    slug = slugify(topic)
    summary_info = summarize(messages, elapsed, model_name)
    source_families = sorted(list(set(str(s.get("source", "")).strip() for s in sources_data if s.get("source"))))

    meta_data = {
        "topic": topic,
        **summary_info,
        "n_sources": len(sources_data),
        "source_families": source_families,
    }

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    sources_path = reports_dir / f"{slug}.sources.json"
    meta_path = reports_dir / f"{slug}.meta.json"
    md_path = reports_dir / f"{slug}.md"

    sources_path.write_text(json.dumps(sources_data, indent=2, ensure_ascii=False), encoding="utf-8")
    meta_path.write_text(json.dumps(meta_data, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(report_text, encoding="utf-8")

    return md_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic).

    PSEUDO-CODE:
      empty topic -> print usage to stderr, return 2
      model = make_model(); start = time.monotonic()
      with open_sandbox() as backend:                # the sandbox is always cleaned up, even on errors
          backend.execute("mkdir -p <WORKDIR>/research/notes <WORKDIR>/report")
          upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
          agent = build_lead_agent(backend, model)
          result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})
          save_outputs(...); on RuntimeError print "FAILED: ..." to stderr and return 1
      print where the report was saved; return 0
    """
    topic = topic.strip()
    if not topic:
        print("Usage: python research.py \"<topic>\"", file=sys.stderr)
        return 2

    print(f"Starting research on topic: {topic}")
    try:
        model = make_model()
    except Exception as exc:
        print(f"FAILED: could not create model: {exc}", file=sys.stderr)
        return 1

    model_name = (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or os.environ.get("LAB_MODEL", "unknown")
    )
    start = time.monotonic()

    try:
        with open_sandbox() as backend:
            backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })

            agent = build_lead_agent(backend, model)
            print(f"Agent running for topic: '{topic}'...", flush=True)
            messages = []
            for event in agent.stream(
                {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                stream_mode="values",
                config={"recursion_limit": 1000},
            ):
                current_msgs = event.get("messages", [])
                if len(current_msgs) > len(messages):
                    for new_msg in current_msgs[len(messages):]:
                        if hasattr(new_msg, "tool_calls") and new_msg.tool_calls:
                            calls = [c.get("name") for c in new_msg.tool_calls]
                            print(f"  [Lead] -> Tool calls: {calls}", flush=True)
                            for c in new_msg.tool_calls:
                                if c.get("name") == "task":
                                    st = c.get("args", {}).get("subagent_type")
                                    desc = str(c.get("args", {}).get("description"))[:120].replace("\n", " ")
                                    print(f"    task args: type={st}, desc={desc}...", flush=True)
                                elif c.get("name") == "execute":
                                    cmd = str(c.get("args", {}).get("command"))[:120].replace("\n", " ")
                                    print(f"    execute args: command={cmd}...", flush=True)
                        elif type(new_msg).__name__ == "ToolMessage":
                            tool_name = getattr(new_msg, "name", "tool")
                            content_preview = str(getattr(new_msg, "content", ""))[:80].replace("\n", " ")
                            print(f"  [Tool] <- {tool_name}: {content_preview}...", flush=True)
                messages = current_msgs

            elapsed = time.monotonic() - start
            report_path = save_outputs(backend, topic, messages, elapsed, model_name)
            print(f"SUCCESS: Report saved to {report_path}")
            return 0
    except RuntimeError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"FAILED: unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))

