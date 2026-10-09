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
import time
import xml.etree.ElementTree

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

_LAST_ARXIV_CALL = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    Handles RetryableError, httpx.HTTPStatusError (429, 500, 502, 503, 504), and httpx.TransportError.
    Uses exponential backoff with random jitter, respects Retry-After header, caps delay at `cap`.
    Gives up after `attempts` tries (does not sleep after the last attempt).
    """
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            is_retryable = False
            retry_after = None

            if isinstance(e, RetryableError):
                is_retryable = True
                retry_after = e.retry_after
            elif isinstance(e, httpx.HTTPStatusError):
                if e.response.status_code in (429, 500, 502, 503, 504):
                    is_retryable = True
                    ra = e.response.headers.get("Retry-After")
                    if ra:
                        try:
                            retry_after = float(ra)
                        except ValueError:
                            pass
            elif isinstance(e, httpx.TransportError):
                is_retryable = True

            if not is_retryable or attempt == attempts - 1:
                raise e

            if retry_after is not None:
                delay = retry_after
            else:
                delay = base * (2 ** attempt) + random.uniform(0.1, 1.0)

            delay = min(delay, cap)
            time.sleep(delay)


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    global _LAST_ARXIV_CALL
    try:
        terms = [t for t in re.findall(r"[\w-]+", query) if t.upper() not in ("AND", "OR", "NOT")]
        if not terms:
            return "NO RESULTS"

        elapsed = time.time() - _LAST_ARXIV_CALL
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)

        search_query = " AND ".join(f"all:{t}" for t in terms)
        clamped_max = max(1, min(max_results, 30))
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": clamped_max,
        }

        def _fetch():
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(ARXIV_URL, params=params)
                resp.raise_for_status()
                return resp.text

        xml_text = None
        try:
            try:
                xml_text = with_retry(_fetch, attempts=3, base=1.0, cap=10.0)
            finally:
                _LAST_ARXIV_CALL = time.time()
        except Exception:
            pass

        if xml_text:
            root = xml.etree.ElementTree.fromstring(xml_text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", ns)
            if entries:
                records = []
                for entry in entries:
                    raw_id = entry.findtext("atom:id", "", namespaces=ns).strip()
                    paper_id = raw_id.split("/abs/")[-1].strip()
                    paper_id = re.sub(r"v\d+$", "", paper_id)
                    url = f"https://arxiv.org/abs/{paper_id}"
                    published = entry.findtext("atom:published", "", namespaces=ns).strip()[:10]
                    raw_title = entry.findtext("atom:title", "", namespaces=ns) or ""
                    title = " ".join(raw_title.split())
                    raw_summary = entry.findtext("atom:summary", "", namespaces=ns) or ""
                    summary = " ".join(raw_summary.split())[:600]
                    records.append({
                        "id": paper_id,
                        "url": url,
                        "published": published,
                        "title": title,
                        "summary": summary,
                    })
                if records:
                    return json.dumps(records, ensure_ascii=False)

        # Fallback when export.arxiv.org rate limits (429): fetch verified arXiv papers via HF papers search index
        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.get(HF_SEARCH_URL, params={"q": " ".join(terms), "limit": clamped_max})
                if resp.status_code == 200:
                    items = resp.json()
                    records = []
                    for item in items:
                        paper = item.get("paper") if isinstance(item, dict) else None
                        if not paper or not isinstance(paper, dict) or not paper.get("id"):
                            continue
                        paper_id = str(paper["id"])
                        url = f"https://arxiv.org/abs/{paper_id}"
                        published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
                        raw_title = str(paper.get("title") or item.get("title") or "")
                        title = " ".join(raw_title.split())
                        raw_summary = str(paper.get("ai_summary") or paper.get("summary") or item.get("summary") or "")
                        summary = " ".join(raw_summary.split())[:600]
                        records.append({
                            "id": paper_id,
                            "url": url,
                            "published": published,
                            "title": title,
                            "summary": summary,
                        })
                    if records:
                        return json.dumps(records, ensure_ascii=False)
        except Exception:
            pass

        return "NO RESULTS"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    try:
        clamped_limit = max(1, min(limit, 100))
        params = {"limit": clamped_limit}
        if date and date.strip():
            params["date"] = date.strip()

        def _fetch():
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(HF_DAILY_URL, params=params)
                resp.raise_for_status()
                return resp.json()

        items = with_retry(_fetch, attempts=5, base=1.0, cap=30.0)
        if not isinstance(items, list) or not items:
            return "NO RESULTS"

        records = []
        for item in items:
            paper = item.get("paper") if isinstance(item, dict) else None
            if not paper or not isinstance(paper, dict) or not paper.get("id"):
                continue
            paper_id = str(paper["id"])
            url = f"https://huggingface.co/papers/{paper_id}"
            published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            raw_title = str(paper.get("title") or item.get("title") or "")
            title = " ".join(raw_title.split())
            raw_summary = str(paper.get("summary") or item.get("summary") or "")
            summary = " ".join(raw_summary.split())[:600]
            upvotes = paper.get("upvotes") or 0
            github = paper.get("githubRepo") or ""
            stars = paper.get("githubStars") or 0

            if keyword and keyword.strip():
                kw = keyword.strip().lower()
                if kw not in (title + " " + summary).lower():
                    continue

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if not records:
            return "NO RESULTS"
        records.sort(key=lambda r: r.get("upvotes", 0), reverse=True)
        return json.dumps(records, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    try:
        if not query or not query.strip():
            return "NO RESULTS"
        clamped_limit = max(1, min(limit, 50))
        params = {"q": query.strip(), "limit": clamped_limit}

        def _fetch():
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(HF_SEARCH_URL, params=params)
                resp.raise_for_status()
                return resp.json()

        items = with_retry(_fetch, attempts=5, base=1.0, cap=30.0)
        if not isinstance(items, list) or not items:
            return "NO RESULTS"

        records = []
        for item in items:
            paper = item.get("paper") if isinstance(item, dict) else None
            if not paper or not isinstance(paper, dict) or not paper.get("id"):
                continue
            paper_id = str(paper["id"])
            url = f"https://huggingface.co/papers/{paper_id}"
            published = str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10]
            raw_title = str(paper.get("title") or item.get("title") or "")
            title = " ".join(raw_title.split())
            raw_summary = str(paper.get("ai_summary") or paper.get("summary") or item.get("summary") or "")
            summary = " ".join(raw_summary.split())[:600]
            upvotes = paper.get("upvotes") or 0
            github = paper.get("githubRepo") or ""
            stars = paper.get("githubStars") or 0

            records.append({
                "id": paper_id,
                "url": url,
                "published": published,
                "title": title,
                "summary": summary,
                "upvotes": upvotes,
                "github": github,
                "stars": stars,
            })

        if not records:
            return "NO RESULTS"
        return json.dumps(records, ensure_ascii=False)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
def _call_exa_mcp(tool_name: str, arguments: dict) -> str:
    exa_key = os.environ.get("EXA_API_KEY", "").strip()
    target_url = f"{EXA_URL}?exaApiKey={exa_key}" if exa_key else EXA_URL
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }

    def _post():
        with httpx.Client(timeout=45.0) as client:
            resp = client.post(target_url, headers=headers, json=payload)
            resp.raise_for_status()

            data = None
            for line in resp.text.splitlines():
                if line.startswith("data:"):
                    data = json.loads(line[5:].strip())
                    break
            if data is None:
                try:
                    data = resp.json()
                except Exception:
                    pass

            if not data:
                raise RuntimeError("Empty response from Exa MCP endpoint")

            if "error" in data:
                err_str = str(data["error"])
                if "rate" in err_str.lower() or "limit" in err_str.lower():
                    raise RetryableError(f"Exa rate limited via error: {err_str}", retry_after=20.0)
                raise RuntimeError(f"JSON-RPC error: {err_str}")

            result = data.get("result", {})
            meta = result.get("_meta", {})
            meta_str = str(meta).lower()
            if "rate" in meta_str or "limit" in meta_str or meta.get("rateLimit") or meta.get("rateLimited"):
                raise RetryableError(f"Exa rate limited via _meta: {meta}", retry_after=20.0)

            content = result.get("content", [])
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    t = item.get("text", "")
                    low_t = t.lower()
                    if ("rate limit" in low_t or "rate-limited" in low_t or "too many requests" in low_t) and len(t) < 500:
                        raise RetryableError(f"Exa rate limit message in content: {t}", retry_after=20.0)
                    texts.append(t)

            return "\n\n".join(texts).strip()

    attempts = 5 if exa_key else 2
    cap = 30.0 if exa_key else 5.0
    return with_retry(_post, attempts=attempts, base=2.0, cap=cap)


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    try:
        if not query or not query.strip():
            return "NO RESULTS"
        obj = objective.strip() if objective and objective.strip() else f"Find research papers and surveys about {query.strip()}"
        num_res = max(1, min(num_results, 20))
        try:
            text = _call_exa_mcp("web_search_exa", {"query": query.strip(), "objective": obj, "numResults": num_res})
            if text and text.strip():
                return text
        except Exception:
            pass

        # Fallback when Exa rate limits: query GitHub and Wikipedia for authoritative web sources
        records = []
        try:
            with httpx.Client(timeout=10.0) as client:
                headers = {"User-Agent": "DeepResearchAgent/1.0 (academic survey; student@univ.edu)"}
                gh_resp = client.get(f"https://api.github.com/search/repositories?q={query.strip()}&per_page={num_res}", headers=headers)
                if gh_resp.status_code == 200:
                    for it in gh_resp.json().get("items", [])[:num_res]:
                        records.append(f"Title: {it.get('full_name')}\nURL: {it.get('html_url')}\nSnippet: {it.get('description') or ''}")
                wiki_resp = client.get(f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query.strip()}&format=json", headers=headers)
                if wiki_resp.status_code == 200:
                    for w in wiki_resp.json().get("query", {}).get("search", [])[:3]:
                        title = w.get("title", "")
                        wiki_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                        snippet = re.sub(r"<[^>]+>", "", w.get("snippet") or "")
                        records.append(f"Title: {title}\nURL: {wiki_url}\nSnippet: {snippet}")
        except Exception:
            pass

        if records:
            return "\n\n".join(records)
        return "NO RESULTS"
    except Exception as exc:
        exa_key = os.environ.get("EXA_API_KEY", "").strip()
        msg = f"ERROR: {type(exc).__name__}: {exc}"
        if exa_key:
            msg = msg.replace(exa_key, "[REDACTED]")
        return msg


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    try:
        clean_url = url.strip()
        if not clean_url:
            return "NO RESULTS"

        # First attempt: Exa MCP endpoint
        try:
            text = _call_exa_mcp("web_fetch_exa", {"urls": [clean_url]})
            if text and text.strip():
                return text[:5000]
        except Exception:
            pass

        # Resilient fallback: direct fetch for accessible web/paper pages
        try:
            with httpx.Client(timeout=20.0, follow_redirects=True) as client:
                resp = client.get(clean_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                resp.raise_for_status()
                body = resp.text
                body = re.sub(r"<script.*?</script>", "", body, flags=re.DOTALL | re.IGNORECASE)
                body = re.sub(r"<style.*?</style>", "", body, flags=re.DOTALL | re.IGNORECASE)
                body = re.sub(r"<[^>]+>", " ", body)
                cleaned = " ".join(body.split())
                if cleaned:
                    return cleaned[:5000]
        except Exception:
            pass

        return "NO RESULTS"
    except Exception as exc:
        exa_key = os.environ.get("EXA_API_KEY", "").strip()
        msg = f"ERROR: {type(exc).__name__}: {exc}"
        if exa_key:
            msg = msg.replace(exa_key, "[REDACTED]")
        return msg


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
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
