"""Web-result rendering for Components V2 messages (no Discord/network).

Pure functions (no I/O) so they are unit-testable. bot.py cannot be
imported by tests (it calls client.run at module level), hence this
module. Used by the /web command, the /prompt web-results button, and
the /prompt agentic search flow (router decision parsing, article
extraction, fetched-content tiers).
"""

import json
import re

from bs4 import BeautifulSoup

V2_TEXT_BUDGET = 3800


def format_web_results(results, max_chars=V2_TEXT_BUDGET):
    """Render search results as markdown for a TextDisplay component.

    Each result: '### [n] [title](url)' + snippet, joined by blank lines.
    Over-budget output is truncated with a notice. Empty input -> ''.
    """
    if not results:
        return ""
    web_text = "\n\n".join(
        f"### [{i}] [{r.get('title') or 'No title'}]({r.get('url', '')})\n"
        f"{r.get('snippet', '')}"
        for i, r in enumerate(results, 1)
    )
    if len(web_text) > max_chars:
        notice = ("\n\n… More results were returned "
                  "but could not fit in this message.")
        web_text = web_text[:max_chars - len(notice)] + notice
    return web_text


ROUTER_SYSTEM = (
    "You decide whether answering the user's question needs a web search. "
    "Reply with ONLY one JSON object, no other text: "
    '{"need_search": true/false, "queries": ["..."]}. '
    "Rules: at most 2 queries; each query must be self-contained "
    "(resolve pronouns from the conversation); pure opinion, recall, or "
    "creative questions need no search (need_search false, queries []). "
    "Current events, facts, prices, releases, people, and anything "
    "time-sensitive need search."
)


def parse_router_decision(text, max_queries=2, fallback_query=""):
    """Parse the router model's verdict into (need_search, queries).

    Returns (True, [fallback_query]) on any failure so the caller keeps
    today's behavior (search the verbatim prompt) instead of breaking.
    Pure function (no I/O) so it is unit-testable.
    """
    try:
        cap = max(1, int(max_queries))
    except (TypeError, ValueError):
        cap = 2
    fallback = [fallback_query] if str(fallback_query or "").strip() else []
    try:
        cleaned = str(text or "").strip()
        if "```" in cleaned:
            m = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned,
                          re.DOTALL | re.IGNORECASE)
            if m:
                cleaned = m.group(1)
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            return True, fallback
        need = data.get("need_search")
        queries = data.get("queries") or []
        if not isinstance(queries, list):
            queries = []
        queries = [str(q).strip() for q in queries
                   if str(q or "").strip()][:cap]
        if need is True:
            return True, queries or fallback
        if need is False:
            return False, []
        return True, queries or fallback
    except Exception:
        return True, fallback


_STRIP_TAGS = ("script", "style", "nav", "header", "footer", "aside",
               "form", "noscript", "iframe")


def extract_article_text(html, max_chars=2000):
    """Pull readable article text out of raw HTML. Never raises.

    Prefers <article>, then <main>, then the <p>-richest container;
    boilerplate tags are dropped. Over-budget text is cut with a
    marker. Empty input or garbage -> ''.
    """
    try:
        budget = int(max_chars)
    except (TypeError, ValueError):
        budget = 2000
    if not html or not str(html).strip():
        return ""
    try:
        soup = BeautifulSoup(str(html), "html.parser")
        for tag in soup(_STRIP_TAGS):
            tag.decompose()
        root = soup.find("article") or soup.find("main")
        if root is None:
            best, best_len = None, 0
            for cand in soup.find_all(["div", "section", "body"]):
                plen = sum(len(p.get_text(" ", strip=True))
                           for p in cand.find_all("p", recursive=False)
                           or cand.find_all("p"))
                if plen > best_len:
                    best, best_len = cand, plen
            root = best or soup
        text = root.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) > budget:
            text = (text[:max(0, budget - 1)] + "…"
                    if budget > 0 else "")
        return text
    except Exception:
        return ""


def format_fetched_tier(results, contents, max_chars=6000):
    """Render fetched page bodies for the model prompt.

    contents: {url: text}. Results without fetched text are skipped
    (caller renders them as snippets). Empty -> ''.
    """
    blocks = []
    for i, r in enumerate(results or [], 1):
        body = (contents or {}).get(r.get("url", ""), "")
        if not str(body or "").strip():
            continue
        blocks.append(
            f"[{i}] {r.get('title', 'No title')}\n"
            f"URL: {r.get('url', '')}\n"
            f"{body.strip()}"
        )
    if not blocks:
        return ""
    text = "\n\n---\n\n".join(blocks)
    if len(text) > max_chars:
        text = text[:max(0, max_chars - 1)] + "…"
    return text


def format_prompt_sources(record, max_chars=V2_TEXT_BUDGET):
    """Render what the model was given for the /prompt results button.

    record: {"searched": bool, "queries": [...], "results": [...],
    "fetched": {url: chars}}. Declined search -> short notice.
    Over-budget output is truncated with a notice. Empty input -> ''.
    """
    if not record:
        return ""
    if not record.get("searched", True):
        return ("The model decided no web search was needed "
                "for this question.")
    parts = []
    queries = record.get("queries") or []
    if queries:
        parts.append("Searched for: " + "; ".join(str(q) for q in queries))
    fetched = record.get("fetched") or {}
    lines = []
    for i, r in enumerate(record.get("results") or [], 1):
        line = (f"### [{i}] [{r.get('title') or 'No title'}]"
                f"({r.get('url', '')})\n{r.get('snippet', '')}")
        if r.get("url", "") in fetched:
            line += "\n📄 full page text was included in the prompt"
        lines.append(line)
    if lines:
        parts.append("\n\n".join(lines))
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        notice = "\n\n… More sources were used but could not fit here."
        text = text[:max_chars - len(notice)] + notice
    return text
