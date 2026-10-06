"""Long-term episodic memory: chunk summarization helpers.

The actual LLM call lives in bot.py (reuses generate_reply with the
local-Ollama-first chain). This module only builds prompts and parses
results so it stays testable without Discord or network.
"""

import json
import re
import time

SUMMARIZER_SYSTEM = (
    "You summarize Discord chat for a bot's long-term memory. "
    "Be concise and factual. Capture: topics discussed, decisions, "
    "open questions, user preferences stated, and any facts worth "
    "remembering. No roleplay, no extra commentary."
)

CURATION_FACTS_SYSTEM = (
    "You curate a bot's per-user fact database. Delete only facts that "
    "are clearly stale, superseded by a newer numbering/version, "
    "duplicated, or worthless trivia. When two facts conflict, keep the "
    "newer. When in doubt, KEEP the fact. Return ONLY a JSON object."
)

CURATION_SUMMARIES_SYSTEM = (
    "You curate a bot's conversation-summary archive. Delete only "
    "summaries that are redundant with another, or contain no durable "
    "information (e.g. 'nothing happened', pure greetings). When in "
    "doubt, KEEP the summary. Return ONLY a JSON object."
)


def build_summary_prompt(messages, max_chars=500):
    lines = []
    for m in messages:
        author = m.get("author_name", "?")
        content = (m.get("content") or "")[:max_chars]
        lines.append(f"{author}: {content}")
    convo = "\n".join(lines)
    return (
        "Summarize this Discord conversation chunk in 5-10 bullet points. "
        "Keep names attached to opinions/facts.\n\n"
        f"{convo}"
    )


def build_multi_fact_prompt(messages, max_chars=300):
    """Prompt extracting per-user facts for ALL human speakers in a chunk.

    Bot rows (role == 'assistant') are excluded by the caller convention:
    this builder skips them defensively as well. Returns prompt text
    expecting a JSON object: {"DisplayName": ["fact", ...]}.
    """
    lines = []
    for m in messages:
        if m.get("role") == "assistant":
            continue
        author = m.get("author_name", "?")
        content = (m.get("content") or "")[:max_chars]
        lines.append(f"{author}: {content}")
    convo = "\n".join(lines)
    return (
        "Extract durable facts about EACH person in this chat. "
        'Return ONLY a JSON object mapping display name to facts, e.g. '
        '{"alice": ["likes osu"], "bob": ["lives in Berlin"]}. '
        "People with no durable facts may be omitted or mapped to []. "
        "Record stable traits only: preferences, skills, ownership, location. "
        "Do NOT record what someone is currently discussing, working on, "
        "or asking about unless stated as a stable trait. "
        "Skip greetings, jokes, transient moods.\n\n"
        f"{convo}"
    )


def _age_str(timestamp, now=None):
    """Human-readable age like '3d ago' for curation prompts; '' if unknown."""
    try:
        ts = float(timestamp or 0)
    except (TypeError, ValueError):
        return ""
    if ts <= 0:
        return ""
    now = time.time() if now is None else now
    secs = max(0, int(now - ts))
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


def build_curate_facts_prompt(rows, max_chars=200, now=None):
    """Prompt the curated deletion of stale user facts.

    `rows` are dicts with 'fact' and 'updated_at'. Numbered from 1 so the
    model can only reference an index (never echo text back). Expects a
    JSON object {"delete": [1-based indices]}.
    """
    lines = []
    for i, r in enumerate(rows, 1):
        age = _age_str(r.get("updated_at"), now)
        fact = (r.get("fact") or "")[:max_chars]
        suffix = f" ({age})" if age else ""
        lines.append(f"{i}. {fact}{suffix}")
    body = "\n".join(lines)
    return (
        "Below are durable facts about one user, numbered. Identify any "
        "that are stale, superseded, duplicated, or worthless trivia. "
        'Reply with ONLY JSON: {"delete": [numbers]}. Use [] to delete '
        "nothing. Never include fact text.\n\n"
        f"{body}"
    )


def build_curate_summaries_prompt(rows, max_chars=400, now=None):
    """Prompt the curated deletion of redundant/empty summaries.

    `rows` are dicts with 'summary' and 'created_at'. Numbered from 1;
    expects {"delete": [1-based indices]}.
    """
    lines = []
    for i, r in enumerate(rows, 1):
        age = _age_str(r.get("created_at"), now)
        text = (r.get("summary") or "")[:max_chars]
        suffix = f" ({age})" if age else ""
        lines.append(f"{i}. {text}{suffix}")
    body = "\n".join(lines)
    return (
        "Below are conversation summaries, numbered oldest first. Identify "
        "only those that are redundant with another entry or contain no "
        'durable information. Reply with ONLY JSON: {"delete": [numbers]}. '
        "Use [] to delete nothing. Never include summary text.\n\n"
        f"{body}"
    )


def parse_curate_indices(text, count):
    """Parse a curation reply into sorted 0-based indices within range.

    Accepts {"delete": [...]}, {"remove": [...]}, or a bare list. Only
    integers in 1..count survive (the prompt numbers from 1); anything
    non-integer, out of range, or duplicated is dropped. Fail-soft: bad
    or empty input -> []. Pure function.
    """
    try:
        count = int(count)
    except (TypeError, ValueError):
        return []
    if not text or count <= 0:
        return []
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip())
    data = None
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        m = re.search(r"\[.*?\]", cleaned, flags=re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                data = None
    if isinstance(data, dict):
        data = data.get("delete", data.get("remove", []))
    if isinstance(data, (int, float)):
        data = [data]
    if not isinstance(data, list):
        return []
    out = set()
    for value in data:
        if isinstance(value, bool):
            continue
        try:
            n = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= count:
            out.add(n - 1)
    return sorted(out)
