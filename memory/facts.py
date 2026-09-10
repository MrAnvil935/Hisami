"""User-fact extraction: robust JSON parsing for LLM output."""

import json
import re
from collections import Counter


def _clean_fact(item, max_len=200):
    if isinstance(item, dict):
        item = item.get("fact") or item.get("text") or ""
    s = str(item).strip().strip("-•*\"'")
    if 2 < len(s) <= max_len:
        return s
    return ""


def _strip_fences(text):
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip())


def parse_facts_json(text: str, max_facts=5, max_len=200):
    if not text:
        return []
    text = str(text).strip()
    # tolerate code fences
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # last resort: extract [...] block
        m = re.search(r"\[.*?\]", text, flags=re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return []
    if isinstance(data, str):
        data = [data]
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        s = _clean_fact(item, max_len)
        if s:
            out.append(s)
        if len(out) >= max_facts:
            break
    return out


def parse_multi_facts_json(text: str, max_facts=5, max_len=200):
    """Parse {"DisplayName": ["fact", ...]} into {name: [facts]}.

    Fail-soft: any unparseable input -> {}. Per-user lists are capped
    at max_facts, entries cleaned with the same rules as parse_facts_json.
    """
    if not text:
        return {}
    cleaned = _strip_fences(text)
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for name, facts in data.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if isinstance(facts, str):
            facts = [facts]
        if not isinstance(facts, list):
            continue
        kept = []
        for item in facts:
            s = _clean_fact(item, max_len)
            if s:
                kept.append(s)
            if len(kept) >= max_facts:
                break
        if kept:
            out[name.strip()] = kept
    return out


def resolve_speakers(chunk):
    """Map display name -> author_id for a message chunk.

    Returns (mapping, ambiguous). Ambiguous names (2+ ids) resolve to
    the most-frequent id in the chunk. Bot rows (role == 'assistant')
    are excluded.
    """
    counts = Counter()
    for m in chunk:
        if m.get("role") == "assistant":
            continue
        name = (m.get("author_name") or "").strip()
        uid = str(m.get("author_id") or "").strip()
        if name and uid:
            counts[(name, uid)] += 1
    mapping, ambiguous = {}, set()
    by_name = {}
    for (name, uid), c in counts.items():
        by_name.setdefault(name, []).append((uid, c))
    for name, pairs in by_name.items():
        pairs.sort(key=lambda p: p[1], reverse=True)
        mapping[name] = pairs[0][0]
        if len(pairs) > 1:
            ambiguous.add(name)
    return mapping, ambiguous
