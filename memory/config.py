"""JSONC config loading: plain JSON plus // and /* */ comments.

The state-machine stripper is string-aware, so comment markers inside
string values (e.g. "http://localhost") are never touched. Trailing
commas are NOT supported — keep the JSON itself valid. Pure functions
so they are unit-testable.
"""

import json


def strip_json_comments(text):
    """Remove // line and /* block */ comments outside string literals."""
    out = []
    i, n = 0, len(text)
    in_string = False
    escape = False
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n
                                 and text[i + 1] == "/"):
                i += 1
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def load_config(path="config.json"):
    """Load config.json (or .jsonc-style content) into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(strip_json_comments(f.read()))


def diff_configs(old, new):
    """Sorted keys whose values differ between two config dicts.

    Missing-vs-present counts as a change. Pure function (no I/O) so
    it is unit-testable; used by /reload to report what changed.
    """
    old = old if isinstance(old, dict) else {}
    new = new if isinstance(new, dict) else {}
    changed = []
    for key in set(old) | set(new):
        try:
            same = (old.get(key) == new.get(key)
                    and (key in old) == (key in new))
        except Exception:
            same = False
        if not same:
            changed.append(key)
    return sorted(changed)


DEFAULT_OLLAMA_BASE = "http://localhost:11434"


def ollama_base(url, default=DEFAULT_OLLAMA_BASE):
    """Derive the Ollama server base (scheme://host:port) from any URL form.

    Accepts the full chat endpoint ("http://host:port/api/chat") or a bare
    base ("http://host:port") — both yield the base, so /api/ps, /api/embed
    and /api/chat follow the one configured value. Garbage in -> default.
    Pure function (no I/O) so it is unit-testable.
    """
    try:
        from urllib.parse import urlparse
        parts = urlparse(str(url or "").strip())
        if parts.scheme in ("http", "https") and parts.netloc:
            return f"{parts.scheme}://{parts.netloc}"
    except Exception:
        pass
    return default
