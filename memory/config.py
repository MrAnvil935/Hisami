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
