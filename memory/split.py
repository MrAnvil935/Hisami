"""Reply splitter: break long replies into Discord-sized chunks.

Discord caps messages at 2000 chars. Instead of hard-truncating, split
into at most max_parts chunks, preferring paragraph breaks, and never
leave a fenced code block open across a chunk boundary (close it at the
end of one chunk, reopen at the start of the next). Anything beyond the
cap is truncated with an ellipsis marker.

Pure functions (no Discord, no I/O) so they are unit-testable.
"""

import re

DISCORD_LIMIT = 2000
FENCE = "```"
ELLIPSIS = "…"

_BLANK_RE = re.compile(r"\n\s*\n")
_NEWLINE_RE = re.compile(r"\n")


def _inside_fence(text):
    """True if text ends inside an unclosed ``` block."""
    return text.count(FENCE) % 2 == 1


def _find_cut(s, limit):
    """Index to cut s (len(s) > limit) so head fits in limit.

    Prefers a blank line, then any newline, both outside fenced blocks.
    Falls back to a hard cut, leaving room for a fence closer when the
    cut lands inside a code block. Returns (cut, inside_fence).
    """
    window = s[:limit]
    blanks = [m.start() for m in _BLANK_RE.finditer(window)]
    for cut in reversed(blanks):
        head = window[:cut].rstrip()
        if head and not _inside_fence(window[:cut]):
            return len(head), False
    newlines = [m.start() for m in _NEWLINE_RE.finditer(window)]
    for cut in reversed(newlines):
        head = window[:cut].rstrip()
        if head and not _inside_fence(window[:cut]):
            return len(head), False
    if _inside_fence(window):
        return max(1, limit - len("\n" + FENCE)), True
    return limit, False


def split_message(text, limit=DISCORD_LIMIT, max_parts=3):
    """Split text into <= max_parts chunks of <= limit chars.

    Fenced blocks are closed/reopened across boundaries. The tail past
    the cap is dropped with an ellipsis marker. Never raises; always
    returns at least one chunk.
    """
    try:
        text = text if isinstance(text, str) else str(text)
    except Exception:
        return ["(unrenderable reply)"]
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = DISCORD_LIMIT
    try:
        max_parts = int(max_parts)
    except (TypeError, ValueError):
        max_parts = 3
    limit = max(limit, len(FENCE) * 2 + 8)
    max_parts = max(max_parts, 1)

    if len(text) <= limit:
        return [text]

    parts, rest = [], text
    for _ in range(max_parts - 1):
        if len(rest) <= limit:
            break
        cut, _ = _find_cut(rest, limit)
        head, rest = rest[:cut], rest[cut:]
        rest = rest.lstrip("\n")
        if not rest:
            rest = ""
            parts.append(head)
            break
        parts.append(head)
    if rest:
        if len(rest) > limit:
            rest = rest[:limit - len(ELLIPSIS)] + ELLIPSIS
        parts.append(rest)

    # Fence repair: every chunk must end outside a code block so
    # formatting never bleeds across messages.
    out, carry_open = [], False
    for p in parts:
        if carry_open:
            p = FENCE + "\n" + p
            if len(p) > limit:
                p = p[:limit]
        if _inside_fence(p):
            if len(p) + len("\n" + FENCE) > limit:
                p = p[:limit - len("\n" + FENCE)]
            p = p + "\n" + FENCE
            carry_open = True
        else:
            carry_open = False
        out.append(p)
    return out or [text[:limit]]
