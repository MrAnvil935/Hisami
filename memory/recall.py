"""Keyword recall over SQLite FTS5 with safe fallbacks.

No new dependencies. FTS5 query is built from sanitized alphanumeric
tokens so user input can never break the MATCH syntax.
"""

import re
import sqlite3

from . import store

_WORD_RE = re.compile(r"[a-z0-9]{2,}")
_MENTION_RE = re.compile(r"<@!?(\d+)>")
_NAME_SPLIT_RE = re.compile(r"[_\-.\s]+")


def tokenize(text: str, max_tokens=10):
    if not text:
        return []
    return _WORD_RE.findall(str(text).lower())[:max_tokens]


def name_mentioned(name, lowered_text, text_words=None):
    """True if display `name` is referenced in already-lowered text.

    Multi-word names match as a substring ("Ann Marie"); single tokens
    match as whole words only, so short forms hit ("nos" from "nos_yous")
    while prefixes don't ("bob" must not fire on "bobby"). Minimum 3
    chars per matchable unit guards against nicks like "Al". Pure function.
    """
    name_l = str(name or "").lower()
    if text_words is None:
        text_words = set(_WORD_RE.findall(lowered_text))
    tokens = [t for t in _NAME_SPLIT_RE.split(name_l) if t]
    if len(tokens) > 1 and len(name_l) >= 3 and name_l in lowered_text:
        return True
    for tok in tokens:
        if len(tok) >= 3 and tok in text_words:
            return True
    return False


def _fts_query(tokens):
    # OR semantics: any keyword match is useful for recall.
    return " OR ".join(tokens)


def search_messages(channel_id, query, limit=5):
    """Return up to `limit` past messages relevant to query.

    Each item: {msg_id, author_name, content, score}. Newest irrelevant
    chatter is excluded by ranking on match quality, not recency.
    """
    tokens = tokenize(query)
    if not tokens:
        return []
    con = sqlite3.connect(store.DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        try:
            rows = con.execute(
                """SELECT channel_id, msg_id, author_name, content,
                          rank AS score
                   FROM messages_fts
                   WHERE messages_fts MATCH ?
                     AND channel_id = ?
                   ORDER BY rank LIMIT ?""",
                (_fts_query(tokens), str(channel_id), int(limit) * 3),
            ).fetchall()
            # FTS5 rank: more negative = better. Normalize for display.
            out = []
            for r in rows[: int(limit)]:
                d = dict(r)
                try:
                    d["score"] = -float(d.get("score", 0.0))
                except (TypeError, ValueError):
                    d["score"] = 0.0
                out.append(d)
            if out:
                return out
        except sqlite3.OperationalError:
            pass  # fall through to LIKE
        # LIKE fallback (also used when FTS has no hits)
        likes = " OR ".join(["LOWER(content) LIKE ?"] * len(tokens))
        params = [f"%{t}%" for t in tokens]
        rows = con.execute(
            f"""SELECT channel_id, msg_id, author_name, content, 0.0 AS score
                FROM messages WHERE channel_id=? AND ({likes})
                ORDER BY msg_id DESC LIMIT ?""",
            (str(channel_id), *params, int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def find_referenced_users(messages, author_id, current_text="", max_users=2):
    """Find OTHER users relevant to the current message.

    Sources, in priority order:
      1. Discord mentions (<@id>) in the current message text,
      2. recent speakers whose display name appears in the current text,
      3. other recent human speakers, newest first.

    Returns (ordered_ids, names) where names maps id -> display name.
    The message author (author_id) and assistant rows are always excluded.
    Pure function (no I/O) so it is unit-testable.
    """
    self_id = str(author_id or "")
    ordered, names = [], {}

    def _add(uid, name=""):
        uid = str(uid or "").strip()
        if not uid or uid == self_id or uid in ordered:
            return
        if len(ordered) >= max_users:
            return
        ordered.append(uid)
        if name and uid not in names:
            names[uid] = name

    text = str(current_text or "")

    # 1. explicit mentions in the current message
    for uid in _MENTION_RE.findall(text):
        _add(uid)

    # collect recent human speakers, newest first
    speakers = []  # (uid, name)
    for m in reversed(messages or []):
        if m.get("role") == "assistant":
            continue
        uid = str(m.get("author_id") or "").strip()
        name = str(m.get("author_name") or "").strip()
        if not uid or uid == self_id:
            continue
        speakers.append((uid, name))
        if name and uid not in names:
            names[uid] = name

    lowered = text.lower()
    text_words = set(_WORD_RE.findall(lowered))

    # 2. speakers named in the current message
    for uid, name in speakers:
        if uid in ordered or len(ordered) >= max_users:
            continue
        if name and name_mentioned(name, lowered, text_words):
            _add(uid, name)

    # 3. other recent speakers, newest first
    for uid, name in speakers:
        if len(ordered) >= max_users:
            break
        _add(uid, name)

    return ordered, names
