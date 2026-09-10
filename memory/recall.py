"""Keyword recall over SQLite FTS5 with safe fallbacks.

No new dependencies. FTS5 query is built from sanitized alphanumeric
tokens so user input can never break the MATCH syntax.
"""

import re
import sqlite3

from . import store

_WORD_RE = re.compile(r"[a-z0-9]{2,}")


def tokenize(text: str, max_tokens=10):
    if not text:
        return []
    return _WORD_RE.findall(str(text).lower())[:max_tokens]


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
