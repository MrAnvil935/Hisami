"""Keyword recall over SQLite FTS5 with safe fallbacks.

No new dependencies. FTS5 query is built from sanitized alphanumeric
tokens so user input can never break the MATCH syntax.
"""

import re
import sqlite3

from . import buffer as mem_buffer
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


def rank_by_overlap(items, query, text_key="fact", time_key="updated_at"):
    """Rank dicts by keyword overlap with query (desc), recency tiebreak.

    Zero-overlap falls back to pure recency order, matching the legacy
    injection behavior. Pure function (no I/O) so it is unit-testable.
    """
    query_tokens = set(tokenize(query, max_tokens=20))
    scored = []
    for item in items:
        text = str(item.get(text_key) or "").lower()
        overlap = (sum(1 for t in query_tokens if t in text)
                   if query_tokens else 0)
        try:
            ts = float(item.get(time_key) or 0)
        except (TypeError, ValueError):
            ts = 0
        scored.append((overlap, ts, item))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    return [s[2] for s in scored]


def cosine_sim(a, b):
    """Cosine similarity of two float sequences. 0.0 on bad input."""
    if a is None or b is None:
        return 0.0
    try:
        import numpy as np
        va = np.asarray(a, dtype="float64").ravel()
        vb = np.asarray(b, dtype="float64").ravel()
        if va.shape != vb.shape or va.size == 0:
            return 0.0
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if not denom:
            return 0.0
        result = float(np.dot(va, vb) / denom)
        return result if result == result else 0.0  # NaN guard
    except Exception:
        return 0.0


def decode_embedding(blob, dim=768):
    """Decode a float32 BLOB from the DB. None when missing/invalid."""
    if not blob:
        return None
    try:
        import numpy as np
        vec = np.frombuffer(bytes(blob), dtype="float32")
        if vec.size != dim:
            return None
        return vec
    except Exception:
        return None


def rank_hybrid(items, query, query_vec=None, text_key="fact",
                time_key="updated_at", vec_key="embedding",
                sem_weight=2.0, kw_weight=0.5):
    """Rank by semantic similarity + keyword overlap + recency.

    Items without a decodable embedding (or when query_vec is None)
    score on keywords only — identical to rank_by_overlap for those rows,
    so mixed migrated/unmigrated pools degrade gracefully. Pure function.
    """
    query_tokens = set(tokenize(query, max_tokens=20))
    scored = []
    for item in items:
        text = str(item.get(text_key) or "").lower()
        overlap = (sum(1 for t in query_tokens if t in text)
                   if query_tokens else 0)
        sem = 0.0
        if query_vec is not None:
            sem = cosine_sim(
                query_vec, decode_embedding(item.get(vec_key)))
        try:
            ts = float(item.get(time_key) or 0)
        except (TypeError, ValueError):
            ts = 0
        scored.append((sem * sem_weight + overlap * kw_weight, ts, item))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    # all-zero scores collapse to pure recency (legacy order)
    return [s[2] for s in scored]


def search_messages(channel_id, query, limit=5, exclude_ids=None):
    """Return up to `limit` past messages relevant to query.

    Each item: {msg_id, author_name, content, score}. Newest irrelevant
    chatter is excluded by ranking on match quality, not recency.
    exclude_ids (message ids already present in the live window) are
    filtered out so recall never re-injects context the prompt already
    has — including the message currently being answered.
    """
    tokens = tokenize(query)
    if not tokens:
        return []
    excluded = {str(x) for x in (exclude_ids or ())}
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
            for r in rows:
                if str(r["msg_id"]) in excluded:
                    continue
                d = dict(r)
                try:
                    d["score"] = -float(d.get("score", 0.0))
                except (TypeError, ValueError):
                    d["score"] = 0.0
                out.append(d)
                if len(out) >= int(limit):
                    break
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
            (str(channel_id), *params, int(limit) * 3),
        ).fetchall()
        return [dict(r) for r in rows
                if str(r["msg_id"]) not in excluded][:int(limit)]
    finally:
        con.close()


def select_recall(recalled, exclude_ids=None, limit=3):
    """Filter, order chronologically (oldest->newest), and cap recall.

    Chronological order keeps the excerpts readable as a timeline rather
    than a random-ranked list, so an answer can't be mislinked to the
    wrong question. exclude_ids drops rows already in the live window.
    Pure function (no I/O) so it is unit-testable.
    """
    excluded = {str(x) for x in (exclude_ids or ())}
    kept = [r for r in (recalled or [])
            if str(r.get("msg_id")) not in excluded]
    kept.sort(key=lambda r: _sort_key(r.get("msg_id")))
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 3
    return kept[:max(0, limit)]


def _sort_key(msg_id):
    try:
        return (0, int(msg_id))
    except (TypeError, ValueError):
        return (1, str(msg_id or ""))


def render_recalled(kept, parents=None, bot_id="", bot_name="Assistant",
                    names=None):
    """Render recall excerpts with explicit framing and reply linkage.

    Header states these are separate excerpts, not a continuous
    conversation. Any excerpt that is a reply carries its question
    inline so the two can't be detached. Mentions resolve via `names`
    (bot id -> @bot_name, unknown ids stay raw). Empty input -> ''.
    Pure function (no I/O).
    """
    if not kept:
        return ""
    parents = parents or {}
    lines = []
    for r in kept:
        content = mem_buffer.resolve_mentions(
            r.get("content", ""), names, bot_id, bot_name)
        label = mem_buffer.author_label(r)
        parent = parents.get(r.get("reply_to")) if r.get("reply_to") else None
        if parent is not None:
            ptext = mem_buffer.resolve_mentions(
                str(parent.get("content") or ""), names, bot_id, bot_name
            )[:mem_buffer.PARENT_TRUNCATE_CHARS]
            lines.append(
                f"- {label} (replying to "
                f"{mem_buffer.author_label(parent)}: {ptext}): {content}")
        else:
            lines.append(f"- {label}: {content}")
    return (
        "\nRelevant past messages (separate excerpts, not a continuous "
        "conversation):\n" + "\n".join(lines) + "\n"
    )


def find_referenced_users(messages, author_id, current_text="", max_users=2,
                          bot_id=""):
    """Find OTHER users relevant to the current message.

    Sources, in priority order:
      1. Discord mentions (<@id>) in the current message text,
      2. recent speakers whose display name appears in the current text,
      3. other recent human speakers, newest first.

    Returns (ordered_ids, names) where names maps id -> display name.
    The message author (author_id), the bot itself (bot_id), and
    assistant rows are always excluded. Pure function (no I/O) so it is
    unit-testable.
    """
    self_id = str(author_id or "")
    skip_ids = {self_id, str(bot_id or "")} - {""}
    ordered, names = [], {}

    def _add(uid, name=""):
        uid = str(uid or "").strip()
        if not uid or uid in skip_ids or uid in ordered:
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
        if not uid or uid in skip_ids:
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


def channel_engaged(messages, bot_id, lookback=30):
    """True if the bot was mentioned within the last `lookback` messages.

    Matches exact Discord mention syntax (<@id> / <@!id>) with a closing
    '>' so longer IDs can't false-positive on a bot-id prefix. Unknown
    or empty bot_id fails OPEN (returns True) so a momentarily missing
    client.user never halts all summarization. Pure function (no I/O).
    """
    bot_id = str(bot_id or "").strip()
    if not bot_id:
        return True
    if int(lookback) <= 0:
        return False
    pattern = re.compile(rf"<@!?{re.escape(bot_id)}>")
    for m in list(messages or [])[-int(lookback):]:
        if pattern.search(str(m.get("content") or "")):
            return True
    return False


def format_reply_context(parent, previous, parent_chars=500, ctx_chars=300,
                         bot_id="", bot_name="Assistant", names=None):
    """Render a replied-to message + preceding context for the prompt.

    parent: dict with author_name/content (DB row or fetched message data).
    previous: list of similar dicts, oldest->newest. Returns '' when the
    parent is missing/empty (caller keeps keyword recall instead).
    Mentions resolve via `names`. Pure function (no I/O) so it is
    unit-testable.
    """
    if not parent or not str(parent.get("content") or "").strip():
        return ""
    lines = [
        "Replied-to message:",
        f"{mem_buffer.author_label(parent)}: "
        f"{mem_buffer.resolve_mentions(str(parent.get('content') or ''), names, bot_id, bot_name)[:parent_chars]}",
    ]
    ctx = [m for m in (previous or [])
           if str(m.get("content") or "").strip()]
    if ctx:
        lines.append("Previous context:")
        lines.extend(
            f"{mem_buffer.author_label(m)}: "
            f"{mem_buffer.resolve_mentions(str(m.get('content') or ''), names, bot_id, bot_name)[:ctx_chars]}"
            for m in ctx
        )
    return "\n" + "\n".join(lines) + "\n"
