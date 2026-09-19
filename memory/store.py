"""SQLite-backed persistent memory for Hisami.

All functions are blocking/sync — call them from async code with
asyncio.to_thread(). Each helper opens its own short-lived connection
so it is safe to use from multiple threads (WAL mode).

Tables:
  messages      - raw short-term buffer (pruned per channel)
  messages_fts  - FTS5 index for keyword recall (manually synced)
  summaries     - long-term episodic memory per channel
  user_facts    - durable per-user facts
  channel_state - summarizer bookkeeping
  image_cache   - vision descriptions keyed by image source URL
"""

import json
import sqlite3
import time

DB_PATH = "memory.db"

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS messages (
    channel_id  TEXT NOT NULL,
    msg_id      INTEGER NOT NULL,
    author_id   TEXT NOT NULL DEFAULT '',
    author_name TEXT NOT NULL DEFAULT '',
    role        TEXT NOT NULL DEFAULT 'user',
    content     TEXT NOT NULL DEFAULT '',
    reply_to    INTEGER,
    created_at  REAL NOT NULL,
    attachments TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (channel_id, msg_id)
);
CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel_id, msg_id);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, author_name, channel_id UNINDEXED, msg_id UNINDEXED
);

CREATE TABLE IF NOT EXISTS summaries (
    chunk_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id TEXT NOT NULL,
    summary    TEXT NOT NULL,
    msg_from   INTEGER NOT NULL DEFAULT 0,
    msg_to     INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    embedding  BLOB
);
CREATE INDEX IF NOT EXISTS idx_summaries_channel ON summaries(channel_id, chunk_id);

CREATE TABLE IF NOT EXISTS user_facts (
    user_id    TEXT NOT NULL,
    fact       TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    updated_at REAL NOT NULL,
    embedding  BLOB,
    PRIMARY KEY (user_id, fact)
);

CREATE TABLE IF NOT EXISTS channel_state (
    channel_id         TEXT PRIMARY KEY,
    last_summary_upto  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS image_cache (
    key         TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    created_at  REAL NOT NULL
);
"""


def configure(path: str):
    global DB_PATH
    DB_PATH = path


def _connect():
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def init_db(path: str | None = None):
    if path:
        configure(path)
    con = _connect()
    try:
        con.executescript(SCHEMA)
        # migrate older databases in place
        cols = [r[1] for r in con.execute("PRAGMA table_info(messages)")]
        if "attachments" not in cols:
            con.execute("ALTER TABLE messages ADD COLUMN attachments TEXT NOT NULL DEFAULT '[]'")
        for table in ("summaries", "user_facts"):
            tcols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
            if "embedding" not in tcols:
                con.execute(f"ALTER TABLE {table} ADD COLUMN embedding BLOB")
        con.commit()
    finally:
        con.close()


# ---------------- messages ----------------

def add_message(channel_id, msg_id, author_id, author_name, role,
                content, reply_to=None, created_at=None, attachments=None):
    created_at = created_at if created_at is not None else time.time()
    # msg_id may be str (discord snowflake) — store as int when possible
    try:
        msg_id = int(msg_id)
    except (TypeError, ValueError):
        msg_id = abs(hash((str(channel_id), str(msg_id)))) % (2 ** 62)
    if reply_to is not None:
        try:
            reply_to = int(reply_to)
        except (TypeError, ValueError):
            reply_to = None
    try:
        attachments_json = json.dumps([
            {"kind": a.get("kind"), "name": a.get("name")}
            for a in (attachments or []) if a.get("kind") in ("image", "video")
        ], ensure_ascii=False)
    except Exception:
        attachments_json = "[]"
    con = _connect()
    try:
        con.execute(
            """INSERT OR REPLACE INTO messages
               (channel_id, msg_id, author_id, author_name, role, content, reply_to, created_at, attachments)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (str(channel_id), msg_id, str(author_id), str(author_name),
             str(role), str(content), reply_to, created_at, attachments_json),
        )
        con.execute(
            "DELETE FROM messages_fts WHERE channel_id=? AND msg_id=?",
            (str(channel_id), msg_id),
        )
        con.execute(
            "INSERT INTO messages_fts (content, author_name, channel_id, msg_id)"
            " VALUES (?,?,?,?)",
            (str(content), str(author_name), str(channel_id), msg_id),
        )
        con.commit()
    finally:
        con.close()


def get_recent(channel_id, limit=50):
    con = _connect()
    try:
        rows = con.execute(
            """SELECT channel_id, msg_id, author_id, author_name, role,
                      content, reply_to, created_at, attachments
               FROM messages WHERE channel_id=? ORDER BY msg_id DESC LIMIT ?""",
            (str(channel_id), int(limit)),
        ).fetchall()
    finally:
        con.close()
    # oldest -> newest
    out = []
    for r in reversed(rows):
        d = dict(r)
        try:
            d["attachments"] = json.loads(d.get("attachments") or "[]")
        except Exception:
            d["attachments"] = []
        out.append(d)
    return out


def get_message_count(channel_id):
    con = _connect()
    try:
        row = con.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE channel_id=?",
            (str(channel_id),),
        ).fetchone()
        return row["c"]
    finally:
        con.close()


def get_messages_by_ids(channel_id, ids):
    """Bulk-fetch messages by id. Returns {msg_id: row dict}.

    Used to recover reply parents that fell outside the prompt window.
    Empty input short-circuits without touching the DB.
    """
    want = set()
    for i in ids or []:
        try:
            want.add(int(i))
        except (TypeError, ValueError):
            continue
    if not want:
        return {}
    q = ",".join("?" for _ in want)
    con = _connect()
    try:
        rows = con.execute(
            f"""SELECT msg_id, author_id, author_name, role,
                       content, reply_to, created_at, attachments
                FROM messages WHERE channel_id=? AND msg_id IN ({q})""",
            (str(channel_id), *want),
        ).fetchall()
    finally:
        con.close()
    out = {}
    for r in rows:
        d = dict(r)
        try:
            d["attachments"] = json.loads(d.get("attachments") or "[]")
        except Exception:
            d["attachments"] = []
        out[d["msg_id"]] = d
    return out


def get_messages_before(channel_id, msg_id, limit=3):
    """Messages strictly before msg_id, oldest->newest, up to limit.

    Supplies the 'little bit of previous context' for replied-to message
    blocks. Attachments parsed like get_recent. Empty on bad input.
    """
    try:
        anchor = int(msg_id)
    except (TypeError, ValueError):
        return []
    con = _connect()
    try:
        rows = con.execute(
            """SELECT msg_id, author_id, author_name, role,
                      content, reply_to, created_at, attachments
               FROM messages WHERE channel_id=? AND msg_id < ?
               ORDER BY msg_id DESC LIMIT ?""",
            (str(channel_id), anchor, int(limit)),
        ).fetchall()
    finally:
        con.close()
    out = []
    for r in reversed(rows):
        d = dict(r)
        try:
            d["attachments"] = json.loads(d.get("attachments") or "[]")
        except Exception:
            d["attachments"] = []
        out.append(d)
    return out


def get_max_msg_id(channel_id):
    con = _connect()
    try:
        row = con.execute(
            "SELECT MAX(msg_id) AS m FROM messages WHERE channel_id=?",
            (str(channel_id),),
        ).fetchone()
        return row["m"] if row and row["m"] is not None else 0
    finally:
        con.close()


def prune_channel(channel_id, keep=200):
    """Keep only the newest `keep` messages. Returns rows deleted."""
    con = _connect()
    try:
        con.execute(
            """DELETE FROM messages WHERE channel_id=?
               AND msg_id NOT IN (
                   SELECT msg_id FROM messages WHERE channel_id=?
                   ORDER BY msg_id DESC LIMIT ?
               )""",
            (str(channel_id), str(channel_id), int(keep)),
        )
        # sync FTS: drop orphan entries
        con.execute(
            """DELETE FROM messages_fts WHERE channel_id=?
               AND msg_id NOT IN (
                   SELECT msg_id FROM messages WHERE channel_id=?
               )""",
            (str(channel_id), str(channel_id)),
        )
        deleted = con.total_changes
        con.commit()
        return deleted
    finally:
        con.close()


def clear_recent(channel_id, limit=50):
    """Delete the N most recent messages (lobotomy: unstick the model).

    Summaries and user_facts are intentionally left untouched.
    Returns number of rows removed.
    """
    con = _connect()
    try:
        rows = con.execute(
            """SELECT msg_id FROM messages WHERE channel_id=?
               ORDER BY msg_id DESC LIMIT ?""",
            (str(channel_id), int(limit)),
        ).fetchall()
        ids = [r["msg_id"] for r in rows]
        if not ids:
            return 0
        q = ",".join("?" for _ in ids)
        con.execute(
            f"DELETE FROM messages WHERE channel_id=? AND msg_id IN ({q})",
            (str(channel_id), *ids),
        )
        con.execute(
            f"DELETE FROM messages_fts WHERE channel_id=? AND msg_id IN ({q})",
            (str(channel_id), *ids),
        )
        con.commit()
        return len(ids)
    finally:
        con.close()


# ---------------- summaries ----------------

def add_summary(channel_id, summary, msg_from=0, msg_to=0, embedding=None):
    con = _connect()
    try:
        cur = con.execute(
            """INSERT INTO summaries (channel_id, summary, msg_from, msg_to, created_at, embedding)
               VALUES (?,?,?,?,?,?)""",
            (str(channel_id), str(summary), int(msg_from), int(msg_to), time.time(),
             bytes(embedding) if embedding is not None else None),
        )
        con.commit()
        return cur.lastrowid
    finally:
        con.close()


def get_latest_summaries(channel_id, limit=2):
    con = _connect()
    try:
        rows = con.execute(
            """SELECT chunk_id, summary, msg_from, msg_to, created_at
               FROM summaries WHERE channel_id=? ORDER BY chunk_id DESC LIMIT ?""",
            (str(channel_id), int(limit)),
        ).fetchall()
    finally:
        con.close()
    return [dict(r) for r in reversed(rows)]


def get_summary_count(channel_id=None):
    con = _connect()
    try:
        if channel_id is None:
            row = con.execute("SELECT COUNT(*) AS c FROM summaries").fetchone()
        else:
            row = con.execute(
                "SELECT COUNT(*) AS c FROM summaries WHERE channel_id=?",
                (str(channel_id),),
            ).fetchone()
        return row["c"]
    finally:
        con.close()


def search_summaries(channel_id, query, limit=2, query_vec=None):
    """Keyword search over summaries, upgraded to hybrid when query_vec given.

    Without a query vector this is the legacy LIKE scorer. With one,
    candidates rank by semantic similarity + keyword boost (same weights
    as style-example retrieval); rows lacking embeddings score keywords
    only. Falls back to latest summaries when nothing matches.
    """
    from .recall import rank_hybrid, tokenize

    tokens = tokenize(query)[:6]
    if not tokens and query_vec is None:
        return get_latest_summaries(channel_id, limit)
    con = _connect()
    try:
        rows = con.execute(
            "SELECT chunk_id, summary, msg_from, msg_to, created_at, embedding"
            " FROM summaries WHERE channel_id=? ORDER BY chunk_id DESC LIMIT 200",
            (str(channel_id),),
        ).fetchall()
    finally:
        con.close()
    if query_vec is None:
        scored = []
        for r in rows:
            text = r["summary"].lower()
            score = sum(1 for t in tokens if t in text)
            if score:
                scored.append((score, r["chunk_id"], dict(r)))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        out = [s[2] for s in scored[:limit]]
        if not out:
            return get_latest_summaries(channel_id, min(limit, 1))
        return out
    ranked = rank_hybrid(
        [dict(r) for r in rows], query, query_vec,
        text_key="summary", time_key="created_at", vec_key="embedding")
    out = ranked[:limit]
    if not any(_has_signal(r, query, query_vec) for r in out):
        return get_latest_summaries(channel_id, min(limit, 1))
    return out


def _has_signal(row, query, query_vec, text_key="summary"):
    from .recall import cosine_sim, decode_embedding, tokenize
    # 0.1 clears random-vector noise (~N(0, 0.036) at 768 dims) while
    # genuinely related texts typically score 0.3+.
    if cosine_sim(query_vec, decode_embedding(row.get("embedding"))) > 0.1:
        return True
    tokens = tokenize(query)[:6]
    text = str(row.get(text_key) or "").lower()
    return any(t in text for t in tokens)


# ---------------- user facts ----------------

def upsert_fact(user_id, fact, confidence=1.0, embedding=None):
    fact = str(fact).strip()
    if not fact or len(fact) > 500:
        return False
    con = _connect()
    try:
        con.execute(
            """INSERT INTO user_facts (user_id, fact, confidence, updated_at, embedding)
               VALUES (?,?,?,?,?)
               ON CONFLICT(user_id, fact) DO UPDATE SET
                 confidence=excluded.confidence, updated_at=excluded.updated_at,
                 embedding=COALESCE(excluded.embedding, user_facts.embedding)""",
            (str(user_id), fact, float(confidence), time.time(),
             bytes(embedding) if embedding is not None else None),
        )
        con.commit()
        return True
    finally:
        con.close()


def get_facts(user_id, limit=5, query=None, query_vec=None):
    con = _connect()
    try:
        if query and query_vec is not None:
            # hybrid path: semantic + keyword over a bounded recency pool
            from .recall import rank_hybrid
            rows = con.execute(
                """SELECT fact, confidence, updated_at, embedding FROM user_facts
                   WHERE user_id=? ORDER BY confidence DESC, updated_at DESC
                   LIMIT 100""",
                (str(user_id),),
            ).fetchall()
            return rank_hybrid(
                [dict(r) for r in rows], query, query_vec,
                text_key="fact", time_key="updated_at",
                vec_key="embedding")[:int(limit)]
        if query:
            # relevance path: rank a bounded recency pool by keyword
            # overlap so topical (even old) facts win; pure recency
            # tiebreak preserves legacy order when nothing matches.
            from .recall import rank_by_overlap
            rows = con.execute(
                """SELECT fact, confidence, updated_at FROM user_facts
                   WHERE user_id=? ORDER BY confidence DESC, updated_at DESC
                   LIMIT 100""",
                (str(user_id),),
            ).fetchall()
            ranked = rank_by_overlap(
                [dict(r) for r in rows], query,
                text_key="fact", time_key="updated_at")
            return ranked[:int(limit)]
        rows = con.execute(
            """SELECT fact, confidence, updated_at, embedding FROM user_facts
               WHERE user_id=? ORDER BY confidence DESC, updated_at DESC LIMIT ?""",
            (str(user_id), int(limit)),
        ).fetchall()
    finally:
        con.close()
    return [dict(r) for r in rows]


def get_fact_count(user_id=None):
    con = _connect()
    try:
        if user_id is None:
            row = con.execute("SELECT COUNT(*) AS c FROM user_facts").fetchone()
        else:
            row = con.execute(
                "SELECT COUNT(*) AS c FROM user_facts WHERE user_id=?",
                (str(user_id),),
            ).fetchone()
        return row["c"]
    finally:
        con.close()


# ---------------- image cache ----------------

def get_image_desc(key, ttl_seconds=7 * 86400):
    """Return cached vision description, or '' on miss/expiry."""
    con = _connect()
    try:
        row = con.execute(
            "SELECT description, created_at FROM image_cache WHERE key=?",
            (str(key),),
        ).fetchone()
    finally:
        con.close()
    if not row:
        return ""
    if time.time() - row["created_at"] > ttl_seconds:
        return ""
    return row["description"] or ""


def set_image_desc(key, description, max_rows=200):
    con = _connect()
    try:
        con.execute(
            """INSERT INTO image_cache (key, description, created_at)
               VALUES (?,?,?)
               ON CONFLICT(key) DO UPDATE SET
                 description=excluded.description, created_at=excluded.created_at""",
            (str(key), str(description), time.time()),
        )
        # cap size: drop oldest beyond max_rows
        con.execute(
            """DELETE FROM image_cache WHERE key NOT IN (
                   SELECT key FROM image_cache ORDER BY created_at DESC LIMIT ?
               )""",
            (int(max_rows),),
        )
        con.commit()
    finally:
        con.close()


# ---------------- channel state ----------------

def get_last_summary_upto(channel_id):
    con = _connect()
    try:
        row = con.execute(
            "SELECT last_summary_upto FROM channel_state WHERE channel_id=?",
            (str(channel_id),),
        ).fetchone()
        return row["last_summary_upto"] if row else 0
    finally:
        con.close()


def set_last_summary_upto(channel_id, msg_id):
    con = _connect()
    try:
        con.execute(
            """INSERT INTO channel_state (channel_id, last_summary_upto)
               VALUES (?,?)
               ON CONFLICT(channel_id) DO UPDATE SET
                 last_summary_upto=excluded.last_summary_upto""",
            (str(channel_id), int(msg_id)),
        )
        con.commit()
    finally:
        con.close()


# ---------------- embedding backfill ----------------

def get_unembedded_summaries(limit=5):
    """Oldest summaries lacking vectors (pre-migration or Ollama was down)."""
    con = _connect()
    try:
        rows = con.execute(
            """SELECT chunk_id, summary FROM summaries
               WHERE embedding IS NULL ORDER BY chunk_id ASC LIMIT ?""",
            (int(limit),),
        ).fetchall()
    finally:
        con.close()
    return [dict(r) for r in rows]


def get_unembedded_facts(limit=5):
    con = _connect()
    try:
        rows = con.execute(
            """SELECT user_id, fact FROM user_facts
               WHERE embedding IS NULL ORDER BY updated_at ASC LIMIT ?""",
            (int(limit),),
        ).fetchall()
    finally:
        con.close()
    return [dict(r) for r in rows]


def set_summary_embedding(chunk_id, embedding):
    con = _connect()
    try:
        con.execute(
            "UPDATE summaries SET embedding=? WHERE chunk_id=?",
            (bytes(embedding) if embedding is not None else None,
             int(chunk_id)),
        )
        con.commit()
    finally:
        con.close()


def set_fact_embedding(user_id, fact, embedding):
    con = _connect()
    try:
        con.execute(
            "UPDATE user_facts SET embedding=? WHERE user_id=? AND fact=?",
            (bytes(embedding) if embedding is not None else None,
             str(user_id), str(fact)),
        )
        con.commit()
    finally:
        con.close()


def _estimate_tokens(text):
    # mirrors buffer.estimate_tokens (kept local to avoid a circular import)
    if not text:
        return 0
    return max(1, len(text) // 4)


def get_unsummarized(channel_id, chunk_size=30, chunk_tokens=None, min_msgs=10):
    """Return oldest unsummarized messages when a chunk is ready, else [].

    Ready means either:
      - count reached chunk_size, or
      - estimated tokens reached chunk_tokens (with at least min_msgs),
        so long verbose messages trigger summarization early instead of
        bloating the raw window.
    """
    upto = get_last_summary_upto(channel_id)
    con = _connect()
    try:
        rows = con.execute(
            """SELECT msg_id, author_id, author_name, role, content FROM messages
               WHERE channel_id=? AND msg_id > ? ORDER BY msg_id ASC LIMIT ?""",
            (str(channel_id), int(upto), int(chunk_size)),
        ).fetchall()
    finally:
        con.close()
    rows = [dict(r) for r in rows]
    if len(rows) >= int(chunk_size):
        return rows
    if (chunk_tokens and len(rows) >= int(min_msgs)
            and sum(_estimate_tokens(r.get("content")) for r in rows)
            >= int(chunk_tokens)):
        return rows
    return []


# ---------------- stats ----------------

def stats(channel_id=None):
    import os

    con = _connect()
    try:
        if channel_id is None:
            m = con.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"]
            s = con.execute("SELECT COUNT(*) AS c FROM summaries").fetchone()["c"]
            f = con.execute("SELECT COUNT(*) AS c FROM user_facts").fetchone()["c"]
        else:
            m = con.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE channel_id=?",
                (str(channel_id),),
            ).fetchone()["c"]
            s = con.execute(
                "SELECT COUNT(*) AS c FROM summaries WHERE channel_id=?",
                (str(channel_id),),
            ).fetchone()["c"]
            f = con.execute("SELECT COUNT(*) AS c FROM user_facts").fetchone()["c"]
    finally:
        con.close()
    try:
        size = os.path.getsize(DB_PATH)
    except OSError:
        size = 0
    return {"messages": m, "summaries": s, "facts": f, "db_bytes": size}
