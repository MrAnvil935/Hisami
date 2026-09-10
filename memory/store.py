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
"""

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
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_channel ON summaries(channel_id, chunk_id);

CREATE TABLE IF NOT EXISTS user_facts (
    user_id    TEXT NOT NULL,
    fact       TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    updated_at REAL NOT NULL,
    PRIMARY KEY (user_id, fact)
);

CREATE TABLE IF NOT EXISTS channel_state (
    channel_id         TEXT PRIMARY KEY,
    last_summary_upto  INTEGER NOT NULL DEFAULT 0
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
        con.commit()
    finally:
        con.close()


# ---------------- messages ----------------

def add_message(channel_id, msg_id, author_id, author_name, role,
                content, reply_to=None, created_at=None):
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
    con = _connect()
    try:
        con.execute(
            """INSERT OR REPLACE INTO messages
               (channel_id, msg_id, author_id, author_name, role, content, reply_to, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (str(channel_id), msg_id, str(author_id), str(author_name),
             str(role), str(content), reply_to, created_at),
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
                      content, reply_to, created_at
               FROM messages WHERE channel_id=? ORDER BY msg_id DESC LIMIT ?""",
            (str(channel_id), int(limit)),
        ).fetchall()
    finally:
        con.close()
    # oldest -> newest
    return [dict(r) for r in reversed(rows)]


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

def add_summary(channel_id, summary, msg_from=0, msg_to=0):
    con = _connect()
    try:
        cur = con.execute(
            """INSERT INTO summaries (channel_id, summary, msg_from, msg_to, created_at)
               VALUES (?,?,?,?,?)""",
            (str(channel_id), str(summary), int(msg_from), int(msg_to), time.time()),
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


def search_summaries(channel_id, query, limit=2):
    """Keyword search over summaries with LIKE fallback (no extra deps)."""
    from .recall import tokenize

    tokens = tokenize(query)[:6]
    if not tokens:
        return get_latest_summaries(channel_id, limit)
    con = _connect()
    try:
        rows = con.execute(
            "SELECT chunk_id, summary, msg_from, msg_to, created_at"
            " FROM summaries WHERE channel_id=? ORDER BY chunk_id DESC LIMIT 50",
            (str(channel_id),),
        ).fetchall()
    finally:
        con.close()
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


# ---------------- user facts ----------------

def upsert_fact(user_id, fact, confidence=1.0):
    fact = str(fact).strip()
    if not fact or len(fact) > 500:
        return False
    con = _connect()
    try:
        con.execute(
            """INSERT INTO user_facts (user_id, fact, confidence, updated_at)
               VALUES (?,?,?,?)
               ON CONFLICT(user_id, fact) DO UPDATE SET
                 confidence=excluded.confidence, updated_at=excluded.updated_at""",
            (str(user_id), fact, float(confidence), time.time()),
        )
        con.commit()
        return True
    finally:
        con.close()


def get_facts(user_id, limit=5):
    con = _connect()
    try:
        rows = con.execute(
            """SELECT fact, confidence, updated_at FROM user_facts
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


def get_unsummarized(channel_id, chunk_size=40):
    """Return oldest unsummarized messages if a full chunk is ready, else []."""
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
    if len(rows) < int(chunk_size):
        return []
    return [dict(r) for r in rows]


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
