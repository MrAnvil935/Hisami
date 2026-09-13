"""Short-term buffer helpers: token-aware window over SQLite history."""

from . import store
from . import vision as mem_vision

PARENT_TRUNCATE_CHARS = 300
STALE_REPLY_MARKER = "replying to an older message outside current context"


def estimate_tokens(text: str) -> int:
    # ~4 chars per token heuristic; avoids new deps like tiktoken.
    if not text:
        return 0
    return max(1, len(text) // 4)


def message_tokens(msg: dict) -> int:
    return estimate_tokens(str(msg.get("author_name", ""))) \
        + estimate_tokens(str(msg.get("content", ""))) + 4  # role/format overhead


def load_window(channel_id, budget_tokens=1500, max_messages=30, fetch_limit=120):
    """Load most recent messages that fit into the token budget.

    Returns oldest->newest list. Always keeps at least the newest message.
    """
    recent = store.get_recent(channel_id, limit=fetch_limit)
    if not recent:
        return []
    # newest-first walk, then reverse
    picked = []
    used = 0
    for msg in reversed(recent):
        t = message_tokens(msg)
        if picked and (used + t > budget_tokens or len(picked) >= max_messages):
            break
        picked.append(msg)
        used += t
    picked.reverse()
    return picked


def format_history_line(msg, parent=None):
    """Render one window message for the prompt (legacy compat shape).

    - plain message: 'author: content [markers]'
    - reply with resolvable parent: 'author (replying to X: Y): content [markers]'
    - reply with missing parent: 'author (replying to an older message
      outside current context): content [markers]' — honest about the gap
      instead of silently flattening the reply into a standalone statement.

    Parent content is truncated (auxiliary context, not primary).
    Media markers are always preserved. Pure function (no I/O).
    """
    markers = mem_vision.format_markers(msg.get("attachments"))
    text = f"{msg.get('author', '?')}: {msg.get('content', '')}"
    if msg.get("reply_to"):
        if parent is not None:
            ptext = str(parent.get("content", ""))[:PARENT_TRUNCATE_CHARS]
            text = (
                f"{msg.get('author', '?')} "
                f"(replying to {parent.get('author', '?')}: {ptext}): "
                f"{msg.get('content', '')}"
            )
        else:
            text = (
                f"{msg.get('author', '?')} "
                f"({STALE_REPLY_MARKER}): "
                f"{msg.get('content', '')}"
            )
    if markers:
        text += markers
    return text
