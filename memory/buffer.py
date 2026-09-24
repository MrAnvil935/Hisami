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


def author_label(msg):
    """Prompt label for a message author: 'name [display]'.

    Accepts both the window shape (author/display_name) and DB rows
    (author_name/display_name). The bracket is skipped when the display
    name is missing or identical to the account name, so plain
    'alice: ...' lines stay noise-free. Pure function (no I/O).
    """
    name = str(msg.get("author", msg.get("author_name", "?")) or "?")
    display = str(msg.get("display_name") or "")
    if display and display != name:
        return f"{name} [{display}]"
    return name


def format_history_line(msg, parent=None):
    """Render one window message for the prompt (legacy compat shape).

    - plain message: 'author [display]: content [markers]'
    - reply with resolvable parent: 'author [display] (replying to X [Y]: ...)'
    - reply with missing parent: 'author [display] (replying to an older
      message outside current context): content [markers]' — honest about
      the gap instead of silently flattening the reply into a standalone
      statement.

    Parent content is truncated (auxiliary context, not primary).
    Media markers are always preserved. Pure function (no I/O).
    """
    markers = mem_vision.format_markers(msg.get("attachments"))
    text = f"{author_label(msg)}: {msg.get('content', '')}"
    if msg.get("reply_to"):
        if parent is not None:
            ptext = str(parent.get("content", ""))[:PARENT_TRUNCATE_CHARS]
            text = (
                f"{author_label(msg)} "
                f"(replying to {author_label(parent)}: {ptext}): "
                f"{msg.get('content', '')}"
            )
        else:
            text = (
                f"{author_label(msg)} "
                f"({STALE_REPLY_MARKER}): "
                f"{msg.get('content', '')}"
            )
    if markers:
        text += markers
    return text
