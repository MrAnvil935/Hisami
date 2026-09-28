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


def format_assistant_line(msg, parent=None):
    """Render a bot message as an assistant turn (no author prefix).

    The role already conveys the speaker, so only content + media
    markers are emitted — but a reply context is preserved: when the
    row carries reply_to and the parent resolves, it renders
    '(replying to X: ...)' just like user turns, so reply threads
    don't flatten into standalone statements. Pure function (no I/O).
    """
    markers = mem_vision.format_markers(msg.get("attachments"))
    text = str(msg.get("content", ""))
    if msg.get("reply_to"):
        if parent is not None:
            ptext = str(parent.get("content", ""))[:PARENT_TRUNCATE_CHARS]
            text = (
                f"(replying to {author_label(parent)}: {ptext}): {text}"
            )
        else:
            text = f"({STALE_REPLY_MARKER}): {text}"
    if markers:
        text += markers
    return text


def format_turns(history, parent_map=None, current_message_id=None,
                 username="", user_message="", display_name=""):
    """Build role-tagged chat turns from the window (oldest -> newest).

    User turns keep 'author [display]: ...' labels (needed to tell
    speakers apart in multi-user channels); assistant turns carry only
    their content. The row whose id == current_message_id is rendered
    from its stored row (reply_to/attachments preserved) with only the
    content swapped for the caller's cleaned text, then appended as the
    final user turn — so the live message (including reply-pings) is
    never duplicated and never stripped of context. Pure function
    (no I/O).
    """
    parent_map = parent_map or {}
    turns = []
    for m in history or []:
        if (current_message_id is not None
                and m.get("id") == current_message_id):
            continue
        parent = (parent_map.get(m.get("reply_to"))
                  if m.get("reply_to") else None)
        if m.get("role") == "assistant":
            turns.append({"role": "assistant",
                          "content": format_assistant_line(m, parent)})
        else:
            turns.append({"role": "user",
                          "content": format_history_line(m, parent)})
    current = next(
        (m for m in history or []
         if current_message_id is not None
         and m.get("id") == current_message_id),
        None,
    )
    if current is not None:
        row = dict(current)
        row["content"] = user_message
        parent = (parent_map.get(row.get("reply_to"))
                  if row.get("reply_to") else None)
        turns.append({"role": "user",
                      "content": format_history_line(row, parent)})
    else:
        label = author_label(
            {"author": username, "display_name": display_name})
        turns.append({"role": "user",
                      "content": f"{label}: {user_message}"})
    return turns
