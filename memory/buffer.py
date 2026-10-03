"""Short-term buffer helpers: token-aware window over SQLite history."""

import re

from . import store
from . import vision as mem_vision

PARENT_TRUNCATE_CHARS = 300
STALE_REPLY_MARKER = "replying to an older message outside current context"

# Leading reply markers injected at render time. The model sometimes
# mimics them; strip_reply_markers removes them from generated output.
REPLY_MARKER_PREFIXES = ("(in reply to ", "(replying to ")


_MENTION_RE = re.compile(r"<@!?(\d+)>")


def resolve_mentions(text, names=None, bot_id="", bot_name="Assistant"):
    """Rewrite <@id> / <@!id> to @DisplayName for known users.

    The bot's own id becomes @bot_name. IDs absent from `names` are left
    raw so missing data is never papered over. Output uses plain '@Name'
    (no <@id> syntax), so it can't re-ping or trip mention detection.
    Pure function (no I/O).
    """
    if not text:
        return text if isinstance(text, str) else ""
    text = str(text)
    names = names or {}
    bot_id = str(bot_id or "").strip()

    def _sub(m):
        uid = m.group(1)
        if bot_id and uid == bot_id:
            return f"@{bot_name}"
        name = names.get(uid)
        if name:
            return f"@{name}"
        return m.group(0)

    return _MENTION_RE.sub(_sub, text)


def resolve_bot_mentions(text, bot_id, bot_name="Assistant"):
    """Back-compat wrapper: resolve only the bot's own mentions."""
    return resolve_mentions(text, None, bot_id, bot_name)


def strip_reply_markers(text):
    """Remove a leading reply-marker the model mimicked, if present.

    Looks for the marker terminator '):' (or a bare ')' when the model
    dropped the colon). This is more robust than a balanced-paren scan,
    which a stray ')' inside quoted parent text would cut short. Loops
    to clear stacked markers. Returns the input unchanged when there is
    no leading marker, and never returns an empty string (falls back to
    the original). Pure function (no I/O).
    """
    original = text if isinstance(text, str) else str(text)
    current = original
    while True:
        lead = current.lstrip()
        head = lead.lower()
        if not any(head.startswith(p) for p in REPLY_MARKER_PREFIXES):
            break
        close = lead.find("):")
        if close != -1:
            rest = lead[close + 2:]
        elif ")" in lead:
            rest = lead[lead.find(")") + 1:]
        else:
            break
        rest = rest.lstrip()
        if not rest:
            break
        current = rest
    return current or original


def estimate_tokens(text: str) -> int:
    # ~4 chars per token heuristic; avoids new deps like tiktoken.
    if not text:
        return 0
    return max(1, len(text) // 4)


def message_tokens(msg: dict) -> int:
    return estimate_tokens(str(msg.get("author_name", ""))) \
        + estimate_tokens(str(msg.get("content", ""))) + 4  # role/format overhead


def _row_time(msg):
    try:
        return float(msg.get("created_at"))
    except (TypeError, ValueError):
        return None


def merge_fragments(rows, max_gap_s=180, break_ids=None):
    """Merge consecutive same-author user fragments into logical messages.

    A split-up thought ("hey", "wait", "ok so") otherwise burns one
    window slot per fragment. Rows oldest->newest in, merged units out.
    Only role == 'user' rows merge, and only when author_id matches and
    the gap between them is 0..max_gap_s seconds (bad/missing timestamps
    never merge). Assistant turns are never fused. max_gap_s <= 0
    disables merging entirely.

    Barriers that always stay their own unit (never merged with a
    neighbour): break_ids (e.g. the live ping being answered) and any
    row that is itself a reply (reply_to set) — a reply is a targeted
    response, so fusing it would blur what it was answering. Pure
    function (no I/O).
    """
    try:
        gap_limit = float(max_gap_s)
    except (TypeError, ValueError):
        gap_limit = 180.0
    if gap_limit <= 0:
        return [dict(r) for r in (rows or [])]
    breaks = {str(b) for b in (break_ids or ())}

    merged = []
    sealed_prev = False
    for row in rows or []:
        row = dict(row)
        row.setdefault("ids", [row.get("msg_id")])
        is_break = str(row.get("msg_id")) in breaks
        is_reply = row.get("reply_to") is not None
        prev = merged[-1] if merged else None
        same_author = (
            prev is not None and not sealed_prev
            and not is_break and not is_reply
            and prev.get("reply_to") is None
            and prev.get("role") == "user" and row.get("role") == "user"
            and str(prev.get("author_id") or "").strip()
            and str(prev.get("author_id")) == str(row.get("author_id"))
        )
        close = False
        if same_author:
            t_prev, t_row = _row_time(prev), _row_time(row)
            close = (t_prev is not None and t_row is not None
                     and 0 <= (t_row - t_prev) <= gap_limit)

        if close:
            joined = dict(row)
            joined["content"] = (
                f"{prev.get('content', '')}\n{row.get('content', '')}".strip()
            )
            joined["attachments"] = (
                list(prev.get("attachments") or [])
                + list(row.get("attachments") or [])
            )
            joined["reply_to"] = (prev.get("reply_to")
                                  if prev.get("reply_to") is not None
                                  else row.get("reply_to"))
            joined["ids"] = (list(prev.get("ids") or [])
                             + list(row.get("ids") or []))
            joined["msg_id"] = row.get("msg_id")
            merged[-1] = joined
            sealed_prev = False
            continue
        merged.append(row)
        sealed_prev = is_break or is_reply
    return merged


def load_window(channel_id, budget_tokens=1500, max_messages=30,
                fetch_limit=120, max_gap_s=180, break_ids=None):
    """Load most recent messages that fit into the token budget.

    Consecutive same-author user fragments are merged first, so a single
    thought split across many messages costs one window slot. break_ids
    (e.g. the live message) stay standalone. Returns oldest->newest list.
    Always keeps at least the newest message.
    """
    recent = store.get_recent(channel_id, limit=fetch_limit)
    if not recent:
        return []
    units = merge_fragments(recent, max_gap_s, break_ids)
    # newest-first walk, then reverse
    picked = []
    used = 0
    for msg in reversed(units):
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


def format_history_line(msg, parent=None, bot_id="", bot_name="Assistant",
                        names=None):
    """Render one window message for the prompt.

    - plain message: 'author [display]: content [markers]'
    - reply with resolvable parent: 'author [display] (replying to
      X [Y]: <truncated parent>): content [markers]'
    - reply with unresolvable parent: 'author [display] (replying to an
      older message outside current context): content [markers]'.

    Mentions in content and parent snippets resolve via `names`
    (author_id -> display name); the bot's own id renders as @bot_name
    and unknown ids stay raw. Parent content is truncated (auxiliary
    context, not primary). Media markers are always preserved. Pure
    function (no I/O).
    """
    markers = mem_vision.format_markers(msg.get("attachments"))
    content = resolve_mentions(msg.get("content", ""), names, bot_id, bot_name)
    text = f"{author_label(msg)}: {content}"
    if msg.get("reply_to"):
        if parent is not None:
            ptext = resolve_mentions(
                str(parent.get("content", "")), names, bot_id, bot_name
            )[:PARENT_TRUNCATE_CHARS]
            text = (
                f"{author_label(msg)} "
                f"(replying to {author_label(parent)}: {ptext}): "
                f"{content}"
            )
        else:
            text = (
                f"{author_label(msg)} "
                f"({STALE_REPLY_MARKER}): "
                f"{content}"
            )
    if markers:
        text += markers
    return text


def format_assistant_line(msg, parent=None, bot_id="", bot_name="Assistant",
                          names=None):
    """Render a bot message as an assistant turn (no author prefix).

    The role already conveys the speaker, so only content + media
    markers are emitted — but a reply context is preserved (quoted, like
    user turns) so reply threads don't flatten into standalone
    statements. Mentions resolve like format_history_line. Lead marker
    mimicry in generated output is handled separately by
    strip_reply_markers. Pure function (no I/O).
    """
    markers = mem_vision.format_markers(msg.get("attachments"))
    text = resolve_mentions(msg.get("content", ""), names, bot_id, bot_name)
    if msg.get("reply_to"):
        if parent is not None:
            ptext = resolve_mentions(
                str(parent.get("content", "")), names, bot_id, bot_name
            )[:PARENT_TRUNCATE_CHARS]
            text = (
                f"(replying to {author_label(parent)}: {ptext}): {text}"
            )
        else:
            text = f"({STALE_REPLY_MARKER}): {text}"
    if markers:
        text += markers
    return text


def format_turns(history, parent_map=None, current_message_id=None,
                 username="", user_message="", display_name="",
                 bot_id="", bot_name="Assistant", names=None):
    """Build role-tagged chat turns from the window (oldest -> newest).

    User turns keep 'author [display]: ...' labels (needed to tell
    speakers apart in multi-user channels); assistant turns carry only
    their content. The row whose id == current_message_id is rendered
    from its stored row (reply_to/attachments preserved) with only the
    content swapped for the caller's cleaned text, then appended as the
    final user turn — so the live message (including reply-pings) is
    never duplicated and never stripped of context.

    Reply parents render with the quoted form for both roles. Pure
    function (no I/O).
    """
    parent_map = parent_map or {}
    history = history or []

    def _is_current(m):
        if current_message_id is None:
            return False
        return current_message_id in (m.get("ids") or [m.get("id")])

    turns = []
    for m in history:
        if _is_current(m):
            continue
        parent = (parent_map.get(m.get("reply_to"))
                  if m.get("reply_to") else None)
        if m.get("role") == "assistant":
            turns.append({"role": "assistant",
                          "content": format_assistant_line(
                              m, parent, bot_id, bot_name, names)})
        else:
            turns.append({"role": "user",
                          "content": format_history_line(
                              m, parent, bot_id, bot_name, names)})
    current = next((m for m in history if _is_current(m)), None)
    if current is not None:
        row = dict(current)
        row["content"] = user_message
        parent = (parent_map.get(row.get("reply_to"))
                  if row.get("reply_to") else None)
        turns.append({"role": "user",
                      "content": format_history_line(
                          row, parent, bot_id, bot_name, names)})
    else:
        label = author_label(
            {"author": username, "display_name": display_name})
        turns.append({"role": "user",
                      "content": f"{label}: "
                                 f"{resolve_mentions(user_message, names, bot_id, bot_name)}"})
    return turns
