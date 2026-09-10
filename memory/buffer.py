"""Short-term buffer helpers: token-aware window over SQLite history."""

from . import store


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
