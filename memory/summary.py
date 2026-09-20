"""Long-term episodic memory: chunk summarization helpers.

The actual LLM call lives in bot.py (reuses generate_reply with the
local-Ollama-first chain). This module only builds prompts and parses
results so it stays testable without Discord or network.
"""

SUMMARIZER_SYSTEM = (
    "You summarize Discord chat for a bot's long-term memory. "
    "Be concise and factual. Capture: topics discussed, decisions, "
    "open questions, user preferences stated, and any facts worth "
    "remembering. No roleplay, no extra commentary."
)


def build_summary_prompt(messages, max_chars=500):
    lines = []
    for m in messages:
        author = m.get("author_name", "?")
        content = (m.get("content") or "")[:max_chars]
        lines.append(f"{author}: {content}")
    convo = "\n".join(lines)
    return (
        "Summarize this Discord conversation chunk in 5-10 bullet points. "
        "Keep names attached to opinions/facts.\n\n"
        f"{convo}"
    )


def build_multi_fact_prompt(messages, max_chars=300):
    """Prompt extracting per-user facts for ALL human speakers in a chunk.

    Bot rows (role == 'assistant') are excluded by the caller convention:
    this builder skips them defensively as well. Returns prompt text
    expecting a JSON object: {"DisplayName": ["fact", ...]}.
    """
    lines = []
    for m in messages:
        if m.get("role") == "assistant":
            continue
        author = m.get("author_name", "?")
        content = (m.get("content") or "")[:max_chars]
        lines.append(f"{author}: {content}")
    convo = "\n".join(lines)
    return (
        "Extract durable facts about EACH person in this chat. "
        'Return ONLY a JSON object mapping display name to facts, e.g. '
        '{"alice": ["likes osu"], "bob": ["lives in Berlin"]}. '
        "People with no durable facts may be omitted or mapped to []. "
        "Record stable traits only: preferences, skills, ownership, location. "
        "Do NOT record what someone is currently discussing, working on, "
        "or asking about unless stated as a stable trait. "
        "Skip greetings, jokes, transient moods.\n\n"
        f"{convo}"
    )
