"""Static style-profile helpers: distill embed corpus into a persona blurb.

Pure functions (no I/O, no network) so they are unit-testable.
Used by the manual style_profile.py script — never in the bot hot path.
"""

import random
import re

DISTILL_SYSTEM = (
    "You distill a Discord user's chatting style into a concise profile. "
    "Be factual and specific. No roleplay, no extra commentary."
)


def extract_responses(texts):
    """Pull Target Response bodies from embed.py build_text records."""
    responses = []
    for text in texts:
        m = re.search(r"Target Response:\n(.*)", str(text), re.DOTALL)
        if m:
            body = m.group(1).strip()
            if body:
                responses.append(body)
    return responses


def sample_responses(responses, n=60, seed=2):
    """Deterministic random sample (seeded so regeneration is reviewable)."""
    rng = random.Random(seed)
    if len(responses) <= n:
        return list(responses)
    return rng.sample(list(responses), n)


def build_distill_prompt(samples):
    numbered = "\n".join(f"{i}. {s[:300]}" for i, s in enumerate(samples, 1))
    return (
        "These are real chat messages from one Discord user. "
        "Describe their chatting style as a concise bullet profile covering: "
        "typical message length, tone, phrasing habits, slang, emoji/punctuation "
        "use, capitalization habits, and things they never do. "
        "Keep it under 400 words.\n\n"
        f"{numbered}"
    )
