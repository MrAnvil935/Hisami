"""Style-RAG example compaction: boilerplate strip + token budgeting.

Pure functions (no I/O, no Discord) so they are unit-testable.
bot.py cannot be imported by tests (it calls client.run at module
level), hence this module.
"""

import re

from .buffer import estimate_tokens


def strip_example_boilerplate(ex):
    """Compact a style-RAG example for prompt injection.

    Drops the PREFIX line added by embed.build_text and the useless
    'User Input: None' stanza that standalone examples carry.
    """
    text = str(ex)
    lines = text.split("\n")
    if lines and ("STYLE EXAMPLE" in lines[0]
                  or "HIGH VALUE INTERACTION" in lines[0]):
        text = "\n".join(lines[1:]).lstrip("\n")
    text = re.sub(r"\nUser Input:\nNone(\n|$)", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fit_examples(examples, budget_tokens):
    """Keep best-ranked examples that fit into the token budget.

    Examples keep their full Context / User Input / Target Response body
    (only the generated boilerplate is stripped), so the token budget is
    the only size limit. Always keeps at least the top example even if
    it alone exceeds the budget.
    """
    fitted = []
    used = 0
    for ex in examples:
        compact = strip_example_boilerplate(ex)
        cost = estimate_tokens(compact) + 2  # "- " + newline
        if fitted and used + cost > budget_tokens:
            break
        fitted.append(compact)
        used += cost
    return fitted


_RESPONSE_RE = re.compile(r"Target Response:\n(.*)", re.DOTALL)


def extract_style_snippets(texts, exclude_texts=(), max_chars=200):
    """Pull short Target-Response bodies for context-free style exemplars.

    Full examples carry a whole conversation; these are just the reply
    line, so many more fit in the prompt for pure tone/phrasing. Empties
    are dropped, each body is truncated to max_chars, and identical
    snippets are de-duplicated. `exclude_texts` (e.g. the full examples
    already injected) is skipped by raw text so the same record is never
    used twice. Order is preserved. Pure function (no I/O).
    """
    excluded = set(str(t) for t in (exclude_texts or ()))
    out = []
    seen = set()
    for text in texts or ():
        if str(text) in excluded:
            continue
        match = _RESPONSE_RE.search(str(text))
        if not match:
            continue
        body = match.group(1).strip()
        if not body:
            continue
        if max_chars and len(body) > max_chars:
            body = body[:max_chars].rstrip() + "…"
        if body in seen:
            continue
        seen.add(body)
        out.append(body)
    return out


def fit_style_snippets(snippets, budget_tokens, max_count=40):
    """Keep rank-ordered snippets within a token and count budget.

    Snippets are tiny, so both limits apply at once. Always keeps at
    least one snippet when any exist (mirrors fit_examples). Pure.
    """
    fitted = []
    used = 0
    try:
        max_count = int(max_count)
    except (TypeError, ValueError):
        max_count = 40
    for snippet in snippets or ():
        if len(fitted) >= max_count:
            break
        cost = estimate_tokens(snippet) + 2  # "- " + newline
        if fitted and used + cost > budget_tokens:
            break
        fitted.append(snippet)
        used += cost
    return fitted

