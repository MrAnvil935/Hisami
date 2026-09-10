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

    Always keeps at least the top example even if it alone exceeds
    the budget.
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
