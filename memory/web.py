"""Web-result rendering for Components V2 messages (no Discord/network).

Pure functions (no I/O) so they are unit-testable. bot.py cannot be
imported by tests (it calls client.run at module level), hence this
module. Used by the /web command and the /prompt web-results button.
"""

V2_TEXT_BUDGET = 3800


def format_web_results(results, max_chars=V2_TEXT_BUDGET):
    """Render search results as markdown for a TextDisplay component.

    Each result: '### [n] [title](url)' + snippet, joined by blank lines.
    Over-budget output is truncated with a notice. Empty input -> ''.
    """
    if not results:
        return ""
    web_text = "\n\n".join(
        f"### [{i}] [{r.get('title') or 'No title'}]({r.get('url', '')})\n"
        f"{r.get('snippet', '')}"
        for i, r in enumerate(results, 1)
    )
    if len(web_text) > max_chars:
        notice = ("\n\n… More results were returned "
                  "but could not fit in this message.")
        web_text = web_text[:max_chars - len(notice)] + notice
    return web_text
