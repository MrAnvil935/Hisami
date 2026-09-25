"""Unit tests for the /prompt agentic web flow in memory/web.py.

Covers router-decision parsing, article extraction, the fetched-content
prompt tier, and the results-button rendering. bot.py itself is not
imported (it calls client.run at module level).

Run:  venv/bin/python -m unittest discover -s tests -v
"""

import unittest

from memory import web as mem_web


class RouterParseTest(unittest.TestCase):
    def test_valid_approve_decline(self):
        need, qs = mem_web.parse_router_decision(
            '{"need_search": true, "queries": ["q1", "q2"]}',
            fallback_query="orig")
        self.assertTrue(need)
        self.assertEqual(qs, ["q1", "q2"])
        need, qs = mem_web.parse_router_decision(
            '{"need_search": false, "queries": ["q1"]}',
            fallback_query="orig")
        self.assertFalse(need)
        self.assertEqual(qs, [])

    def test_fenced_json(self):
        need, qs = mem_web.parse_router_decision(
            '```json\n{"need_search": true, "queries": ["qq"]}\n```',
            fallback_query="orig")
        self.assertTrue(need)
        self.assertEqual(qs, ["qq"])

    def test_garbage_falls_back_to_verbatim(self):
        need, qs = mem_web.parse_router_decision(
            "I think maybe search?", fallback_query="orig")
        self.assertTrue(need)
        self.assertEqual(qs, ["orig"])
        need, qs = mem_web.parse_router_decision(None, fallback_query="")
        self.assertTrue(need)
        self.assertEqual(qs, [])

    def test_query_cap_and_cleanup(self):
        need, qs = mem_web.parse_router_decision(
            '{"need_search": true, '
            '"queries": ["a", "", "b", "c"]}',
            max_queries=2, fallback_query="orig")
        self.assertEqual(qs, ["a", "b"])

    def test_missing_need_defaults_to_search(self):
        need, qs = mem_web.parse_router_decision(
            '{"queries": ["q"]}', fallback_query="orig")
        self.assertTrue(need)
        self.assertEqual(qs, ["q"])

    def test_empty_queries_use_fallback(self):
        need, qs = mem_web.parse_router_decision(
            '{"need_search": true, "queries": []}', fallback_query="orig")
        self.assertTrue(need)
        self.assertEqual(qs, ["orig"])


class ExtractTest(unittest.TestCase):
    ARTICLE = ("<html><head><title>t</title><script>evil()</script></head>"
               "<body><nav>menu menu</nav><article><h1>Head</h1>"
               "<p>First paragraph here.</p><p>Second one.</p></article>"
               "</body></html>")

    def test_article_preferred_boilerplate_dropped(self):
        text = mem_web.extract_article_text(self.ARTICLE)
        self.assertIn("First paragraph", text)
        self.assertNotIn("evil", text)
        self.assertNotIn("menu", text)

    def test_main_fallback(self):
        html = ("<html><body><main><p>Main body text.</p></main>"
                "</body></html>")
        self.assertIn("Main body text.",
                      mem_web.extract_article_text(html))

    def test_garbage_returns_empty(self):
        self.assertEqual(mem_web.extract_article_text(""), "")
        self.assertEqual(mem_web.extract_article_text(None), "")
        self.assertEqual(mem_web.extract_article_text(
            "<html><body><script>x</script></body></html>"), "")

    def test_cap_with_marker(self):
        html = f"<html><body><p>{'w' * 5000}</p></body></html>"
        text = mem_web.extract_article_text(html, max_chars=100)
        self.assertLessEqual(len(text), 100)
        self.assertTrue(text.endswith("…"))


class FetchedTierTest(unittest.TestCase):
    RESULTS = [
        {"title": "T1", "url": "https://a.example/x", "snippet": "s1"},
        {"title": "T2", "url": "https://b.example/y", "snippet": "s2"},
    ]

    def test_fetched_rendered_snippets_skipped(self):
        tier = mem_web.format_fetched_tier(
            self.RESULTS, {"https://a.example/x": "FULL BODY"})
        self.assertIn("FULL BODY", tier)
        self.assertIn("T1", tier)
        self.assertNotIn("s2", tier)  # caller renders those as snippets
        self.assertNotIn("T2", tier)

    def test_empty_contents(self):
        self.assertEqual(mem_web.format_fetched_tier(self.RESULTS, {}), "")
        self.assertEqual(mem_web.format_fetched_tier([], {}), "")


class PromptSourcesTest(unittest.TestCase):
    RECORD = {
        "searched": True,
        "queries": ["q1", "q2"],
        "results": [
            {"title": "T1", "url": "https://a.example/x", "snippet": "s1"},
            {"title": "T2", "url": "https://b.example/y", "snippet": "s2"},
        ],
        "fetched": {"https://a.example/x": 1500},
    }

    def test_queries_and_markers(self):
        text = mem_web.format_prompt_sources(self.RECORD)
        self.assertIn("Searched for: q1; q2", text)
        self.assertIn("s1", text)
        self.assertIn("📄 full page text was included in the prompt", text)
        # only the fetched source gets the marker
        self.assertEqual(text.count("📄"), 1)

    def test_declined(self):
        text = mem_web.format_prompt_sources(
            {"searched": False, "queries": [],
             "results": [], "fetched": {}})
        self.assertIn("no web search was needed", text)

    def test_empty(self):
        self.assertEqual(mem_web.format_prompt_sources(None), "")
        self.assertEqual(mem_web.format_prompt_sources({}), "")


if __name__ == "__main__":
    unittest.main()
