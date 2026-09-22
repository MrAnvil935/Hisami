"""Unit tests for memory/debug.py (magnifying-glass reaction debug).

Covers normalize_record over both backend shapes plus format_debug_record
(section presence, optional reasoning, truncation, malformed input).
bot.py itself is not imported (it calls client.run at module level).

Run:  venv/bin/python -m unittest discover -s tests -v
"""

import unittest

from memory import debug as mem_debug


OPENROUTER_RESPONSE = {
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "xd lol",
            "reasoning": "user greeted, match vibe",
            "reasoning_details": [
                {"type": "reasoning.text", "text": "detail one"},
                {"type": "reasoning.summary", "summary": "summ one"},
            ],
        },
    }],
    "usage": {"prompt_tokens": 100, "completion_tokens": 20,
              "total_tokens": 120},
}

OPENROUTER_REQUEST = {
    "model": "some-model:free",
    "messages": [
        {"role": "system", "content": "be funny"},
        {"role": "user", "content": "hi"},
    ],
    "temperature": 0.9,
    "top_p": 0.95,
}

OLLAMA_RESPONSE = {
    "message": {"role": "assistant", "content": "hey",
                "thinking": "keep it short"},
    "prompt_eval_count": 50,
    "eval_count": 10,
    "eval_duration": 123456,
}

OLLAMA_REQUEST = {
    "model": "somemodel:12b",
    "messages": [{"role": "user", "content": "hi"}],
    "options": {"temperature": 0.9, "top_p": 0.95,
                "num_ctx": 10000, "num_predict": 800},
    "stream": False,
}


class NormalizeTest(unittest.TestCase):
    def test_openrouter_extracts_reasoning_and_usage(self):
        rec = mem_debug.normalize_record(
            "openrouter", "m", OPENROUTER_REQUEST, OPENROUTER_RESPONSE,
            1234, fallback=True, attempts=2, reply_text="xd lol")
        self.assertEqual(rec["backend"], "openrouter")
        self.assertTrue(rec["fallback"])
        self.assertEqual(rec["attempts"], 2)
        self.assertEqual(rec["latency_ms"], 1234)
        self.assertIn("user greeted", rec["reasoning"])
        self.assertIn("detail one", rec["reasoning"])
        self.assertEqual(rec["usage"]["prompt_tokens"], 100)
        self.assertEqual(rec["params"]["temperature"], 0.9)
        self.assertEqual(len(rec["messages"]), 2)
        self.assertEqual(rec["reply"], "xd lol")

    def test_openrouter_without_reasoning(self):
        resp = {"choices": [{"message": {"content": "hi"}}],
                "usage": {}}
        rec = mem_debug.normalize_record(
            "openrouter", "m", OPENROUTER_REQUEST, resp, 10)
        self.assertEqual(rec["reasoning"], "")
        self.assertEqual(rec["usage"], {})

    def test_ollama_extracts_thinking_and_eval_counts(self):
        rec = mem_debug.normalize_record(
            "ollama", "m", OLLAMA_REQUEST, OLLAMA_RESPONSE, 99)
        self.assertEqual(rec["reasoning"], "keep it short")
        self.assertEqual(rec["usage"]["eval_count"], 10)
        self.assertEqual(rec["params"]["num_ctx"], 10000)
        self.assertFalse(rec["fallback"])

    def test_malformed_inputs_never_crash(self):
        rec = mem_debug.normalize_record("x", "m", None, None, None)
        self.assertEqual(rec["messages"], [])
        self.assertEqual(rec["reasoning"], "")
        self.assertEqual(rec["usage"], {})
        self.assertEqual(rec["params"], {})

    def test_list_content_renders_image_placeholder(self):
        req = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]}]}
        rec = mem_debug.normalize_record("openrouter", "m", req, {}, 1)
        content = rec["messages"][0]["content"]
        self.assertIn("look", content)
        self.assertIn("[image", content)
        self.assertNotIn("AAAA", content)


class FormatTest(unittest.TestCase):
    def test_sections_present_with_reasoning(self):
        rec = mem_debug.normalize_record(
            "openrouter", "m", OPENROUTER_REQUEST, OPENROUTER_RESPONSE,
            1234, reply_text="xd lol")
        text = mem_debug.format_debug_record(rec)
        for section in ("## Prompt messages", "## Reasoning",
                        "## Reply (sent text)", "## Usage",
                        "openrouter / m", "xd lol", "user greeted"):
            self.assertIn(section, text)

    def test_reasoning_section_omitted_when_absent(self):
        rec = mem_debug.normalize_record(
            "ollama", "m", OLLAMA_REQUEST,
            {"message": {"content": "hey"}}, 5, reply_text="hey")
        text = mem_debug.format_debug_record(rec)
        self.assertNotIn("## Reasoning", text)
        self.assertIn("hey", text)

    def test_truncation_keeps_header_and_reply(self):
        big = "z" * 5000
        req = {"messages": [{"role": "user", "content": big}]}
        rec = mem_debug.normalize_record(
            "openrouter", "m", req, OPENROUTER_RESPONSE, 1,
            reply_text="short reply")
        text = mem_debug.format_debug_record(rec, max_chars=1000)
        self.assertLessEqual(len(text), 1200)
        self.assertIn("truncated", text)
        self.assertIn("short reply", text)
        self.assertIn("Hisami debug", text)

    def test_malformed_record_renders_fallback(self):
        text = mem_debug.format_debug_record(None)
        self.assertIn("Hisami debug", text)
        text = mem_debug.format_debug_record({"backend": "x"})
        self.assertIn("## Usage", text)

    def test_custom_bot_name(self):
        rec = mem_debug.normalize_record(
            "openrouter", "m", OPENROUTER_REQUEST, OPENROUTER_RESPONSE,
            1, reply_text="hi")
        text = mem_debug.format_debug_record(rec, bot_name="TestBot")
        self.assertIn("# TestBot debug", text)
        self.assertNotIn("Hisami", text)
        fallback = mem_debug.format_debug_record(None, bot_name="TestBot")
        self.assertIn("# TestBot debug", fallback)


if __name__ == "__main__":
    unittest.main()
