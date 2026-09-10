"""Unit tests for Hisami persistent memory (stdlib unittest only).

Covers the memory/ package plus the pure helpers of style_profile.py.
bot.py itself is not imported (it calls client.run at module level).

Run:  venv/bin/python -m unittest discover -s tests -v
"""

import json
import os
import shutil
import tempfile
import unittest

import style_profile
from memory import buffer as mem_buffer
from memory import examples as mem_examples
from memory import facts as mem_facts
from memory import llmlog as mem_llmlog
from memory import recall as mem_recall
from memory import store as mem_store
from memory import styleprofile as mem_styleprofile
from memory import summary as mem_summary


class TempDBMixin:
    """Isolated SQLite DB per test (WAL sidecars cleaned up too)."""

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        mem_store.configure(self.path)
        mem_store.init_db()
        super().setUp()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.path)
        except OSError:
            pass
        for suffix in ("-wal", "-shm", "-journal"):
            try:
                os.remove(self.path + suffix)
            except OSError:
                pass

    def _seed(self, channel="c1", n=10):
        for i in range(1, n + 1):
            mem_store.add_message(
                channel, i, f"u{i % 2}", f"user{i % 2}",
                "user", f"message number {i} about minecraft",
            )


class StoreTest(TempDBMixin, unittest.TestCase):
    def test_add_and_recent_order(self):
        self._seed(n=5)
        recent = mem_store.get_recent("c1", limit=10)
        self.assertEqual(len(recent), 5)
        self.assertEqual(recent[0]["msg_id"], 1)  # oldest first
        self.assertEqual(recent[-1]["msg_id"], 5)

    def test_recent_returns_author_id(self):
        mem_store.add_message("c9", 1, "u1", "alice", "user", "hello")
        rows = mem_store.get_recent("c9", 5)
        self.assertEqual(rows[0]["author_id"], "u1")

    def test_clear_recent_keeps_summaries_and_facts(self):
        self._seed(n=10)
        mem_store.add_summary("c1", "talked about minecraft", 1, 5)
        mem_store.upsert_fact("u1", "likes osu")
        removed = mem_store.clear_recent("c1", 4)
        self.assertEqual(removed, 4)
        self.assertEqual(mem_store.get_message_count("c1"), 6)
        self.assertEqual(len(mem_store.get_latest_summaries("c1")), 1)
        self.assertEqual(len(mem_store.get_facts("u1")), 1)

    def test_summary_chunking(self):
        self._seed(n=39)
        self.assertEqual(mem_store.get_unsummarized("c1", 40), [])
        mem_store.add_message("c1", 40, "u1", "alice", "user", "fortieth")
        chunk = mem_store.get_unsummarized("c1", 40)
        self.assertEqual(len(chunk), 40)
        mem_store.add_summary("c1", "s", 1, 40)
        mem_store.set_last_summary_upto("c1", 40)
        self.assertEqual(mem_store.get_unsummarized("c1", 40), [])


class BufferTest(TempDBMixin, unittest.TestCase):
    def test_token_budget_trims_oldest(self):
        self._seed(n=20)
        window = mem_buffer.load_window(
            "c1", budget_tokens=40, max_messages=30)
        # tiny budget must trim to a few newest messages
        self.assertTrue(1 <= len(window) <= 5)
        self.assertEqual(window[-1]["msg_id"], 20)


class RecallTest(TempDBMixin, unittest.TestCase):
    def test_fts_recall(self):
        mem_store.add_message("c1", 1, "u1", "alice", "user",
                              "I love playing osumania rhythm games")
        mem_store.add_message("c1", 2, "u2", "bob", "user",
                              "minecraft skyblock server is down again")
        mem_store.add_message("c1", 3, "u1", "alice", "user",
                              "weather today is nice")
        hits = mem_recall.search_messages("c1", "skyblock minecraft server")
        self.assertTrue(hits)
        self.assertIn("skyblock", hits[0]["content"])


class FactsTest(unittest.TestCase):
    def test_parse_single_user(self):
        self.assertEqual(
            mem_facts.parse_facts_json('["likes osu", "plays minecraft"]'),
            ["likes osu", "plays minecraft"])
        self.assertEqual(
            mem_facts.parse_facts_json('```json\n["lives in Berlin"]\n```'),
            ["lives in Berlin"])
        self.assertEqual(mem_facts.parse_facts_json("no json here"), [])
        self.assertEqual(mem_facts.parse_facts_json("[]"), [])

    def test_parse_multi_user(self):
        self.assertEqual(
            mem_facts.parse_multi_facts_json(
                '{"alice": ["likes osu"], "bob": []}'),
            {"alice": ["likes osu"]})
        self.assertEqual(
            mem_facts.parse_multi_facts_json(
                '```json\n{"alice": "plays mc"}\n```'),
            {"alice": ["plays mc"]})
        self.assertEqual(mem_facts.parse_multi_facts_json("nope"), {})
        self.assertEqual(mem_facts.parse_multi_facts_json('["not", "obj"]'), {})

    def test_multi_user_caps_per_user(self):
        big = {"alice": [f"fact number {i}" for i in range(20)]}
        parsed = mem_facts.parse_multi_facts_json(json.dumps(big))
        self.assertEqual(len(parsed["alice"]), 5)

    def test_resolve_speakers(self):
        chunk = [
            {"author_name": "alice", "author_id": "u1", "role": "user"},
            {"author_name": "alice", "author_id": "u1", "role": "user"},
            {"author_name": "bob", "author_id": "u2", "role": "user"},
            {"author_name": "Assistant", "author_id": "b", "role": "assistant"},
            {"author_name": "sam", "author_id": "u3", "role": "user"},
            {"author_name": "sam", "author_id": "u4", "role": "user"},
            {"author_name": "sam", "author_id": "u4", "role": "user"},
        ]
        mapping, ambiguous = mem_facts.resolve_speakers(chunk)
        self.assertEqual(mapping["alice"], "u1")
        self.assertEqual(mapping["bob"], "u2")
        self.assertNotIn("Assistant", mapping)
        self.assertEqual(mapping["sam"], "u4")  # most frequent wins
        self.assertIn("sam", ambiguous)
        self.assertNotIn("alice", ambiguous)


class SummaryPromptTest(unittest.TestCase):
    def test_summary_prompt_builder(self):
        msgs = [{"author_name": "alice", "content": "hello world"}]
        p = mem_summary.build_summary_prompt(msgs)
        self.assertIn("alice", p)
        self.assertIn("hello", p)

    def test_multi_fact_prompt_skips_assistants(self):
        msgs = [
            {"author_name": "alice", "role": "user", "content": "i love osu"},
            {"author_name": "Assistant", "role": "assistant", "content": "xd"},
        ]
        p = mem_summary.build_multi_fact_prompt(msgs)
        self.assertIn("alice", p)
        self.assertIn("i love osu", p)
        # assistant content must not appear as a speaker line
        self.assertNotIn("Assistant: xd", p)


class ExamplesTest(unittest.TestCase):
    def test_strip_boilerplate(self):
        raw = ("STYLE EXAMPLE (tone, phrasing, personality)\n"
               "\nContext:\nartige: hi\n\nUser Input:\nNone\n\n"
               "Target Response:\nxd")
        compact = mem_examples.strip_example_boilerplate(raw)
        self.assertNotIn("STYLE EXAMPLE", compact)
        self.assertNotIn("None", compact)
        self.assertIn("xd", compact)
        # reply-type prefix stripped too, content kept
        raw2 = ("HIGH VALUE INTERACTION (conversation reply)\n\n"
                "Target Response:\nhello there")
        self.assertIn("hello there",
                      mem_examples.strip_example_boilerplate(raw2))

    def test_fit_budget(self):
        examples = [f"example response number {i} " + "x" * 200
                    for i in range(10)]
        fitted = mem_examples.fit_examples(examples, 1200)
        # best-ranked kept first, total within budget, never empty
        self.assertTrue(fitted)
        self.assertTrue(fitted[0].startswith("example response number 0"))
        total = sum(mem_buffer.estimate_tokens(e) + 2 for e in fitted)
        self.assertLessEqual(total, 1200)
        # tiny budget still keeps the top example
        tiny = mem_examples.fit_examples(examples, 1)
        self.assertEqual(len(tiny), 1)


class LlmLogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        mem_llmlog.configure(self.tmp, enabled=True)

    def tearDown(self):
        mem_llmlog.configure("logs", enabled=True)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_record_shape(self):
        path = mem_llmlog.log_call(
            "openrouter", "m", "chat",
            {"model": "m", "messages": []},
            {"choices": []}, 12, "ok")
        self.assertIsNotNone(path)
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        # pretty-printed: spans multiple lines, blank-line separated
        self.assertIn("\n", raw.strip())
        record = json.loads(raw.strip())
        self.assertEqual(
            set(record),
            {"ts", "backend", "model", "purpose",
             "request", "response", "latency_ms", "status"})
        # bodies stored, headers never accepted/stored
        self.assertNotIn("headers", record)
        self.assertNotIn("Authorization", json.dumps(record))

    def test_multi_record_parse(self):
        for i in range(3):
            mem_llmlog.log_call(
                "ollama", f"m{i}", "chat",
                {"n": i}, {"r": i}, i, "ok")
        records = mem_llmlog.read_records(mem_llmlog.dump_file())
        self.assertEqual(len(records), 3)
        self.assertEqual([r["model"] for r in records], ["m0", "m1", "m2"])

    def test_rotation(self):
        mem_llmlog.configure(self.tmp, enabled=True,
                             max_bytes=200, backups=2)
        for i in range(10):
            mem_llmlog.log_call(
                "ollama", "m", "chat",
                {"n": i, "pad": "x" * 100}, {"r": i}, i, "ok")
        self.assertTrue(os.path.exists(mem_llmlog.dump_file() + ".1"))
        # every file on disk parses cleanly
        for suffix in ("", ".1", ".2"):
            p = mem_llmlog.dump_file() + suffix
            if os.path.exists(p):
                mem_llmlog.read_records(p)

    def test_disabled_noop(self):
        mem_llmlog.configure(self.tmp, enabled=False)
        self.assertIsNone(mem_llmlog.log_call(
            "ollama", "m", "chat", {}, {}, 1, "ok"))
        self.assertFalse(os.path.exists(mem_llmlog.dump_file()))

    def test_never_raises(self):
        mem_llmlog.configure("/nonexistent-dir-xyz", enabled=True)
        self.assertIsNone(mem_llmlog.log_call(
            "ollama", "m", "chat", {}, {}, 1, "ok"))


class StyleProfileTest(unittest.TestCase):
    def test_extract_responses(self):
        texts = [
            "STYLE EXAMPLE\n\nContext:\na: hi\n\nUser Input:\nNone\n\n"
            "Target Response:\nxd",
            "HIGH VALUE INTERACTION\n\nTarget Response:\nhello there",
            "no target here",
        ]
        self.assertEqual(
            mem_styleprofile.extract_responses(texts), ["xd", "hello there"])

    def test_sample_deterministic(self):
        pool = [f"msg {i}" for i in range(200)]
        a = mem_styleprofile.sample_responses(pool, n=60, seed=2)
        b = mem_styleprofile.sample_responses(pool, n=60, seed=2)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 60)
        # small pools returned whole
        self.assertEqual(mem_styleprofile.sample_responses(["x"], n=60), ["x"])

    def test_distill_prompt_contains_samples(self):
        p = mem_styleprofile.build_distill_prompt(["xd", "lol ok"])
        self.assertIn("xd", p)
        self.assertIn("lol ok", p)
        self.assertIn("400", p)  # length guidance present

    def test_openrouter_payload_has_no_secrets(self):
        payload = style_profile.build_openrouter_payload(["xd"], "m")
        self.assertEqual(payload["model"], "m")
        self.assertTrue(payload["messages"])
        blob = json.dumps(payload)
        self.assertNotIn("Authorization", blob)
        self.assertNotIn("Bearer", blob)
        self.assertNotIn("sk-or", blob)

    def test_build_messages_shape(self):
        msgs = style_profile.build_messages(["xd"])
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertIn("xd", msgs[1]["content"])


if __name__ == "__main__":
    unittest.main()
