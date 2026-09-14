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
from memory import config as mem_config
from memory import examples as mem_examples
from memory import facts as mem_facts
from memory import llmlog as mem_llmlog
from memory import recall as mem_recall
from memory import store as mem_store
from memory import styleprofile as mem_styleprofile
from memory import summary as mem_summary
from memory import vision as mem_vision
from memory import web as mem_web


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

    def test_get_messages_by_ids(self):
        mem_store.add_message("cb", 1, "u1", "alice", "user", "first")
        mem_store.add_message("cb", 2, "u2", "bob", "user", "second",
                              attachments=[{"kind": "video", "name": "v.mp4"}])
        got = mem_store.get_messages_by_ids("cb", [1, 2, 999])
        self.assertEqual(set(got), {1, 2})
        self.assertEqual(got[1]["author_name"], "alice")
        self.assertEqual(got[2]["attachments"],
                         [{"kind": "video", "name": "v.mp4"}])
        # channel isolation + empty input
        self.assertEqual(mem_store.get_messages_by_ids("other", [1]), {})
        self.assertEqual(mem_store.get_messages_by_ids("cb", []), {})

    def test_clear_recent_keeps_summaries_and_facts(self):
        self._seed(n=10)
        mem_store.add_summary("c1", "talked about minecraft", 1, 5)
        mem_store.upsert_fact("u1", "likes osu")
        removed = mem_store.clear_recent("c1", 4)
        self.assertEqual(removed, 4)
        self.assertEqual(mem_store.get_message_count("c1"), 6)
        self.assertEqual(len(mem_store.get_latest_summaries("c1")), 1)
        self.assertEqual(len(mem_store.get_facts("u1")), 1)

    def test_get_facts_query_ranking(self):
        mem_store.upsert_fact("uq", "plays minecraft daily")
        mem_store.upsert_fact("uq", "likes hiking")
        mem_store.upsert_fact("uq", "owns a farm")
        # legacy path (no query): recency order
        legacy = [f["fact"] for f in mem_store.get_facts("uq", 3)]
        self.assertEqual(legacy[0], "owns a farm")
        # query path: topical old fact wins despite recency
        ranked = [f["fact"]
                  for f in mem_store.get_facts("uq", 3, query="minecraft?")]
        self.assertEqual(ranked[0], "plays minecraft daily")

    def test_pool_reaches_beyond_20(self):
        # oldest fact sits past the old 20-row pool cutoff: ranking must
        # still surface it when topical (regression test for pool width)
        mem_store.upsert_fact("uw", "ancient pottery techniques")
        for i in range(24):
            mem_store.upsert_fact("uw", f"filler hobby number {i}")
        ranked = [f["fact"] for f in mem_store.get_facts(
            "uw", 3, query="pottery kiln")]
        self.assertEqual(ranked[0], "ancient pottery techniques")

    def test_fact_embedding_roundtrip(self):
        import numpy as _np
        vec = _np.array([0.25] * 8, dtype="float32")
        mem_store.upsert_fact("ue", "loves stargazing", embedding=vec.tobytes())
        rows = mem_store.get_facts("ue", 5)
        self.assertEqual(len(rows), 1)
        back = mem_recall.decode_embedding(rows[0]["embedding"], dim=8)
        self.assertIsNotNone(back)
        self.assertAlmostEqual(float(back[0]), 0.25)

    def test_get_facts_hybrid_ranking(self):
        import numpy as _np

        def _vec(x, y):
            v = _np.zeros(768, dtype="float32")
            v[0], v[1] = x, y
            return v

        mem_store.upsert_fact(
            "uh", "ancient pottery techniques",
            embedding=_vec(1.0, 0.0).tobytes())
        mem_store.upsert_fact(
            "uh", "brand new hobby",
            embedding=_vec(0.0, 1.0).tobytes())
        # "ceramics kiln" shares no keywords: semantic path alone must win
        ranked = [f["fact"] for f in mem_store.get_facts(
            "uh", 5, query="ceramics kiln", query_vec=_vec(1.0, 0.05))]
        self.assertEqual(ranked[0], "ancient pottery techniques")

    def test_summary_embedding_roundtrip(self):
        import numpy as _np
        vec = _np.zeros(768, dtype="float32")
        vec[0] = 0.75
        q = _np.zeros(768, dtype="float32")
        q[0] = 0.75
        cid = mem_store.add_summary("cs", "talked about stars", 1, 5,
                                    embedding=vec.tobytes())
        found = mem_store.search_summaries("cs", "stars", query_vec=q)
        self.assertTrue(any(s["chunk_id"] == cid for s in found))

    def test_summary_chunking(self):
        self._seed(n=39)
        self.assertEqual(mem_store.get_unsummarized("c1", 40), [])
        mem_store.add_message("c1", 40, "u1", "alice", "user", "fortieth")
        chunk = mem_store.get_unsummarized("c1", 40)
        self.assertEqual(len(chunk), 40)
        mem_store.add_summary("c1", "s", 1, 40)
        mem_store.set_last_summary_upto("c1", 40)
        self.assertEqual(mem_store.get_unsummarized("c1", 40), [])

    def test_summary_early_token_trigger(self):
        # 12 long messages: under count 30 but over 3000-token budget
        for i in range(1, 13):
            mem_store.add_message(
                "ct", i, "u1", "alice", "user", "x" * 1200)
        chunk = mem_store.get_unsummarized(
            "ct", 30, chunk_tokens=3000, min_msgs=10)
        self.assertEqual(len(chunk), 12)

    def test_summary_min_msgs_guard(self):
        # token budget hit but too few messages -> no silly tiny summary
        for i in range(1, 6):
            mem_store.add_message(
                "cm", i, "u1", "alice", "user", "x" * 2000)
        self.assertEqual(
            mem_store.get_unsummarized("cm", 30, chunk_tokens=3000,
                                       min_msgs=10), [])

    def test_summary_default_chunk_size(self):
        self._seed(channel="cd", n=29)
        self.assertEqual(mem_store.get_unsummarized("cd"), [])
        mem_store.add_message("cd", 30, "u1", "alice", "user", "thirtieth")
        self.assertEqual(len(mem_store.get_unsummarized("cd")), 30)


class BufferTest(TempDBMixin, unittest.TestCase):
    def test_token_budget_trims_oldest(self):
        self._seed(n=20)
        window = mem_buffer.load_window(
            "c1", budget_tokens=40, max_messages=30)
        # tiny budget must trim to a few newest messages
        self.assertTrue(1 <= len(window) <= 5)
        self.assertEqual(window[-1]["msg_id"], 20)

    def test_format_plain(self):
        self.assertEqual(
            mem_buffer.format_history_line(
                {"author": "a", "content": "hi"}, None),
            "a: hi")

    def test_format_reply(self):
        parent = {"author": "bob", "content": "is this real?"}
        self.assertEqual(
            mem_buffer.format_history_line(
                {"author": "a", "content": "yes", "reply_to": 5}, parent),
            "a (replying to bob: is this real?): yes")

    def test_format_reply_truncates_parent(self):
        parent = {"author": "bob", "content": "x" * 900}
        out = mem_buffer.format_history_line(
            {"author": "a", "content": "yes", "reply_to": 5}, parent)
        self.assertIn("x" * 300, out)
        self.assertNotIn("x" * 301, out)

    def test_format_missing_parent(self):
        out = mem_buffer.format_history_line(
            {"author": "a", "content": "yes", "reply_to": 999}, None)
        self.assertIn("outside current context", out)
        self.assertTrue(out.startswith("a (replying"))
        self.assertTrue(out.endswith("yes"))

    def test_format_markers_preserved_on_reply(self):
        parent = {"author": "bob", "content": "look"}
        out = mem_buffer.format_history_line(
            {"author": "a", "content": "nice", "reply_to": 5,
             "attachments": [{"kind": "image", "name": "p.png"}]}, parent)
        self.assertIn("[image: p.png]", out)
        self.assertIn("replying to bob", out)


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

    def test_rank_by_overlap(self):
        items = [
            {"fact": "likes hiking", "updated_at": 30},
            {"fact": "plays minecraft daily", "updated_at": 10},
            {"fact": "owns a farm", "updated_at": 20},
        ]
        ranked = mem_recall.rank_by_overlap(items, "minecraft server?")
        self.assertEqual(ranked[0]["fact"], "plays minecraft daily")
        # no overlap -> pure recency, legacy order preserved
        ranked = mem_recall.rank_by_overlap(items, "zzz qqq")
        self.assertEqual([r["fact"] for r in ranked],
                         ["likes hiking", "owns a farm", "plays minecraft daily"])
        self.assertEqual(mem_recall.rank_by_overlap(items, ""), ranked)

    def test_cosine_sim(self):
        self.assertAlmostEqual(
            mem_recall.cosine_sim([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(
            mem_recall.cosine_sim([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertEqual(mem_recall.cosine_sim([1.0], [1.0, 0.0]), 0.0)
        self.assertEqual(mem_recall.cosine_sim(None, [1.0]), 0.0)
        self.assertEqual(mem_recall.cosine_sim([0.0, 0.0], [1.0, 0.0]), 0.0)

    def test_decode_embedding(self):
        import numpy as _np
        vec = _np.array([0.5, -0.25], dtype="float32")
        back = mem_recall.decode_embedding(vec.tobytes(), dim=2)
        self.assertIsNotNone(back)
        self.assertAlmostEqual(float(back[0]), 0.5)
        self.assertIsNone(mem_recall.decode_embedding(None))
        self.assertIsNone(mem_recall.decode_embedding(b"junk", dim=2))
        self.assertIsNone(
            mem_recall.decode_embedding(vec.tobytes(), dim=7))

    def test_rank_hybrid(self):
        import numpy as _np

        def _vec(x, y):
            v = _np.zeros(768, dtype="float32")
            v[0], v[1] = x, y
            return v.tobytes()

        items = [
            {"fact": "likes hiking", "updated_at": 30,
             "embedding": _vec(0.0, 1.0)},
            {"fact": "unrelated old note", "updated_at": 5,
             "embedding": _vec(1.0, 0.0)},
            {"fact": "no vector here", "updated_at": 40,
             "embedding": None},
        ]
        q = _np.zeros(768, dtype="float32")
        q[0], q[1] = 1.0, 0.1
        ranked = mem_recall.rank_hybrid(items, "zzz", q)
        # semantic winner first despite oldest timestamp
        self.assertEqual(ranked[0]["fact"], "unrelated old note")
        # no vectors anywhere -> keyword/recency order preserved
        ranked = mem_recall.rank_hybrid(items, "hiking", None)
        self.assertEqual(ranked[0]["fact"], "likes hiking")

    def test_channel_engaged(self):
        msgs = [
            {"content": "hello everyone"},
            {"content": "hey <@123> what is up"},
        ]
        self.assertTrue(mem_recall.channel_engaged(msgs, "123"))
        self.assertTrue(mem_recall.channel_engaged(msgs, "123", lookback=1))
        # nickname mention form
        self.assertTrue(mem_recall.channel_engaged(
            [{"content": "yo <@!123>"}], "123"))
        # longer ID must not false-positive on a bot-id prefix
        self.assertFalse(mem_recall.channel_engaged(
            [{"content": "hi <@12345>"}], "123"))
        # no mention / empty / lookback window
        self.assertFalse(mem_recall.channel_engaged(msgs, "999"))
        self.assertFalse(mem_recall.channel_engaged([], "123"))
        self.assertFalse(mem_recall.channel_engaged(msgs, "123", lookback=0))
        old = [{"content": "hi <@123>"}] + [{"content": "x"}] * 5
        self.assertFalse(mem_recall.channel_engaged(old, "123", lookback=5))
        self.assertTrue(mem_recall.channel_engaged(old, "123", lookback=6))
        # unknown bot fails open (never halts summarization)
        self.assertTrue(mem_recall.channel_engaged([], ""))
        self.assertTrue(mem_recall.channel_engaged([], None))

    def test_referenced_users_mention(self):
        msgs = [
            {"author_id": "u1", "author_name": "alice", "role": "user",
             "content": "hi"},
        ]
        ids, names = mem_recall.find_referenced_users(
            msgs, "u9", "what does <@123> like?", max_users=2)
        # explicit mention first, recent speakers fill remaining slots
        self.assertEqual(ids, ["123", "u1"])
        self.assertEqual(names, {"u1": "alice"})

    def test_referenced_users_name_and_speakers(self):
        msgs = [
            {"author_id": "u1", "author_name": "alice", "role": "user",
             "content": "hello"},
            {"author_id": "u2", "author_name": "bob", "role": "user",
             "content": "hey"},
            {"author_id": "b", "author_name": "Assistant", "role": "assistant",
             "content": "xd"},
        ]
        # named speaker wins over recency
        ids, names = mem_recall.find_referenced_users(
            msgs, "u9", "does alice play osu?", max_users=2)
        self.assertEqual(ids[0], "u1")
        self.assertEqual(names["u1"], "alice")
        # no name match -> newest speakers first, self + bot excluded
        ids, _ = mem_recall.find_referenced_users(
            msgs, "u2", "what is everyone up to?", max_users=2)
        self.assertEqual(ids, ["u1"])
        self.assertNotIn("b", ids)

    def test_referenced_users_cap_and_self(self):
        msgs = [
            {"author_id": "u1", "author_name": "alice", "role": "user",
             "content": "a"},
            {"author_id": "u2", "author_name": "bob", "role": "user",
             "content": "b"},
        ]
        ids, _ = mem_recall.find_referenced_users(
            msgs, "u1", "hi all", max_users=1)
        self.assertEqual(ids, ["u2"])
        ids, _ = mem_recall.find_referenced_users([], "u1", "hi", max_users=2)
        self.assertEqual(ids, [])

    def test_nickname_matching(self):
        # token of an underscored name matches ("nos" from "nos_yous")
        self.assertTrue(mem_recall.name_mentioned(
            "nos_yous", "does nos play osu?"))
        # dash/dot/space-separated tokens match whole words
        self.assertTrue(mem_recall.name_mentioned(
            "Ann-Marie", "ask ann about it"))
        self.assertTrue(mem_recall.name_mentioned(
            "Ann Marie", "ann marie is here"))
        # no substring false positives ("bob" must not fire on "bobby")
        self.assertFalse(mem_recall.name_mentioned("bob", "bobby is here"))
        self.assertTrue(mem_recall.name_mentioned("bob", "hey bob!"))
        # short names never match
        self.assertFalse(mem_recall.name_mentioned("Al", "say hi to al"))
        self.assertFalse(mem_recall.name_mentioned("", "hello"))

    def test_referenced_users_nickname(self):
        msgs = [
            {"author_id": "u7", "author_name": "nos_yous", "role": "user",
             "content": "i main inkling"},
        ]
        ids, names = mem_recall.find_referenced_users(
            msgs, "u9", "is nos good at splatoon?", max_users=2)
        self.assertEqual(ids[0], "u7")
        self.assertEqual(names["u7"], "nos_yous")


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


class WebFormatTest(unittest.TestCase):
    def test_numbering_and_links(self):
        out = mem_web.format_web_results([
            {"title": "A", "url": "https://a.test", "snippet": "snip a"},
            {"title": "", "url": "", "snippet": ""},
        ])
        self.assertIn("### [1] [A](https://a.test)", out)
        self.assertIn("snip a", out)
        self.assertIn("### [2] [No title]()", out)

    def test_empty(self):
        self.assertEqual(mem_web.format_web_results([]), "")

    def test_truncation(self):
        big = [{"title": f"t{i}", "url": f"https://x/{i}",
                "snippet": "s" * 500} for i in range(10)]
        out = mem_web.format_web_results(big, max_chars=1000)
        self.assertLessEqual(len(out), 1000)
        self.assertIn("could not fit", out)
        # small budget keeps full text untouched
        small = mem_web.format_web_results(big[:1])
        self.assertNotIn("could not fit", small)


class VisionTest(unittest.TestCase):
    def test_extract_image_urls(self):
        text = ("look https://x.com/a.png, and http://y.org/b.JPG?w=1 "
                "not this https://z.net/page.html (see pic.")
        self.assertEqual(mem_vision.extract_image_urls(text),
                         ["https://x.com/a.png", "http://y.org/b.JPG?w=1"])
        self.assertEqual(mem_vision.extract_image_urls("no links"), [])
        self.assertEqual(mem_vision.extract_image_urls(""), [])

    def test_classify_attachments(self):
        from types import SimpleNamespace as NS
        atts = [
            NS(content_type="image/png", filename="a.png",
               url="https://cdn/x/a.png"),
            NS(content_type="video/mp4", filename="b.mp4",
               url="https://cdn/x/b.mp4"),
            NS(content_type="application/pdf", filename="c.pdf",
               url="https://cdn/x/c.pdf"),
            NS(content_type="", filename="d.jpg", url="https://cdn/x/d.jpg"),
            NS(content_type="image/png", filename="e.png", url=""),
        ]
        out = mem_vision.classify_attachments(atts)
        self.assertEqual(
            [(a["kind"], a["name"]) for a in out],
            [("image", "a.png"), ("video", "b.mp4"), ("image", "d.jpg")])

    def test_downscale(self):
        from PIL import Image
        import io as _io
        img = Image.new("RGB", (2000, 100), "red")
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        small = mem_vision.downscale(buf.getvalue())
        out = Image.open(_io.BytesIO(small))
        self.assertEqual(out.format, "JPEG")
        self.assertLessEqual(max(out.size), 512)
        self.assertGreater(max(out.size), 0)

    def test_cache_key_stable(self):
        self.assertEqual(mem_vision.cache_key("https://x/a.png"),
                         mem_vision.cache_key("https://x/a.png"))
        self.assertNotEqual(mem_vision.cache_key("https://x/a.png"),
                            mem_vision.cache_key("https://x/b.png"))

    def test_format_markers(self):
        self.assertEqual(
            mem_vision.format_markers(
                [{"kind": "image", "name": "a.png"},
                 {"kind": "video", "name": "b.mp4"}]),
            " [image: a.png] [video: b.mp4]")
        self.assertEqual(mem_vision.format_markers([]), "")
        self.assertEqual(mem_vision.format_markers(None), "")


class VisionStoreTest(TempDBMixin, unittest.TestCase):
    def test_attachments_roundtrip(self):
        mem_store.add_message(
            "cv", 1, "u1", "alice", "user", "look",
            attachments=[{"kind": "image", "name": "a.png",
                          "url": "https://x/a.png"},
                         {"kind": "other", "name": "x.zip"}])
        rows = mem_store.get_recent("cv", 5)
        # only image/video kinds persist
        self.assertEqual(rows[0]["attachments"],
                         [{"kind": "image", "name": "a.png"}])

    def test_image_cache(self):
        self.assertEqual(mem_store.get_image_desc("k1"), "")
        mem_store.set_image_desc("k1", "a cat", max_rows=200)
        self.assertEqual(
            mem_store.get_image_desc("k1", ttl_seconds=60), "a cat")
        self.assertEqual(
            mem_store.get_image_desc("k1", ttl_seconds=-1), "")


class ConfigTest(unittest.TestCase):
    def test_strip_line_comments(self):
        text = '{\n// full line comment\n"a": 1 // trailing\n}'
        self.assertEqual(mem_config.strip_json_comments(text),
                         '{\n\n"a": 1 \n}')

    def test_strip_block_comments(self):
        text = '{"a": /* inline */ 1, /* multi\nline */ "b": 2}'
        self.assertEqual(json.loads(mem_config.strip_json_comments(text)),
                         {"a": 1, "b": 2})

    def test_urls_and_escapes_untouched(self):
        text = ('{"url": "http://localhost:11434/api/chat",\n'
                '"q": "a // not comment /* nor this",\n'
                '"e": "quote \\" // still string"}')
        parsed = json.loads(mem_config.strip_json_comments(text))
        self.assertEqual(parsed["url"], "http://localhost:11434/api/chat")
        self.assertEqual(parsed["q"], "a // not comment /* nor this")
        self.assertEqual(parsed["e"], 'quote " // still string')

    def test_load_real_config(self):
        if not os.path.exists("config.json"):
            self.skipTest("run from project root")
        cfg = mem_config.load_config("config.json")
        for key in ("discord_token", "model", "ollama_model",
                    "memory_db_path", "summary_model", "vision_model"):
            self.assertIn(key, cfg)


if __name__ == "__main__":
    unittest.main()
