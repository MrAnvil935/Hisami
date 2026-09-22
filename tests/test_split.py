"""Unit tests for memory/split.py (fence-aware reply splitting).

bot.py itself is not imported (it calls client.run at module level).

Run:  venv/bin/python -m unittest discover -s tests -v
"""

import unittest

from memory import split as mem_split


def balanced(chunk):
    return chunk.count("```") % 2 == 0


class SplitTest(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(mem_split.split_message("hi"), ["hi"])
        exact = "x" * 2000
        self.assertEqual(mem_split.split_message(exact), [exact])

    def test_paragraph_split_prefers_blank_line(self):
        p1, p2, p3 = "a" * 800, "b" * 800, "c" * 800
        parts = mem_split.split_message(
            f"{p1}\n\n{p2}\n\n{p3}", limit=2000, max_parts=3)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], f"{p1}\n\n{p2}")
        self.assertEqual(parts[1], p3)
        for p in parts:
            self.assertLessEqual(len(p), 2000)

    def test_codeblock_never_left_open(self):
        code = "\n".join(f"line{i} " + "x" * 60 for i in range(30))
        text = "intro\n```python\n" + code + "\n```\noutro"
        parts = mem_split.split_message(text, limit=1000, max_parts=4)
        self.assertGreater(len(parts), 1)
        for p in parts:
            self.assertLessEqual(len(p), 1000)
            self.assertTrue(balanced(p), f"unbalanced fence in: {p[:80]}")
        # every code line survived somewhere (all fits in 4x1000)
        joined = "\n".join(parts).replace("```python", "").replace("```", "")
        for i in range(30):
            self.assertIn(f"line{i}", joined)
        self.assertIn("intro", joined)
        self.assertIn("outro", joined)

    def test_unclosed_fence_balanced(self):
        text = "hey\n```\n" + "y" * 3000
        parts = mem_split.split_message(text, limit=1000, max_parts=4)
        for p in parts:
            self.assertLessEqual(len(p), 1000)
            self.assertTrue(balanced(p))

    def test_beyond_cap_truncates_with_marker(self):
        parts = mem_split.split_message("x" * 7000, limit=2000,
                                        max_parts=3)
        self.assertEqual(len(parts), 3)
        for p in parts:
            self.assertLessEqual(len(p), 2000)
        self.assertTrue(parts[-1].endswith("…"))

    def test_max_parts_one_is_truncate(self):
        parts = mem_split.split_message("y" * 2500, limit=2000,
                                        max_parts=1)
        self.assertEqual(len(parts), 1)
        self.assertLessEqual(len(parts[0]), 2000)
        self.assertTrue(parts[0].endswith("…"))

    def test_long_single_line_hard_cuts(self):
        parts = mem_split.split_message("z" * 7000, limit=2000,
                                        max_parts=3)
        self.assertEqual(len(parts), 3)
        self.assertEqual("".join(parts)[:5999], "z" * 5999)
        self.assertTrue(parts[-1].endswith("…"))

    def test_garbage_input_never_crashes(self):
        self.assertTrue(mem_split.split_message(None))
        self.assertTrue(mem_split.split_message(12345))
        self.assertTrue(mem_split.split_message("x" * 100, limit="bad",
                                                max_parts="bad"))


if __name__ == "__main__":
    unittest.main()
