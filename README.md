# Discord RAG Mimic Bot

A Discord chatbot that mimics a specific user's chatting style using Retrieval-Augmented Generation (RAG).

The bot builds an embedding index from exported Discord messages and uses it as style examples during generation. It supports:

* OpenRouter as the primary inference backend, Ollama as local-first option
* Local embeddings using `nomic-embed-text`
* HNSW vector search for fast example retrieval
* Persistent per-channel conversation memory (SQLite)
* Long-term memory: channel summaries + relevance-ranked user facts + keyword recall
* Static style profile distilled once from the embedding corpus
* Web search (DuckDuckGo, 3 retries on failure) for questions about
  current events, plus a `/web` command showing raw results with no AI
  involved (result count 1–10, default 5)
* `/prompt` with `web:True` runs an agentic flow: a small model
  (local-first, `prompt_router_*`) decides if search is needed and
  issues up to 2 queries, top result pages are fetched for full
  article text, and the results button shows the queries plus what
  the model was given
* Slash commands for status information and maintenance
* Two-sink debug logging: concise terminal + full payloads on disk
* Replies on @-mention in servers; in DMs every message gets a reply
  (5s per-user cooldown still applies everywhere)
* Long replies split into at most `reply_max_parts` (default 3)
  Discord-sized messages instead of truncating — paragraph breaks
  preferred, fenced code blocks closed/reopened across chunks so
  formatting never breaks; only the first chunk is a reply (one ping)

---

## Setup

### 1. Export Discord messages

Export the desired Discord channel(s) using **DiscordChatExporter** in **JSON** format.

Rename each exported file to:

```
export.json
```

Open `converter.py` and set:

```python
TARGET_USER_ID = ...
```

to the Discord user ID of the person you want the bot to imitate.

Run:

```bash
venv/bin/python converter.py
```

This produces:

```
dataset.jsonl
```

If you exported multiple channels, simply concatenate the resulting `dataset.jsonl` files into one.

---

### 2. Generate embeddings

Start the Ollama server.

Pull the embedding model:

```bash
ollama pull nomic-embed-text
```

Run:

```bash
venv/bin/python embed.py
```

This generates:

```
index.bin
texts.json
```

These files are required by the bot.

---

### 3. Configure the bot

Edit `config.json` (JSONC — `//` and `/* */` comments allowed, every key
documented inline).

At minimum you'll need:

* Discord bot token
* OpenRouter API key

Tunable groups (all have built-in defaults, see `config.json`):

* `model` / `fallback_model` — main chat chain (local Ollama first, then OpenRouter)
* `ollama_autoload` — preload Ollama models at startup (default `false`;
  a 12b + vision + embed load can exceed VRAM unasked). The embedding model
  preloads whenever the server is up regardless of this flag; generation
  models only with it on. With it off, startup logs one block naming
  configured-but-unloaded models and how to enable preloading.
* `summary_ollama_model` / `summary_model` — background chain for memory
  summarization and fact extraction (local first, OpenRouter last resort)
* `vision_ollama_model` / `vision_model` — image description chain
  (local first, OpenRouter last resort); see `vision_*` limits.
  Both models must be vision-capable (text-only models return nothing useful).
  `vision_ollama_ctx` should stay generous (default 8192): images consume
  hundreds of context tokens each and a small ctx truncates them silently.
  `summary_max_tokens` / `vision_max_tokens` are enforced on both backends
  (Ollama `num_predict` included), so raise them on length failures.
* `max_examples` / `examples_max_tokens` — style-example retrieval limits
* `memory_*` — buffer size, summary chunk size, recall limits, cooldowns
* `search_*` — web search triggers and limits
* `log_*` / `llm_dump_enabled` — debug logging verbosity and retention

---

### 4. Install dependencies

A virtualenv inside the project folder is the expected setup:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

---

### 5. Generate the static style profile (recommended)

Distill the embedding corpus into a hand-reviewed persona blurb (local
summary model first, OpenRouter summary model as fallback):

```bash
venv/bin/python style_profile.py
```

Review and edit `style_profile.txt`, then (re)start the bot. Without this
file the bot falls back to generic hardcoded style lines.

---

### 6. Start the bot

```bash
venv/bin/python bot.py
```

Make sure the Ollama server is running — at minimum for `nomic-embed-text`.
If a generation model is loaded, the bot prefers it; otherwise it uses
OpenRouter. If Ollama is down entirely, the bot degrades gracefully
(keyword-only retrieval, OpenRouter generation).

---

## Memory system

Short-term history lives in `memory.db` (SQLite, WAL mode) instead of RAM,
so it survives restarts. On top of that, long-term layers feed every
prompt:

* **Channel summaries** — every 30 messages per channel are summarized by the
  background chain into `summaries` (or earlier if the backlog passes a token
  budget, so long messages don't bloat the raw window); the 2 most relevant
  are injected, ranked by embedding similarity + keyword boost like facts.
  Channels with no bot mention in the recent buffer are skipped
  until mentioned again (DMs always qualify and are summarized like
  any channel).
* **Per-user facts** — durable traits extracted for all speakers batched
  on each summary chunk, embedded at write time. At prompt time facts rank
  by embedding similarity + keyword boost (recency tiebreak), so a topical
  old fact beats newer trivial ones; top 5 per user are injected. Disable
  with `memory_semantic_rank: false` for pure keyword ranking (also the
  automatic fallback when Ollama is down).
* **Peer recall** — when someone mentions, names, or chats alongside other
  users, their facts are injected too (`Known about <name>:`), so the bot
  can answer *about* people, not just *to* the author. Name matching is
  whole-word and nickname-tolerant (`nos` matches speaker `nos_yous`).
* **Image understanding** — the first image in a ping (upload, link, or
  replied-to message) is downscaled to 512px and described by a separate
  local-first vision chain; descriptions are cached by source URL.
  Videos are never processed, but history lines are annotated
  (`[image: foo.png]`, `[video: bar.mp4]`) so the model knows media exists.
* **Keyword recall** — FTS5 search over past messages, top 3 injected.
  When a ping replies to a message outside the recent window, this block is
  replaced by that message plus preceding context (DB first, Discord fetch
  fallback; fetched messages are stored so memory keeps continuity).

`/clearmemory` clears only the recent short-term buffer (to unstick a looping
model); summaries and facts are preserved. `/status` reports buffer,
summary, fact and DB sizes plus model load states. `/reload` re-reads
`config.json` without restarting (admin only, ephemeral reply): model
names, temperatures, token budgets, timeouts, prompts, and presence apply
immediately — including the OpenRouter key and `style_profile.txt`.
Changed keys are listed; token, DB path, log dir, and slash-command
names need a restart and are reported as such. A broken config keeps the
old one with an error message.

---

## Prompt assembly

Per reply, from fattest to leanest section (see `[prompt] sections` in the
debug log):

1. `Style summary:` — static profile from `style_profile.txt` (or generic
   fallback lines if missing)
2. `Examples:` — top-ranked style examples (default 10, token-budgeted)
3. Long-term memory blocks (summaries, recalled messages, user facts)
4. Web results (if a search trigger fired)
5. Recent conversation window (token-budgeted) — lines render as
   `account [display]: message`, with the bracket skipped when the
   server nickname equals the account name (same format in the
   replied-to context block)

---

## Debugging

Terminal output stays at INFO (one line per model call). Full detail goes to
`logs/` (gitignored):

* `logs/bot.log` — rotating file log at DEBUG, including full prompts
  (`PROMPT SENT TO MODEL … [END PROMPT]`) and per-section sizes.
* `logs/llm.jsonl` — one pretty-printed JSON object per LLM call with full
  request/response bodies, purpose tag (`chat`, `prompt`, `summary`,
  `facts`, `vision`), latency and status. Never contains HTTP headers
  (no API keys); image bytes are logged as size placeholders, not blobs.
  Rotated with `.1`, `.2` backups.

Useful commands:

```bash
tail -f logs/bot.log
venv/bin/python -c "
from memory.llmlog import read_records
import json; print(json.dumps(read_records('logs/llm.jsonl')[-1], indent=2))"
```

### 🔍 reaction debug

React with 🔍 to any bot reply and the full generation behind it is DM'd
to you as a markdown file: backend + model (incl. fallback flag),
latency/attempts, params, the complete prompt messages, reasoning (only
if the response already contained it), the sent reply text, and usage
counters. Works for both Ollama and OpenRouter answers. Anyone can use
it. Messages with no stored record (older than `debug_record_keep`
replies, or sent before the last restart) get an ❌ reaction instead.
If your DMs are closed you'll get a short in-channel notice.
Tune via `debug_reaction_emoji`, `debug_max_file_chars`,
`debug_record_keep`.

---

## Tests

```bash
venv/bin/python -m unittest discover -s tests -v
```

Covers the `memory/` package (store, buffer, recall, facts, summaries,
examples, log dump, style profile, vision) with stdlib `unittest` only.
`bot.py` itself is not imported by tests.

---

## Project Structure

```
bot.py             Discord bot (events, prompt assembly, generation)
memory/            Persistent memory package (SQLite + retrieval helpers)
  store.py         Schema, message/summary/fact/image-cache persistence (WAL + FTS5)
  buffer.py        Token-aware short-term window
  recall.py        Keyword recall, peer-user detection, fact ranking
  summary.py       Summarizer and fact-extraction prompt builders
  web.py           Web-result rendering for Components V2
  facts.py         Fact JSON parsing + speaker resolution
  examples.py      Style-example compaction + token budgeting
  llmlog.py        Full-bodied LLM call dump (llm.jsonl)
  styleprofile.py  Corpus sampling + distillation helpers
  vision.py        Image detection, downscaling, prompt markers
style_profile.py   Manual static style-profile generator
converter.py       Converts Discord exports into a dataset
embed.py           Builds the HNSW embedding index
config.json        Configuration (keep real tokens out of git)
tests/             Unit tests (stdlib only)
venv/              Virtualenv (gitignored)
logs/              Debug logs (gitignored)
memory.db          Runtime memory DB (gitignored)
index.bin          HNSW index (gitignored, built locally)
texts.json         Indexed examples (gitignored, built locally)
style_profile.txt  Static style profile (generated + hand-edited)
```

---

## License

Copyright (c) 2026 MrAnvil935. Licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE)
([canonical text](https://polyformproject.org/licenses/noncommercial/1.0.0)).

In plain words: you may use, modify, and share this code for free for
personal, educational, and research purposes. Any commercial use — selling
it, hosting it as a paid service, bundling it into a commercial product,
or otherwise using it for commercial advantage — requires separate
permission from the copyright holder (contact via the GitHub repository).

The license covers the code. Personal chat data (`texts.json`,
`memory.db`, `style_profile.txt`) is gitignored and never distributed
with the repository.

---

## Notes

* This project is intended for personal and educational use.
* Do not commit `config.json` with real tokens, `venv/`, `logs/`,
  `memory.db`, `index.bin`, `texts.json` or your `style_profile.txt`
  (persona-specific) — see `.gitignore`.
* The quality of the bot depends heavily on the amount of Discord messages available, the prompt, and the language model being used. Larger models generally produce noticeably better results.
* The bot has been tested on both desktop Linux and **Termux**. It should run anywhere Python and the required dependencies are supported.
* This project was developed with significant assistance from generative AI tools. While all code has been reviewed and adapted for this project, AI played a major role in development, debugging, and refinement.
