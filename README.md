# Discord RAG Mimic Bot

A Discord chatbot that mimics a specific user's chatting style using Retrieval-Augmented Generation (RAG).

The bot builds an embedding index from exported Discord messages and uses it as style examples during generation. It supports:

* OpenRouter as the primary inference backend, Ollama as local-first option
* Local embeddings using `nomic-embed-text`
* HNSW vector search for fast example retrieval
* Persistent per-channel conversation memory (SQLite)
* Long-term memory: channel summaries + per-user facts + keyword recall
* Static style profile distilled once from the embedding corpus
* Web search (DuckDuckGo) for questions about current events
* Slash commands for status information and maintenance
* Two-sink debug logging: concise terminal + full payloads on disk

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

Edit `config.json`.

At minimum you'll need:

* Discord bot token
* OpenRouter API key

Tunable groups (all have built-in defaults, see `config.json`):

* `model` / `fallback_model` — main chat chain (local Ollama first, then OpenRouter)
* `summary_ollama_model` / `summary_model` — background chain for memory
  summarization and fact extraction (local first, OpenRouter last resort)
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
so it survives restarts. On top of that, three long-term layers feed every
prompt:

* **Channel summaries** — every 30 messages per channel are summarized by the
  background chain into `summaries` (or earlier if the backlog passes a token
  budget, so long messages don't bloat the raw window); the 2 most relevant
  are injected.
* **Per-user facts** — durable traits extracted for all speakers batched
  on each summary chunk; top 5 per user are injected.
* **Peer recall** — when someone mentions, names, or chats alongside other
  users, their facts are injected too (`Known about <name>:`), so the bot
  can answer *about* people, not just *to* the author.
* **Keyword recall** — FTS5 search over past messages, top 3 injected.

`/clearmemory` clears only the recent short-term buffer (to unstick a looping
model); summaries and facts are preserved. `/status` reports buffer,
summary, fact and DB sizes plus model load states.

---

## Prompt assembly

Per reply, from fattest to leanest section (see `[prompt] sections` in the
debug log):

1. `Style summary:` — static profile from `style_profile.txt` (or generic
   fallback lines if missing)
2. `Examples:` — top-ranked style examples (default 10, token-budgeted)
3. Long-term memory blocks (summaries, recalled messages, user facts)
4. Web results (if a search trigger fired)
5. Recent conversation window (token-budgeted)

---

## Debugging

Terminal output stays at INFO (one line per model call). Full detail goes to
`logs/` (gitignored):

* `logs/bot.log` — rotating file log at DEBUG, including full prompts
  (`PROMPT SENT TO MODEL … [END PROMPT]`) and per-section sizes.
* `logs/llm.jsonl` — one pretty-printed JSON object per LLM call with full
  request/response bodies, purpose tag (`chat`, `prompt`, `summary`,
  `facts`), latency and status. Never contains HTTP headers (no API keys).
  Rotated with `.1`, `.2` backups.

Useful commands:

```bash
tail -f logs/bot.log
venv/bin/python -c "
from memory.llmlog import read_records
import json; print(json.dumps(read_records('logs/llm.jsonl')[-1], indent=2))"
```

---

## Tests

```bash
venv/bin/python -m unittest discover -s tests -v
```

Covers the `memory/` package (store, buffer, recall, facts, summaries,
examples, log dump, style profile) with stdlib `unittest` only. `bot.py`
itself is not imported by tests.

---

## Project Structure

```
bot.py             Discord bot (events, prompt assembly, generation)
memory/            Persistent memory package (SQLite + retrieval helpers)
  store.py         Schema, message/summary/fact persistence (WAL + FTS5)
  buffer.py        Token-aware short-term window
  recall.py        Keyword recall over past messages
  summary.py       Summarizer and fact-extraction prompt builders
  facts.py         Fact JSON parsing + speaker resolution
  examples.py      Style-example compaction + token budgeting
  llmlog.py        Full-bodied LLM call dump (llm.jsonl)
  styleprofile.py  Corpus sampling + distillation helpers
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

## Notes

* This project is intended for personal and educational use.
* Do not commit `config.json` with real tokens, `venv/`, `logs/`,
  `memory.db`, `index.bin` or `texts.json` — see `.gitignore`.
* The quality of the bot depends heavily on the amount of Discord messages available, the prompt, and the language model being used. Larger models generally produce noticeably better results.
* The bot has been tested on both desktop Linux and **Termux**. It should run anywhere Python and the required dependencies are supported.
* This project was developed with significant assistance from generative AI tools. While all code has been reviewed and adapted for this project, AI played a major role in development, debugging, and refinement.
