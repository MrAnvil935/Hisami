# Discord RAG Mimic Bot

A simple Discord chatbot that mimics a specific user's chatting style using Retrieval-Augmented Generation (RAG).

The bot builds an embedding index from exported Discord messages and uses it as style examples during generation. It supports:

* OpenRouter as the primary inference backend
* Ollama as an optional local fallback
* Local embeddings using `nomic-embed-text`
* HNSW vector search for fast example retrieval
* Per-channel conversation memory
* Slash commands for status information and maintenance

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
python converter.py
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
python embed.py
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

You can also customize:

* primary model
* fallback model
* prompt
* retrieval settings
* history size
* other runtime options

---

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Start the bot

```bash
python bot.py
```

Make sure ollama server is running as at minimum you need access to nomic-embed-text. If you want to also use local model for generation you have to load it manually and then bot will try to use it as 1st option. 

---

## Project Structure

```
converter.py     Converts Discord exports into a dataset
embed.py         Builds the HNSW embedding index
bot.py           Discord bot
config.json      Configuration
dataset.jsonl    Training dataset
index.bin        HNSW index
texts.json       Indexed examples
```

---

## Dependencies

```
aiohappyeyeballs==2.6.2
aiohttp==3.14.1
aiosignal==1.4.0
attrs==26.1.0
audioop-lts==0.2.2
certifi==2026.6.17
charset-normalizer==3.4.7
discord==2.3.2
discord.py==2.7.1
frozenlist==1.8.0
hnswlib==0.8.0
idna==3.18
multidict==6.7.1
numpy==2.5.0
propcache==0.5.2
requests==2.34.2
tqdm==4.68.3
urllib3==2.7.0
yarl==1.24.2
```

Or install them directly from `requirements.txt`.

---

## Notes

* This project is intended for personal and educational use.
* The quality of the bot depends heavily on the amount of Discord messages available, the prompt, and the language model being used. Larger models generally produce noticeably better results.
* The bot has been tested on both desktop Linux and **Termux**. It should run anywhere Python and the required dependencies are supported.
* This project was developed with significant assistance from generative AI tools. While all code has been reviewed and adapted for this project, AI played a major role in development, debugging, and refinement.

