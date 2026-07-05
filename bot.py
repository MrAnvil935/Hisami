import discord
import requests
import json
import numpy as np
import hnswlib
import random
import traceback
import os
import time
import re
import asyncio
from collections import defaultdict

START_TIME = time.time()

IMAGE_FOLDER = "images"
VALID_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp"
)

# ============================================================
# CONFIG
# ============================================================

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# You can change those in config

DISCORD_TOKEN = config["discord_token"]

OPENROUTER_API_KEY = config["openrouter_api_key"]
MODEL = config.get("model", "openrouter/free")

MASTER_PROMPT = config["master_prompt"]

MAX_HISTORY = config["max_history"]
MAX_EXAMPLES = config["max_examples"]

OLLAMA_URL = config["ollama_url"]
OLLAMA_MODEL = config["ollama_model"]
MAX_OLLAMA_TOKENS = config["ollama_max_tokens"]
OLLAMA_TIMEOUT = config["ollama_timeout"]

BOTNAME = config["botname"]

CLEAR_COMMAND_NAME = config["clear_command_name"]
CLEAR_COMMAND_DESCRIPTION = config["clear_command_description"]
CLEAR_COMMAND_TEXT = config["clear_command_text"]
RANDOMIMAGE_COMMAND_NAME = config["randomimage_command_name"]
RANDOMIMAGE_COMMAND_DESCRIPTION = config["randomimage_command_description"]
RANDOMIMAGE_COMMAND_TEXT = config["randomimage_command_text"]
STATUS_COMMAND_NAME = config["status_command_name"]
STATUS_COMMAND_DESCRIPTION = config["status_command_description"]

# Don't touch those unless you know what you are doing

DIM = 768  # nomic-embed-text embedding size

EMBED_MODEL = "nomic-embed-text"
ASSISTANT_NAME = "Assistant" # I would leave it as it is or it can cause issues with output quality

# ============================================================
# LOAD HNSW
# ============================================================

try:
    # --- load texts ---
    with open("texts.json", "r", encoding="utf-8") as f:
        indexed_texts = json.load(f)

    # --- rebuild HNSW index object ---
    index = hnswlib.Index(space="l2", dim=DIM)

    # you MUST know max_elements used during build
    index.init_index(max_elements=len(indexed_texts), ef_construction=200, M=16)

    # --- load saved index ---
    index.load_index("index.bin")

    # optional but recommended for query speed/quality
    index.set_ef(50)

    print(f"Loaded HNSW index with {len(indexed_texts)} entries")

except Exception as e:
    print("Index load failed:", e)
    index = None
    indexed_texts = []

# ============================================================
# MEMORY
# ============================================================

conversation_history = defaultdict(list)

def add_message(channel_id, message_id, author, role, content, reply_to=None):
    conversation_history[channel_id].append({
        "id": message_id,
        "author": author,
        "role": role,
        "content": content,
        "reply_to": reply_to
    })

    conversation_history[channel_id] = conversation_history[channel_id][-MAX_HISTORY:]

# ============================================================
# EMBEDDING
# ============================================================

def get_embedding(text):
    res = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=30
    )
    res.raise_for_status()
    return res.json()["embeddings"][0]

# ============================================================
# RETRIEVAL
# ============================================================

def build_search_query(channel_id, user_message):
    """
    Build an embedding query from the recent conversation.
    Older messages contribute less because we only keep the
    last ~8 exchanges.
    """

    history = conversation_history[channel_id][-8:]

    parts = []

    for msg in history:
        parts.append(f"{msg['author']}: {msg['content']}")

    parts.append(user_message)

    return "\n".join(parts)


def embedding_search(search_text, k):
    if index is None:
        return []

    vec = np.array([get_embedding(search_text)], dtype="float32")

    labels, distances = index.knn_query(vec, k=min(k * 6, len(indexed_texts)))

    results = []

    for idx, dist in zip(labels[0], distances[0]):

        if not (0 <= idx < len(indexed_texts)):
            continue

        # HNSW L2 distance
        similarity = 1.0 / (1.0 + dist)

        results.append({
            "text": indexed_texts[idx],
            "score": similarity
        })

    return results


def keyword_search(search_text, k):
    q = set(search_text.lower().split())

    results = []

    for text in indexed_texts:

        words = set(text.lower().split())

        overlap = len(q & words)

        if overlap == 0:
            continue

        results.append({
            "text": text,
            "score": overlap
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:k]


def retrieve_examples(channel_id, user_message, limit=MAX_EXAMPLES):

    if not indexed_texts:
        return []

    search_text = build_search_query(channel_id, user_message)

    semantic = embedding_search(search_text, limit)
    keyword = keyword_search(search_text, limit)

    merged = {}

    # semantic is worth more
    for item in semantic:
        merged[item["text"]] = item["score"] * 2.0

    # keyword boosts existing score
    for item in keyword:
        merged[item["text"]] = merged.get(item["text"], 0) + item["score"] * 0.5

    if not merged:
        return []

    # --------------------------------------------------------
    # Score + Temperature
    # --------------------------------------------------------

    TEMPERATURE = 0.30
    # 0.0 = deterministic
    # 0.2 = tiny randomness
    # 0.3 = recommended
    # 0.5 = noticeable randomness
    # 1.0 = almost random

    scored = []

    for text, score in merged.items():

        noisy = score * random.uniform(
            1.0 - TEMPERATURE,
            1.0 + TEMPERATURE
        )

        scored.append((noisy, text))

    scored.sort(reverse=True)

    return [text for _, text in scored[:limit]]

# ============================================================
# PROMPT
# ============================================================

def build_prompt(channel_id, user_message, username):

    examples = retrieve_examples(
      channel_id,
      user_message
 )

    prompt = f"""
{MASTER_PROMPT}

STYLE PROFILE:
- Casual Discord language
- Short responses
- Slang-heavy

"""

    if examples:
        prompt += "\nExamples:\n"
        for ex in examples:
            prompt += f"- {ex}\n"

    prompt += "\nConversation:\n"

    # Step 3: build lookup table
    msg_map = {
        m["id"]: m
        for m in conversation_history[channel_id]
    }

    # Step 4: inject reply context
    for m in conversation_history[channel_id]:

        text = f"{m['author']}: {m['content']}"

        if m.get("reply_to"):
            parent = msg_map.get(m["reply_to"])

            if parent:
                text = (
                    f"{m['author']} "
                    f"(replying to {parent['author']}: {parent['content']}): "
                    f"{m['content']}"
                )

        prompt += text + "\n"

    prompt += f"\n{username}: {user_message}\n{ASSISTANT_NAME}:"

    return prompt

# ============================================================
# GENERATION
# ============================================================
def strip_thinking(text):
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    return text.strip()


def is_ollama_model_loaded():
    try:
        r = requests.get("http://localhost:11434/api/ps", timeout=2)
        r.raise_for_status()

        data = r.json()

        for model in data.get("models", []):
            if model["name"].startswith(OLLAMA_MODEL):
                return True

    except Exception:
        pass

    return False
def generate_ollama_response(prompt):
    print("\n" + "=" * 80)
    print("PROMPT SENT TO MODEL")
    print("=" * 80)
    print(prompt)
    print("=" * 80 + "\n")

    payload = {
    "model": OLLAMA_MODEL,
    "messages": [
        {
            "role": "system",
            "content": MASTER_PROMPT
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    "options": {
        "temperature": 0.9,
        "top_p": 0.95,
        "num_ctx": MAX_OLLAMA_TOKENS
    },
    "think": False,
    "stream": False,
    "keep_alive": "30m"
}

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)

        if r.status_code == 200:
            data = r.json()

            # Debug once if you want to inspect the response:
            # print(json.dumps(data, indent=2))

            text = data["message"]["content"]
            return strip_thinking(text)

    except Exception as e:
        print("Ollama failed:", e)

    return None

def generate_openrouter_response(prompt, use_fallback=False):

    print("\n" + "=" * 80)
    print("PROMPT SENT TO MODEL")
    print("=" * 80)
    print(prompt)
    print("=" * 80 + "\n")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Discord RAG Bot"
    }

    model_to_use = config["fallback_model"] if use_fallback else MODEL

    payload = {
        "model": model_to_use,
        "messages": [
            {
                "role": "system",
                "content": MASTER_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.9,
        "top_p": 0.95
    }

    MAX_RETRIES = 5

    for attempt in range(MAX_RETRIES):

        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )

        # ---- DEBUG OUTPUT ----
        print(f"\n[{model_to_use}] Attempt {attempt + 1}/{MAX_RETRIES}")
        print("Status:", r.status_code)
        print("Response:")
        print(r.text)
        print("-" * 80)
        # ----------------------

        data = r.json()

        # OpenRouter sometimes returns HTTP 200 with an embedded error.
        if "error" in data:
            error = data["error"]
            message = error.get("message", "")
            code = error.get("code")

            print(f"Embedded OpenRouter error ({code}): {message}")

            # Treat temporary upstream failures as retryable.
            if code in (429, 502, 503, 504):
                delay = (1.5 ** attempt) + random.uniform(0, 1)
                print(
                    f"[{model_to_use}] Embedded HTTP {code} → "
                    f"retry {attempt + 1}/{MAX_RETRIES} in {delay:.2f}s"
                )
                time.sleep(delay)
                continue

            # Daily quota exhausted
            if "free-models-per-day" in message:
                return None

            # Anything else: give up on this model
            return None

        # Normal successful response
        if r.status_code == 200:
            text = data["choices"][0]["message"]["content"].strip()

            if use_fallback:
                text = f"-# [fallback: {model_to_use}]\n{text}"

            return text

        # Handle non-429 errors immediately
        if r.status_code not in (429, 502, 503, 504):
            r.raise_for_status()

        # Parse OpenRouter error message
        try:
            error = r.json().get("error", {})
            message = error.get("message", "")
        except Exception:
            message = ""

        # Don't retry if the daily free quota is exhausted
        if "free-models-per-day" in message:
            print(f"[{model_to_use}] Daily free quota exhausted.")
            return None

        # Retry temporary rate limits
        retry_after = r.headers.get("Retry-After")

        if retry_after:
            delay = float(retry_after)
        else:
            delay = (1.5 ** attempt) + random.uniform(0, 1)

        print(
            f"[{model_to_use}] Rate limited → retry "
            f"{attempt + 1}/{MAX_RETRIES} in {delay:.2f}s"
)

        time.sleep(delay)

    return None

# ============================================================
# DISCORD
# ============================================================

def is_embed_model_available():
    try:
        r = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": EMBED_MODEL,
                "input": "test"
            },
            timeout=3
        )
        return r.status_code == 200
    except:
        return False


def format_uptime():
    secs = int(time.time() - START_TIME)

    days, secs = divmod(secs, 86400)
    hours, secs = divmod(secs, 3600)
    minutes, secs = divmod(secs, 60)

    parts = []

    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")

    parts.append(f"{secs}s")

    return " ".join(parts)

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

@tree.command(name=CLEAR_COMMAND_NAME, description=CLEAR_COMMAND_DESCRIPTION)
async def clear_memory(interaction: discord.Interaction):
    await interaction.response.defer()
    channel_id = interaction.channel_id

    conversation_history[channel_id].clear()

    await interaction.followup.send(
        {CLEAR_COMMAND_TEXT},
        ephemeral=False  # set True if you want only the user to see it
    )
    
@tree.command(
    name=STATUS_COMMAND_NAME,
    description=STATUS_COMMAND_DESCRIPTION
)
async def status(interaction: discord.Interaction):
    await interaction.response.defer()

    image_count = len([
        f for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(VALID_IMAGE_EXTENSIONS)
    ])

    memory_count = len(
        conversation_history[interaction.channel_id]
    )

    ping = round(client.latency * 1000)

    text = (
        f"## {BOTNAME} status\n\n"

        f"**Ollama model**\n"
        f"- {OLLAMA_MODEL}\n"
        f"- {'🟢 Loaded' if is_ollama_model_loaded() else '🔴 Not loaded'}\n\n"

        f"**Embedding model**\n"
        f"- {EMBED_MODEL}\n"
        f"- {'🟢 Reachable' if is_embed_model_available() else '🔴 Offline'}\n\n"

        f"**OpenRouter main**\n"
        f"- {MODEL}\n\n"

        f"**OpenRouter fallback**\n"
        f"- {config['fallback_model']}\n\n"

        f"**HNSW index**\n"
        f"- {'🟢 Loaded' if index is not None else '🔴 Missing'}\n"
        f"- {len(indexed_texts)} entries\n\n"

        f"**Ping**\n"
        f"- {ping} ms\n\n"

        f"**Uptime**\n"
        f"- {format_uptime()}\n\n"

        f"**Random images**\n"
        f"- {image_count}\n\n"

        f"**Conversation memory**\n"
        f"- {memory_count}/{MAX_HISTORY}"
    )

    await interaction.followup.send(text)
@tree.command(name=RANDOMIMAGE_COMMAND_NAME, description=RANDOMIMAGE_COMMAND_DESCRIPTION)
async def random_image(interaction: discord.Interaction):
    await interaction.response.defer()

    image_folder = "images"

    # Supported image extensions
    valid_extensions = (".png", ".jpg", ".jpeg", ".gif", ".webp")

    images = [
        f for f in os.listdir(image_folder)
        if f.lower().endswith(valid_extensions)
    ]

    if not images:
        await interaction.followup.send(
            "No images found.",
            ephemeral=False
        )
        return

    chosen = random.choice(images)
    path = os.path.join(image_folder, chosen)

    try:
        await interaction.user.send(file=discord.File(path))

        await interaction.followup.send(
            RANDOMIMAGE_COMMAND_TEXT,
            ephemeral=False
        )

    except discord.Forbidden:
        await interaction.followup.send(
            "I couldn't DM you. Please enable DMs from server members.",
            ephemeral=False
        )

@client.event
async def on_ready():
    await tree.sync()
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message):

    if message.author.bot:
        return

    reply_to = None

    if message.reference and message.reference.message_id:
        reply_to = message.reference.message_id

    add_message(
        message.channel.id,
        message.id,
        str(message.author),
        "user",
        message.content,
        reply_to
    )

    if client.user not in message.mentions:
        return

    cleaned = (
        message.content
        .replace(f"<@{client.user.id}>", "")
        # .replace(f"<@!{client.user.id}>", "")
        .strip()
    )

    if not cleaned:
        await message.reply("Say something after pinging me.")
        return
    try:
        async with message.channel.typing():
            # Build prompt once
            prompt = build_prompt(
            message.channel.id,
            cleaned,
            str(message.author)
            )

            reply = None

            # 1) Use Ollama only if the model is already loaded
            if is_ollama_model_loaded():
                print("Using loaded Ollama model")
                reply = await asyncio.to_thread(
                    generate_ollama_response,
                    prompt
                    )

            # 2) OpenRouter primary model
            if reply is None:
                print(f"Using OpenRouter ({MODEL})")
                reply = await asyncio.to_thread(
                    generate_openrouter_response,
                    prompt,
                    use_fallback=False
                )

            # 3) OpenRouter fallback model
            if reply is None:
                print(f"Using OpenRouter fallback ({config['fallback_model']})")
                reply = await asyncio.to_thread(
                    generate_openrouter_response,
                    prompt,
                    use_fallback=True
                )

            # 4) FINAL SAFETY NET
            if reply is None:
                reply = "All models are currently unavailable 💀"

            if len(reply) > 1900:
                reply = reply[:1900] + "..."

            sent = await message.reply(reply)

            add_message(
                message.channel.id,
                sent.id,
                ASSISTANT_NAME,
                "assistant",
                reply,
                reply_to=message.id
            )

    except Exception:
        tb = traceback.format_exc()
        await message.reply(f"Error:\n```{tb[-1500:]}```")
client.run(DISCORD_TOKEN)
