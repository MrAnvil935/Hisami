import json

# ==========================
# CONFIG
# ==========================

INPUT_FILE = "export.json"
OUTPUT_FILE = "dataset.jsonl"

TARGET_USER_ID = "put the id of the person you want to use as personality base here"
CONTEXT_SIZE = 3

# ==========================
# SAFE STREAM PARSER
# ==========================

def load_messages_safe(path):

    decoder = json.JSONDecoder()
    messages = []

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # find start of messages array
    start = text.find('"messages"')
    if start == -1:
        print("No messages field found")
        return []

    start = text.find("[", start)
    if start == -1:
        print("No messages array found")
        return []

    i = start + 1  # move past '['

    while i < len(text):

        # skip whitespace / commas
        while i < len(text) and text[i] in " \n\r\t,":
            i += 1

        if i >= len(text) or text[i] == "]":
            break

        try:
            obj, end = decoder.raw_decode(text, i)
            messages.append(obj)
            i = end
        except json.JSONDecodeError:
            print(f"Stopped at position {i} (likely truncated JSON)")
            break

    print(f"Recovered {len(messages)} messages")
    return messages


messages = load_messages_safe(INPUT_FILE)

# ==========================
# MERGE SPLIT MESSAGES
# ==========================

def merge_messages(messages):

    merged = []
    buffer = None

    for msg in messages:

        content = msg.get("content", "").strip()
        if not content:
            continue

        if (
            buffer
            and buffer["author"]["id"] == msg["author"]["id"]
            and msg["timestamp"][:16] == buffer["timestamp"][:16]
        ):
            buffer["content"] += " " + content
        else:
            if buffer:
                merged.append(buffer)
            buffer = msg.copy()

    if buffer:
        merged.append(buffer)

    return merged


messages = merge_messages(messages)

# ==========================
# MAP IDS
# ==========================

msg_map = {m.get("id"): m for m in messages if "id" in m}

# ==========================
# HELPERS
# ==========================

def clean_content(msg):
    return msg.get("content", "").strip()

def build_context(index):
    ctx = []

    i = index - 1
    while i >= 0 and len(ctx) < CONTEXT_SIZE:
        m = messages[i]

        content = clean_content(m)

        if content and len(content) > 2:
            ctx.append({
                "author": m["author"]["name"],
                "content": content
            })

        i -= 1

    return list(reversed(ctx))


# ==========================
# MAIN LOOP
# ==========================

out_count = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for i, msg in enumerate(messages):

        if msg["author"]["id"] != TARGET_USER_ID:
            continue

        content = clean_content(msg)

        if not content or len(content) < 3:
            continue

        context = build_context(i)

        # reply detection
        is_reply = False

        if msg.get("reference") and msg["reference"].get("messageId"):
            ref_id = msg["reference"]["messageId"]

            if ref_id in msg_map:
                is_reply = True

        entry = {
            "type": "reply" if is_reply else "standalone",
            "input": "reply_detected" if is_reply else None,
            "response": content,
            "context": context,
            "channel": "unknown",
            "timestamp": msg["timestamp"]
        }

        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        out_count += 1

        if out_count % 1000 == 0:
            print(f"Saved {out_count} entries...")

print(f"Done. Saved {out_count} entries.")
