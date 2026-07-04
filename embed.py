import json
import requests
import numpy as np
import hnswlib
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

INPUT_FILE = "dataset.jsonl"
INDEX_FILE = "index.bin"   # changed extension (optional)
TEXTS_FILE = "texts.json"

DIM = 768  # nomic embedding size

# HNSW params (safe defaults)
M = 16
EF_CONSTRUCTION = 200

# ============================================================
# EMBEDDING
# ============================================================

def get_embedding(text):
    res = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": text
        }
    )
    res.raise_for_status()
    return res.json()["embeddings"][0]

# ============================================================
# BUILD TEXT (WITH WEIGHTING)
# ============================================================

def build_text(record):

    context = "\n".join(
        f"{m['author']}: {m['content']}"
        for m in record.get("context", [])
    )

    if record.get("type") == "reply":
        prefix = "HIGH VALUE INTERACTION (conversation reply, strong behavioral signal)"
    else:
        prefix = "STYLE EXAMPLE (tone, phrasing, personality)"

    input_text = record.get("input")
    if input_text == "reply_detected":
        input_text = "implicit reply in conversation"

    return f"""
{prefix}

Context:
{context}

User Input:
{input_text}

Target Response:
{record.get("response")}
""".strip()

# ============================================================
# MAIN
# ============================================================

def main():

    texts = []
    embeddings = []

    total = 0
    skipped = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Building index"):
            record = json.loads(line)

            text = build_text(record)

            try:
                emb = get_embedding(text)

                embeddings.append(emb)
                texts.append(text)
                total += 1

            except Exception as e:
                print(f"Error embedding: {e}")
                skipped += 1
                continue

    # ===== BUILD INDEX =====
    print("Initializing HNSW index...")

    index = hnswlib.Index(space='l2', dim=DIM)
    index.init_index(
        max_elements=len(embeddings),
        ef_construction=EF_CONSTRUCTION,
        M=M
    )

    index.add_items(
        np.array(embeddings).astype("float32"),
        np.arange(len(embeddings))
    )

    # optional but recommended for search quality
    index.set_ef(50)

    # ===== SAVE =====
    index.save_index(INDEX_FILE)

    with open(TEXTS_FILE, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    print("\n==============================")
    print(f"Indexed: {total}")
    print(f"Skipped: {skipped}")
    print("==============================")

# ============================================================

if __name__ == "__main__":
    main()
