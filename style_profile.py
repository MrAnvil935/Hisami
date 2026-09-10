"""One-shot static style-profile generator (manual use only).

Samples Target Responses from texts.json, distills them into a persona
profile via the LOCAL summary Ollama model, and writes style_profile.txt
for human review/editing. Never touches OpenRouter and never runs inside
the bot — re-run deliberately (e.g. after re-embedding a new corpus).

Usage:
    venv/bin/python style_profile.py [--samples 60] [--seed 2]
                                      [--out style_profile.txt]
"""

import argparse
import json
import sys

import requests

from memory import styleprofile as sp


"""One-shot static style-profile generator (manual use only).

Samples Target Responses from texts.json, distills them into a persona
profile via the LOCAL summary Ollama model (OpenRouter summary model as
fallback), and writes style_profile.txt for human review/editing. Never
runs inside the bot — re-run deliberately (e.g. after re-embedding a new
corpus).

Usage:
    venv/bin/python style_profile.py [--samples 60] [--seed 2]
                                      [--out style_profile.txt]
                                      [--no-openrouter]
"""

import argparse
import json
import sys
import time

import requests

from memory import styleprofile as sp

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_config():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_messages(samples):
    return [
        {"role": "system", "content": sp.DISTILL_SYSTEM},
        {"role": "user", "content": sp.build_distill_prompt(samples)},
    ]


def distill_ollama(samples, ollama_url, model, timeout=120):
    payload = {
        "model": model,
        "messages": build_messages(samples),
        "options": {"temperature": 0.2, "top_p": 0.95, "num_ctx": 4096},
        "think": False,
        "stream": False,
    }
    r = requests.post(ollama_url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def build_openrouter_payload(samples, model, temperature=0.2):
    # Payload only — the Authorization header is attached at send time
    # and never logged or written anywhere.
    return {
        "model": model,
        "messages": build_messages(samples),
        "temperature": temperature,
        "top_p": 0.95,
        "max_tokens": 1000,
    }


def distill_openrouter(samples, api_key, model, temperature=0.2,
                       timeout=60, tries=3):
    payload = build_openrouter_payload(samples, model, temperature)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Hisami style-profile generator",
    }
    last = None
    for attempt in range(1, tries + 1):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers,
                              json=payload, timeout=timeout)
            data = r.json()
            if data.get("error"):
                last = f"API error: {data['error'].get('message', data['error'])}"
            elif r.status_code == 200:
                return data["choices"][0]["message"]["content"].strip()
            else:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last = str(e)
        if attempt < tries:
            time.sleep(2 ** attempt)
    raise RuntimeError(last or "unknown OpenRouter failure")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default="style_profile.txt")
    ap.add_argument("--no-openrouter", action="store_true",
                    help="do not fall back to OpenRouter, Ollama only")
    args = ap.parse_args()

    config = load_config()
    ollama_url = config["ollama_url"]
    ollama_model = config.get("summary_ollama_model", "gemma3n:e4b")

    with open("texts.json", "r", encoding="utf-8") as f:
        texts = json.load(f)

    responses = sp.extract_responses(texts)
    print(f"extracted {len(responses)} responses from {len(texts)} records")
    samples = sp.sample_responses(responses, n=args.samples, seed=args.seed)
    print(f"sampled {len(samples)} (seed={args.seed})")

    backend = "ollama"
    try:
        profile = distill_ollama(samples, ollama_url, ollama_model)
        print(f"distilled via local Ollama ({ollama_model})")
    except Exception as e:
        if args.no_openrouter:
            print(f"Ollama distillation failed ({e}).", file=sys.stderr)
            print("Start Ollama and pull the summary model, then re-run.",
                  file=sys.stderr)
            sys.exit(1)
        or_model = config.get("summary_model", "openrouter/free")
        print(f"Ollama failed ({e}), falling back to OpenRouter ({or_model})...")
        try:
            profile = distill_openrouter(
                samples,
                config["openrouter_api_key"],
                or_model,
                temperature=config.get("summary_temperature", 0.2),
                timeout=config.get("summary_timeout", 60),
            )
            backend = "openrouter"
            print(f"distilled via OpenRouter ({or_model})")
        except Exception as e2:
            print(f"OpenRouter fallback also failed ({e2}).", file=sys.stderr)
            sys.exit(1)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(profile + "\n")

    print(f"wrote {args.out} ({len(profile)} chars, via {backend})")
    print("Review and hand-edit it before restarting the bot.")


if __name__ == "__main__":
    main()
