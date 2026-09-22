"""Magnifying-glass debug: normalize + format one generation for DM delivery.

bot.py keeps a message-id -> record map (populated when a chat reply is
sent) and the on_raw_reaction_add handler. This module only shapes the
data so it stays testable without Discord or network:

- normalize_record() flattens one backend call (Ollama or OpenRouter)
  into a small JSON-safe dict. Reasoning is display-only: whatever the
  stored response body already contains (OpenRouter `reasoning` /
  `reasoning_details`, Ollama `thinking`) is surfaced, never requested.
- format_debug_record() renders that dict as readable markdown text for
  the DM'd file. Missing keys never crash; oversized bodies are cut
  with a [truncated] marker so the header/reply/usage always survive.
"""

REASONING_MAX_CHARS = 20000


def _as_text(content):
    """Render message content (str or OpenAI-style part list) as text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(
                    part.get("text"), str):
                parts.append(part["text"])
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                parts.append(f"[image {len(str(url))} chars]")
            else:
                parts.append(f"[non-text part: {part.get('type', '?')}]")
        return "\n".join(parts)
    return str(content)


def _extract_reasoning(backend, response):
    """Reasoning text already present in the response body, else ''."""
    try:
        if backend == "openrouter":
            msg = (response.get("choices") or [{}])[0].get("message") or {}
            bits = []
            if isinstance(msg.get("reasoning"), str) and msg["reasoning"]:
                bits.append(msg["reasoning"])
            details = msg.get("reasoning_details")
            if isinstance(details, list):
                for item in details:
                    if not isinstance(item, dict):
                        continue
                    for key in ("text", "summary"):
                        val = item.get(key)
                        if isinstance(val, str) and val:
                            bits.append(val)
                            break
            return "\n\n".join(bits)
        msg = (response or {}).get("message") or {}
        thinking = msg.get("thinking", "")
        return thinking if isinstance(thinking, str) else ""
    except Exception:
        return ""


def _extract_usage(backend, response):
    """Token/cost counters already present in the response body, else {}."""
    try:
        if backend == "openrouter":
            usage = (response or {}).get("usage") or {}
            return {k: v for k, v in usage.items()
                    if isinstance(v, (int, float))
                    and not isinstance(v, bool)}
        body = response or {}
        usage = {}
        for key in ("prompt_eval_count", "eval_count"):
            if isinstance(body.get(key), int):
                usage[key] = body[key]
        for key in ("total_duration", "prompt_eval_duration",
                    "eval_duration"):
            if isinstance(body.get(key), (int, float)):
                usage[key + "_ns"] = body[key]
        return usage
    except Exception:
        return {}


def _extract_params(backend, request):
    """Generation knobs echoed from the request payload, else {}."""
    try:
        params = {}
        if backend == "openrouter":
            for key in ("temperature", "top_p", "max_tokens"):
                if request.get(key) is not None:
                    params[key] = request[key]
        else:
            options = (request or {}).get("options") or {}
            for key in ("temperature", "top_p", "num_ctx", "num_predict"):
                if options.get(key) is not None:
                    params[key] = options[key]
        return params
    except Exception:
        return {}


def normalize_record(backend, model, request, response, latency_ms,
                     status="ok", fallback=False, attempts=1, reply_text=""):
    """Flatten one backend call into a JSON-safe debug dict."""
    request = request if isinstance(request, dict) else {}
    response = response if isinstance(response, dict) else {}
    try:
        messages = request.get("messages")
        if not isinstance(messages, list):
            messages = []
    except Exception:
        messages = []
    return {
        "backend": str(backend),
        "model": str(model),
        "status": str(status),
        "fallback": bool(fallback),
        "attempts": attempts if isinstance(attempts, int) else 1,
        "latency_ms": latency_ms if isinstance(latency_ms, int) else -1,
        "params": _extract_params(backend, request),
        "messages": [
            {"role": m.get("role", "?"),
             "content": _as_text(m.get("content", ""))}
            for m in messages if isinstance(m, dict)
        ],
        "reasoning": _extract_reasoning(backend, response),
        "usage": _extract_usage(backend, response),
        "reply": reply_text if isinstance(reply_text, str) else "",
    }


def format_debug_record(record, max_chars=200000, bot_name="Hisami"):
    """Render a normalized record as readable markdown. Never raises."""
    try:
        rec = record if isinstance(record, dict) else {}
        backend = rec.get("backend", "?")
        model = rec.get("model", "?")
        lines = [
            f"# {bot_name} debug — {backend} / {model}",
            "",
            f"status: {rec.get('status', '?')} | "
            f"fallback: {rec.get('fallback', False)} | "
            f"attempts: {rec.get('attempts', '?')} | "
            f"latency: {rec.get('latency_ms', '?')} ms",
        ]
        params = rec.get("params") or {}
        if params:
            lines.append("params: " + ", ".join(
                f"{k}={v}" for k, v in params.items()))
        head = "\n".join(lines) + "\n"

        convo_lines = ["", "## Prompt messages"]
        messages = rec.get("messages") or []
        convo_lines.append(f"({len(messages)} messages)")
        for m in messages:
            role = m.get("role", "?") if isinstance(m, dict) else "?"
            content = (m.get("content", "") if isinstance(m, dict) else "")
            convo_lines.append(f"\n### [{role}]")
            convo_lines.append(str(content))
        convo = "\n".join(convo_lines) + "\n"

        reasoning = rec.get("reasoning") or ""
        reason = ""
        if isinstance(reasoning, str) and reasoning:
            if len(reasoning) > REASONING_MAX_CHARS:
                reasoning = (reasoning[:REASONING_MAX_CHARS] +
                             f"\n... [reasoning truncated: "
                             f"{len(rec['reasoning'])} chars total]")
            reason = "\n## Reasoning\n\n" + reasoning + "\n"

        usage = rec.get("usage") or {}
        tail_lines = ["", "## Reply (sent text)", ""]
        tail_lines.append(str(rec.get("reply") or "(none)"))
        tail_lines += ["", "## Usage"]
        if usage:
            tail_lines.extend(f"{k}: {v}" for k, v in usage.items())
        else:
            tail_lines.append("(no usage counters in response)")
        tail = "\n".join(tail_lines) + "\n"

        try:
            budget = int(max_chars)
        except (TypeError, ValueError):
            budget = 200000
        fixed = len(head) + len(reason) + len(tail)
        if len(convo) > budget - fixed:
            keep = max(0, budget - fixed - 200)
            convo = (convo[:keep] +
                     f"\n... [prompt truncated: "
                     f"{len(convo)} chars total, showing {keep}]")
        return head + convo + reason + tail
    except Exception:
        return f"# {bot_name} debug\n\n(record failed to render)\n"
