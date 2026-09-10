"""Full-bodied LLM call dump: one pretty-printed record per call.

Pure I/O helper (no Discord, no network) so it is unit-testable.
bot.py cannot be imported by tests (it calls client.run at module
level), hence this module.

NOTE: records are pretty-printed (indent=2) with a blank line between
them for direct reading, so this file is NOT strict one-object-per-line
JSONL despite the .jsonl extension (kept to avoid config churn).
Parse multi-record files with read_records() below, not json.loads()
per line.

Only request/response *bodies* are stored — never HTTP headers, so
the OpenRouter key cannot leak into the log files.
"""

import json
import logging
import os
import time

log = logging.getLogger("hisami.llm")

_DIR = "logs"
_ENABLED = True
_MAX_BYTES = 5242880
_BACKUPS = 3


def configure(directory="logs", enabled=True, max_bytes=5242880, backups=3):
    global _DIR, _ENABLED, _MAX_BYTES, _BACKUPS
    _DIR = directory
    _ENABLED = bool(enabled)
    _MAX_BYTES = int(max_bytes)
    _BACKUPS = int(backups)


def dump_file():
    return os.path.join(_DIR, "llm.jsonl")


def read_records(path=None):
    """Parse all records from a (possibly multi-block) dump file."""
    with open(path or dump_file(), encoding="utf-8") as f:
        text = f.read()
    decoder = json.JSONDecoder()
    records, idx = [], 0
    while idx < len(text.strip()):
        record, end = decoder.raw_decode(text, idx)
        records.append(record)
        idx = end
        while idx < len(text) and text[idx].isspace():
            idx += 1
    return records


def _rotate(path):
    """Roll dump file over when it exceeds the byte cap. Fail-soft."""
    try:
        if not os.path.exists(path):
            return
        if os.path.getsize(path) < _MAX_BYTES:
            return
        if _BACKUPS < 1:
            os.remove(path)
            return
        oldest = f"{path}.{_BACKUPS}"
        if os.path.exists(oldest):
            os.remove(oldest)
        for i in range(_BACKUPS - 1, 0, -1):
            src = f"{path}.{i}"
            if os.path.exists(src):
                os.rename(src, f"{path}.{i + 1}")
        os.rename(path, f"{path}.1")
    except Exception as e:
        log.warning("llm dump rotation failed: %s", e)


def log_call(backend, model, purpose, request, response,
             latency_ms, status):
    """Append one record. Fail-soft: never raises. Returns path or None."""
    if not _ENABLED:
        return None
    try:
        record = {
            "ts": time.time(),
            "backend": backend,
            "model": model,
            "purpose": purpose,
            "request": request,
            "response": response,
            "latency_ms": latency_ms,
            "status": status,
        }
        path = dump_file()
        _rotate(path)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, indent=2, ensure_ascii=False) + "\n\n")
        return path
    except Exception as e:
        log.warning("llm dump failed: %s", e)
        return None
