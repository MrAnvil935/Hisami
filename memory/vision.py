"""Image helpers for the vision chain: detection, classification, downscale.

Pure functions (Pillow + stdlib only, no Discord/network) so they are
unit-testable. Downloading + model calls live in bot.py.
"""

import hashlib
import io
import re

from PIL import Image

MAX_SIDE = 512

_IMAGE_EXTS = ("png", "jpg", "jpeg", "gif", "webp")
_VIDEO_EXTS = ("mp4", "mov", "webm", "avi", "mkv", "m4v")

_URL_RE = re.compile(r"https?://[^\s<>\"]+")
_TRAILING_PUNCT = ".,;:!?)'\""


def extract_image_urls(text):
    """Find direct image URLs in message text (first match matters most)."""
    if not text:
        return []
    out = []
    for raw in _URL_RE.findall(str(text)):
        url = raw.rstrip(_TRAILING_PUNCT)
        path = url.split("?", 1)[0].lower()
        if path.endswith(_IMAGE_EXTS):
            out.append(url)
    return out


def _kind_from_name(filename):
    name = str(filename or "").lower()
    if name.endswith(_IMAGE_EXTS):
        return "image"
    if name.endswith(_VIDEO_EXTS):
        return "video"
    return ""


def classify_attachments(attachments):
    """Discord Attachment list -> [{kind, name, url}] (images/videos only).

    Prefers content_type, falls back to filename extension. Entries that
    are neither image nor video are dropped.
    """
    out = []
    for att in attachments or []:
        ctype = str(getattr(att, "content_type", "") or "")
        name = str(getattr(att, "filename", "") or "file")
        url = str(getattr(att, "url", "") or "")
        if not url:
            continue
        if ctype.startswith("image/"):
            kind = "image"
        elif ctype.startswith("video/"):
            kind = "video"
        else:
            kind = _kind_from_name(name)
        if kind:
            out.append({"kind": kind, "name": name, "url": url})
    return out


def downscale(image_bytes, max_side=MAX_SIDE):
    """Downscale so the longest side is max_side; return JPEG bytes.

    RGBA/palette sources are flattened onto white for JPEG compatibility.
    Raises on undecodable input (callers treat as fail-soft).
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def cache_key(source_url):
    """Stable cache key for an image source URL."""
    return hashlib.sha256(str(source_url).encode("utf-8")).hexdigest()


def format_markers(attachments):
    """Render stored attachment metadata as prompt markers.

    Input: list of {kind, name} (as stored in messages.attachments).
    Output e.g. ' [image: foo.png] [video: bar.mp4]' ('' when none).
    """
    parts = []
    for att in attachments or []:
        kind = att.get("kind")
        name = att.get("name", "file")
        if kind in ("image", "video"):
            parts.append(f"[{kind}: {name}]")
    return (" " + " ".join(parts)) if parts else ""
