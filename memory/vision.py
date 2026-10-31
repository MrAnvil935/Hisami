"""Image helpers for the vision chain: detection, classification, downscale.

Pure functions (Pillow + stdlib only, no Discord/network) so they are
unit-testable. Downloading + model calls live in bot.py.
"""

import hashlib
import io
import re
from urllib.parse import unquote, urlsplit

from PIL import Image

MAX_SIDE = 512

_IMAGE_EXTS = ("png", "jpg", "jpeg", "gif", "webp")
_VIDEO_EXTS = ("mp4", "mov", "webm", "avi", "mkv", "m4v")

# CDNs that serve an image even when the path has no extension.
_IMAGE_HOSTS = ("media.tenor.com", "c.tenor.com", "media.giphy.com",
                "i.giphy.com")

_URL_RE = re.compile(r"https?://[^\s<>\"]+")
_TRAILING_PUNCT = ".,;:!?)'\""


def _looks_like_image(url):
    """True if a URL most likely serves an image.

    Covers the extension form (path ends in an image ext), the query form
    (e.g. pbs.twimg.com/...?format=jpg), and known GIF CDNs. Pure.
    """
    low = str(url or "").lower()
    if not low:
        return False
    path = low.split("?", 1)[0]
    if path.endswith(_IMAGE_EXTS):
        return True
    m = re.search(r"[?&]format=([a-z0-9]+)", low)
    if m and m.group(1) in _IMAGE_EXTS:
        return True
    try:
        return urlsplit(low).netloc in _IMAGE_HOSTS
    except Exception:
        return False


def extract_image_urls(text):
    """Find direct image URLs in message text (first match matters most)."""
    if not text:
        return []
    out = []
    for raw in _URL_RE.findall(str(text)):
        url = raw.rstrip(_TRAILING_PUNCT)
        if _looks_like_image(url):
            out.append(url)
    return out


def _kind_from_name(filename):
    name = str(filename or "").lower()
    path = name.split("?", 1)[0]
    if path.endswith(_IMAGE_EXTS):
        return "image"
    if path.endswith(_VIDEO_EXTS):
        return "video"
    m = re.search(r"[?&]format=([a-z0-9]+)", name)
    if m:
        if m.group(1) in _IMAGE_EXTS:
            return "image"
        if m.group(1) in _VIDEO_EXTS:
            return "video"
    return ""


def _name_from_url(url, fallback="image"):
    """Basename of a URL path for display, or `fallback` when empty."""
    try:
        base = unquote(urlsplit(str(url)).path.rsplit("/", 1)[-1])
        return base or fallback
    except Exception:
        return fallback


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


def classify_embeds(embeds):
    """Discord Embed list -> [{kind, name, url}] (images/videos only).

    Handles media Discord renders inside an embed (linked GIFs, images,
    gifv). Per embed: always take `image.url`; take `thumbnail.url` only
    for real-media embeds (image/gifv/video — a gifv thumbnail is a real
    frame), never for rich/link/article previews (usually a favicon); take
    `video.url` as video. Extension/`format=` decides the kind, with the
    embed type as fallback. Duplicate URLs are dropped. Pure (no I/O).
    """
    out, seen = [], set()
    for i, emb in enumerate(embeds or (), 1):
        etype = str(getattr(emb, "type", "") or "").lower()
        image = getattr(emb, "image", None)
        thumb = getattr(emb, "thumbnail", None)
        video = getattr(emb, "video", None)
        cands = []
        iurl = getattr(image, "url", "") if image is not None else ""
        if iurl:
            cands.append(("image", iurl))
        if etype in ("image", "gifv", "video"):
            turl = getattr(thumb, "url", "") if thumb is not None else ""
            if turl:
                cands.append(("image", turl))
        vurl = getattr(video, "url", "") if video is not None else ""
        if vurl:
            cands.append(("video", vurl))
        for hint, url in cands:
            url = str(url or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            kind = _kind_from_name(url) or hint
            out.append({"kind": kind, "name": _name_from_url(url,
                                                             f"embed-{i}"),
                        "url": url})
    return out


def collect_media(attachments, embeds):
    """Merge attachment + embed media into one de-duplicated list.

    Order is attachments first, then embeds (each source's own order), so
    the first image found is deterministic. Pure (no I/O).
    """
    out, seen = [], set()
    for item in list(classify_attachments(attachments)) + \
            list(classify_embeds(embeds)):
        url = item.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(item)
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
