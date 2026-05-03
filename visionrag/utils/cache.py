"""Vision output caching keyed by page image content hash.

Cache versioning
----------------
The cache key mixes a `_VISION_CACHE_VERSION` token into the image hash so
that when we change the vision prompt's OUTPUT SCHEMA (not just the wording),
every existing cached entry is invalidated and pages get re-extracted on the
next ingest. Bump the version whenever the shape of the returned JSON
changes — do NOT bump for incidental prompt tweaks, as that wastes money
re-running vision on unchanged content.

Bump history:
  v1 — initial schema: formulas = {latex, description}
  v2 — enriched schema: formulas = {name, latex, variables[], context};
       derivations carry {goal, assumptions, result}
  v3 — added prose_text (authoritative page transcription from the image),
       section_identifiers (dotted numbers visible on page), key_terms
       (exam-relevant literal terms)
"""

import hashlib
import json
from pathlib import Path

from visionrag.config import get_settings

# Bump this when the vision prompt's JSON schema changes. See module docstring.
_VISION_CACHE_VERSION = "v3"


def _cache_dir() -> Path:
    return Path(get_settings().vision_cache_dir)


def _page_hash(image_path: str) -> str:
    """SHA256 of (version-token || image-bytes).

    Version-token is prepended so a schema bump silently invalidates the
    whole cache without needing anyone to rm -rf the cache dir.
    """
    h = hashlib.sha256()
    h.update(_VISION_CACHE_VERSION.encode("utf-8"))
    h.update(b"\0")
    with open(image_path, "rb") as f:
        while True:
            chunk = f.read(1 << 16)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


async def get_cached_vision(image_path: str) -> dict | None:
    h = _page_hash(image_path)
    cache_file = _cache_dir() / f"{h}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    return None


async def set_cached_vision(image_path: str, data: dict) -> None:
    d = _cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    h = _page_hash(image_path)
    (d / f"{h}.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
