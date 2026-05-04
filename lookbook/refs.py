"""Concrete `ImageRef` implementations.

Refs are the unit of identity in lookbook. The orchestrator passes refs
around; scorers open them lazily; the manifest is keyed by `image_id`.

Three impls are provided here. None of them require Pillow at import time;
`open()` defers the import.

All three expose ``local_path(cache_dir=None) -> str`` — a path on the
local filesystem that the bytes are reachable through. Path refs return
their existing path; bytes/url refs materialize once into a content-
addressed cache and reuse the result. This is the ergonomic surface for
downstream tools that want a file (e.g. ffmpeg, lookbook → fal upload).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional


def _hash_id(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()[:16]


def _materialized_cache_dir(cache_dir: Optional[str] = None) -> str:
    """Where to land bytes that need a path. Honors ``$LOOKBOOK_REFS_CACHE_DIR``
    then a temp subdirectory."""
    if cache_dir:
        out = cache_dir
    else:
        out = os.environ.get(
            "LOOKBOOK_REFS_CACHE_DIR",
            os.path.join(tempfile.gettempdir(), "lookbook_refs"),
        )
    os.makedirs(out, exist_ok=True)
    return out


def _ext_from_payload(payload: bytes) -> str:
    """Sniff a likely extension from the first few bytes. Defaults to .bin."""
    head = payload[:16]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if head[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if head[:4] == b"GIF8":
        return ".gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return ".webp"
    if head[:4] == b"BM\x00\x00" or head[:2] == b"BM":
        return ".bmp"
    return ".bin"


def to_local_path(ref, *, cache_dir: Optional[str] = None) -> str:
    """Return a filesystem path where the bytes of ``ref`` can be read.

    Works on any object that exposes ``.local_path(cache_dir)`` — i.e.
    every concrete ``ImageRef`` shipped with lookbook. Provided as a
    free function so downstream code doesn't need to know which subtype
    it has.
    """
    return ref.local_path(cache_dir=cache_dir)


@dataclass
class PathImageRef:
    """An image referenced by filesystem path."""

    path: str
    image_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.image_id:
            # Cheap, stable, content-independent id; full content hash is a
            # separate (more expensive) scorer.
            self.image_id = _hash_id(os.path.abspath(self.path).encode("utf-8"))

    def open(self):
        from PIL import Image  # lazy

        return Image.open(self.path)

    def bytes(self) -> bytes:
        with open(self.path, "rb") as f:
            return f.read()

    def local_path(self, *, cache_dir: Optional[str] = None) -> str:
        """Return the existing on-disk path. ``cache_dir`` is ignored."""
        return self.path


@dataclass
class BytesImageRef:
    """An image stored as raw bytes in memory."""

    payload: bytes
    image_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.image_id:
            self.image_id = _hash_id(self.payload)

    def open(self):
        import io
        from PIL import Image  # lazy

        return Image.open(io.BytesIO(self.payload))

    def bytes(self) -> bytes:
        return self.payload

    def local_path(self, *, cache_dir: Optional[str] = None) -> str:
        """Materialize the bytes into a file (cached by image_id)."""
        d = _materialized_cache_dir(cache_dir)
        ext = _ext_from_payload(self.payload)
        path = os.path.join(d, f"{self.image_id}{ext}")
        if not os.path.exists(path):
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(self.payload)
            os.replace(tmp, path)
        return path


@dataclass
class UrlImageRef:
    """An image referenced by URL.

    The bytes are fetched on first access and cached on the instance.
    """

    url: str
    image_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _cached: Optional[bytes] = field(default=None, repr=False)

    def __post_init__(self):
        if not self.image_id:
            self.image_id = _hash_id(self.url.encode("utf-8"))

    def bytes(self) -> bytes:
        if self._cached is None:
            from urllib.request import urlopen  # lazy

            with urlopen(self.url) as r:
                self._cached = r.read()
        return self._cached

    def open(self):
        import io
        from PIL import Image  # lazy

        return Image.open(io.BytesIO(self.bytes()))

    def local_path(self, *, cache_dir: Optional[str] = None) -> str:
        """Download (once) and return the local path."""
        d = _materialized_cache_dir(cache_dir)
        payload = self.bytes()
        ext = _ext_from_payload(payload)
        path = os.path.join(d, f"{self.image_id}{ext}")
        if not os.path.exists(path):
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(payload)
            os.replace(tmp, path)
        return path
