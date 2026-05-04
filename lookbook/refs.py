"""Concrete `ImageRef` implementations.

Refs are the unit of identity in lookbook. The orchestrator passes refs
around; scorers open them lazily; the manifest is keyed by `image_id`.

Three impls are provided here. None of them require Pillow at import time;
`open()` defers the import.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional


def _hash_id(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()[:16]


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
