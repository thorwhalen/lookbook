"""Turn a source spec into a stream of `ImageRef`s.

Phase 0 supports: a single directory path (recursive), a single file, or an
already-iterable of refs. URL lists, zip archives, and cloud buckets are
deferred to Phase 1+.

`ingest_to_store` (Phase 4) additionally records `image_id -> {"path": ...}`
into `stores.images` so the HTTP layer can serve image bytes by id.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import Optional, Union

from lookbook.base import ImageRef
from lookbook.refs import PathImageRef

ImageExtensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif")

Source = Union[str, os.PathLike, Iterable[ImageRef]]


def _walk_paths(root: str) -> Iterator[str]:
    if os.path.isfile(root):
        yield root
        return
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(ImageExtensions):
                yield os.path.join(dirpath, fn)


def ingest(source: Source) -> list[ImageRef]:
    """Materialize a source into a list of ImageRefs.

    >>> import tempfile, os
    >>> with tempfile.TemporaryDirectory() as d:
    ...     for name in ['a.jpg', 'b.png', 'note.txt']:
    ...         open(os.path.join(d, name), 'wb').close()
    ...     refs = ingest(d)
    >>> sorted(os.path.basename(r.path) for r in refs)
    ['a.jpg', 'b.png']
    """
    if isinstance(source, (str, os.PathLike)):
        return [PathImageRef(path=p) for p in _walk_paths(os.fspath(source))]
    return list(source)


def ingest_to_store(source: Source, stores) -> list[ImageRef]:
    """Ingest plus a side-effect: record `image_id -> {"path": ...}` for
    every `PathImageRef` into `stores.images`.

    The HTTP layer relies on this mapping to serve image bytes by id.
    Returns the same list of refs `ingest()` would return.
    """
    refs = ingest(source)
    images_store = stores.images
    for r in refs:
        if isinstance(r, PathImageRef):
            try:
                images_store[r.image_id] = {"path": os.path.abspath(r.path)}
            except Exception:
                # Some store backends require JSON values; if the codec
                # disagrees, just skip. The HTTP layer can still serve
                # whatever is recorded.
                pass
    return refs
