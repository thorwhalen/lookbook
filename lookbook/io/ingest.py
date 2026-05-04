"""Turn a source spec into a stream of `ImageRef`s.

Phase 0 supports: a single directory path (recursive), a single file, or an
already-iterable of refs. URL lists, zip archives, and cloud buckets are
deferred to Phase 1+.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import Union

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
