"""Repository pattern for lookbook persistence.

Three stores hold the package's state. All three are MutableMappings so the
backend (filesystem, SQLite, S3, Mongo, in-memory) is swappable:

- `images`: image_id -> bytes / path / url payload (depends on ingest)
- `manifest`: "image_id::metric_id" -> Annotation (as dict via codec)
- `runs`: run_id -> run record (recipe, kept set, report)

The default factory wires everything to JSON-on-disk under the user's app
data folder (via config2py). Tests pass `dict()` everywhere to skip I/O.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional

from dol import Files, JsonFiles, wrap_kvs

from lookbook.base import Annotation, ManifestKey
from lookbook._paths import default_data_root, subdir

# ---------------------------------------------------------------------------
# Codecs and key transforms
# ---------------------------------------------------------------------------

# Separator must be filesystem-safe on every OS we care about. Windows
# disallows `: \ / * ? " < > |` in filenames, so we can't use `::`.
# `--` is safe on every OS, never appears in our image_ids (hex sha1
# prefixes) or metric_ids (snake_case), and reads as "joined".
_KEY_SEP = "--"


def _key_to_str(key: ManifestKey) -> str:
    image_id, metric_id = key
    return f"{image_id}{_KEY_SEP}{metric_id}.json"


def _str_to_key(s: str) -> ManifestKey:
    if s.endswith(".json"):
        s = s[: -len(".json")]
    image_id, _, metric_id = s.partition(_KEY_SEP)
    return (image_id, metric_id)


def _annotation_to_dict(a: Annotation) -> dict:
    d = asdict(a)
    d["timestamp"] = a.timestamp.isoformat()
    return d


def _dict_to_annotation(d: Mapping[str, Any]) -> Annotation:
    ts = d.get("timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    return Annotation(
        image_id=d["image_id"],
        metric_id=d["metric_id"],
        value=d["value"],
        config_hash=d.get("config_hash", ""),
        cost_tier=d.get("cost_tier", 0),
        timestamp=ts or datetime.now(timezone.utc),
        backend=d.get("backend", ""),
    )


def manifest_codec(store: MutableMapping) -> MutableMapping:
    """Wrap a string-keyed JSON-dict store so it speaks Annotations.

    Keys flow as `(image_id, metric_id) <-> "image_id::metric_id.json"`.
    Values flow as `Annotation <-> dict`.
    """
    return wrap_kvs(
        store,
        key_of_id=_str_to_key,
        id_of_key=_key_to_str,
        obj_of_data=_dict_to_annotation,
        data_of_obj=_annotation_to_dict,
    )


# ---------------------------------------------------------------------------
# Stores bundle and factory
# ---------------------------------------------------------------------------


@dataclass
class Stores:
    """Bundle of all stores lookbook uses for persistence."""

    images: MutableMapping
    manifest: MutableMapping
    runs: MutableMapping
    embeddings: MutableMapping  # space_id -> MutableMapping[image_id, vector]
    root: Optional[str] = None  # filesystem root, when applicable


def _make_embeddings_dict(root: Optional[str]) -> MutableMapping:
    """Top-level mapping of space_id -> per-space embedding store.

    The default is in-memory; persistent vector indexes are deferred to a
    later phase (Phase 2). When `root` is given we still keep the top-level
    in memory but each space-store gets its own subdirectory of JSON files
    holding the raw lists.
    """
    if root is None:
        return {}

    class _LazyEmbeddingSpaces(dict):
        def __missing__(self, space_id):
            store = JsonFiles(subdir(root, space_id))
            self[space_id] = store
            return store

    return _LazyEmbeddingSpaces()


def get_stores(
    *,
    root: Optional[str] = None,
    images_store: Optional[MutableMapping] = None,
    manifest_store: Optional[MutableMapping] = None,
    runs_store: Optional[MutableMapping] = None,
    embeddings: Optional[MutableMapping] = None,
) -> Stores:
    """Build the lookbook Stores bundle.

    All arguments are optional. When `root` is None and no stores are
    supplied, the user's app data folder is used. Pass `dict()` for any
    slot to keep that store in memory (useful for tests).

    >>> stores = get_stores(
    ...     images_store={}, manifest_store={}, runs_store={}, embeddings={}
    ... )
    >>> stores.manifest is not None
    True
    """
    if (
        images_store is None
        and manifest_store is None
        and runs_store is None
        and embeddings is None
        and root is None
    ):
        root = default_data_root()

    # `images` holds image-id -> metadata records (e.g. {"path": ...}).
    # The HTTP layer uses this to serve bytes by id. Users who want to
    # persist actual image bytes can pass a `Files`-backed store.
    images = (
        images_store
        if images_store is not None
        else (JsonFiles(subdir(root, "images")) if root else {})
    )

    if manifest_store is not None:
        manifest = manifest_store
    elif root:
        manifest = manifest_codec(JsonFiles(subdir(root, "manifest")))
    else:
        manifest = {}

    runs = (
        runs_store
        if runs_store is not None
        else (JsonFiles(subdir(root, "runs")) if root else {})
    )

    if embeddings is not None:
        emb = embeddings
    else:
        emb = _make_embeddings_dict(root)

    return Stores(
        images=images,
        manifest=manifest,
        runs=runs,
        embeddings=emb,
        root=root,
    )
