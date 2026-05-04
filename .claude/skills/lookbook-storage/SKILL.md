---
name: lookbook-storage
description: Use when working on lookbook persistence — swapping storage backends (filesystem, SQLite, S3, Mongo, Redis), debugging the manifest, designing a new store codec, or wiring stores for tests. Triggers on questions about lookbook.store, get_stores(), the repository pattern, manifest persistence, or "where does lookbook save its data".
---

# lookbook — storage and the repository pattern

`lookbook` follows the repository pattern. Every persistent thing the package
touches goes through a `MutableMapping` — and every concrete backend
(filesystem, SQLite, S3, Mongo) is just a different `MutableMapping`
implementation. This document is the operational guide for working with that
layer.

## The three (and a half) stores

`lookbook.store.Stores` bundles:

| Slot | Type | Keys | Values | Default backend |
|---|---|---|---|---|
| `images` | `MutableMapping` | `image_id` (str) | bytes / path / url payload | `dol.Files` |
| `manifest` | `MutableMapping` | `(image_id, metric_id)` tuple | `Annotation` | `dol.JsonFiles` + codec |
| `runs` | `MutableMapping` | `run_id` (str) | dict (JSON-able) | `dol.JsonFiles` |
| `embeddings` | `MutableMapping[space_id, MutableMapping]` | `space_id` → per-space store | per-space: `image_id` → vector | dict-of-`JsonFiles` |

The `embeddings` slot is a *mapping of mappings* — one per embedding space
(`clip_vit_l14`, `dinov2_vitb14`, `arcface_r100`, etc.) — because different
metrics use different spaces and they should be persisted independently.

## The default factory

```python
from lookbook import get_stores

# Default: persists to ~/.local/share/lookbook (Linux) or
# ~/Library/Application Support/lookbook (macOS).
stores = get_stores()

# Test mode: everything in memory, nothing touches disk.
stores = get_stores(
    images_store={}, manifest_store={}, runs_store={}, embeddings={},
)

# Custom root: same layout, different folder.
stores = get_stores(root="/path/to/some/dir")

# Mix-and-match: keep manifest in memory, runs on disk.
from dol import JsonFiles
stores = get_stores(
    manifest_store={},
    runs_store=JsonFiles("/path/runs/"),
)
```

The path defaults route through `lookbook._paths` (which routes through
`config2py`). Do not hardcode paths — call `default_data_root()` etc.

## The manifest codec

The manifest is a `MutableMapping[(image_id, metric_id), Annotation]`, but
`JsonFiles` only speaks `str -> dict`. The codec bridges them. From
`lookbook/store.py`:

```python
def manifest_codec(store: MutableMapping) -> MutableMapping:
    """Wrap a string-keyed JSON-dict store so it speaks Annotations."""
    return wrap_kvs(
        store,
        key_of_id=_str_to_key,        # filename -> tuple
        id_of_key=_key_to_str,        # tuple -> filename
        obj_of_data=_dict_to_annotation,
        data_of_obj=_annotation_to_dict,
    )
```

Filename format: `"{image_id}::{metric_id}.json"`. Pick this up from disk and
the codec re-hydrates `Annotation` instances on read, including parsing the
ISO-format timestamp.

## Swapping backends

The whole point of the pattern. Examples:

### SQLite for the manifest

```python
from sqldol import SqlTupleKVStore  # or whichever sqldol class fits
from lookbook.store import manifest_codec, get_stores

raw_manifest = SqlTupleKVStore("sqlite:///path/lookbook.db", "manifest")
manifest = manifest_codec(raw_manifest)
stores = get_stores(manifest_store=manifest)
```

### S3 for everything

```python
from s3dol import S3BinaryStore, S3JsonStore
from lookbook.store import manifest_codec, get_stores

stores = get_stores(
    images_store=S3BinaryStore("my-bucket", prefix="lookbook/images/"),
    manifest_store=manifest_codec(S3JsonStore("my-bucket", prefix="lookbook/manifest/")),
    runs_store=S3JsonStore("my-bucket", prefix="lookbook/runs/"),
    embeddings={},  # in-memory; embeddings are usually local-only
)
```

### Mongo for runs (queryable)

```python
from mongodol import MongoStore
stores = get_stores(runs_store=MongoStore(db="lookbook", collection="runs"))
```

The principle: any swap is a single arg to `get_stores()`. Nothing in
`pipeline.py`, `manifest.py`, or the scorers/selectors knows or cares.

## Working with the manifest in code

Always go through `lookbook.manifest` helpers when possible:

```python
from lookbook.manifest import (
    get_annotation, has_annotation, put_annotation,
    iter_annotations_for, image_ids, value_of,
)

# Read
v = value_of(manifest, image_id, "blur", default=0)

# Iterate
for ann in iter_annotations_for(manifest, image_id):
    print(ann.metric_id, ann.value)

# Write
put_annotation(manifest, Annotation(image_id=..., metric_id=..., value=...))
```

These shield you from changes to the codec, key shape, or persistence layer.

## Common pitfalls

1. **Forgetting `manifest_codec` when wrapping a custom string-keyed store.**
   The pipeline puts `Annotation` objects in; without the codec, JSON
   serialization will fail or you'll get raw dicts back on read.
2. **Listing a `dol` store keys with `for key in dict(store)`.** Some `dol`
   stores re-read the disk on every iteration. Pin to `list(store)` once
   when you need a stable snapshot.
3. **Embeddings as one big store.** Don't. Each embedding space wants its
   own substore; sizes and codecs differ. The default factory already does
   this lazily — keep that contract.
4. **Tests touching the user's app data folder.** Always pass `dict()` for
   every slot in tests, OR pass `tmp_path` as `root=`. Never let
   `get_stores()` default to the user folder in CI.
5. **Mutating annotations in place.** `Annotation` is `frozen=True` by
   design. Build a new one with the changed fields and re-`put_annotation`.

## Adding a new persistent thing

Suppose you want to persist run-level audit logs. The pattern:

1. Add a `MutableMapping` slot to `Stores` (extend the dataclass).
2. Extend `get_stores()` with a kwarg + default backend (`JsonFiles`).
3. Helpers go in their own module (e.g. `lookbook.audit`) — never in `store.py`.
4. Tests start with `dict()` for the new slot; FS tests use `tmp_path`.

Every new persistent slot is a small extension; the abstraction does the
heavy lifting.
