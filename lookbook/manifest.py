"""Helpers for reading and writing the manifest.

The manifest *is* a `MutableMapping[(image_id, metric_id), Annotation]`.
This module is a tiny ergonomics layer on top — nothing here is essential
to the abstraction; it exists so callers don't have to spell out tuple keys.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Optional

from lookbook.base import Annotation, ImageId, Manifest, MetricId


def get_annotation(
    manifest: Manifest, image_id: ImageId, metric_id: MetricId
) -> Optional[Annotation]:
    """Return the annotation if present, else None."""
    return manifest.get((image_id, metric_id))


def has_annotation(
    manifest: Manifest, image_id: ImageId, metric_id: MetricId
) -> bool:
    return (image_id, metric_id) in manifest


def put_annotation(manifest: Manifest, annotation: Annotation) -> None:
    """Write an annotation to the manifest, keyed by (image_id, metric_id)."""
    manifest[annotation.key] = annotation


def iter_annotations_for(
    manifest: Manifest, image_id: ImageId
) -> Iterator[Annotation]:
    """Yield every annotation for a given image."""
    for key in list(manifest):
        if key[0] == image_id:
            yield manifest[key]


def image_ids(manifest: Manifest) -> Iterable[ImageId]:
    """The set of distinct image_ids that appear in the manifest."""
    seen: set[ImageId] = set()
    for key in manifest:
        if key[0] not in seen:
            seen.add(key[0])
            yield key[0]


def value_of(
    manifest: Manifest,
    image_id: ImageId,
    metric_id: MetricId,
    default: Any = None,
) -> Any:
    """Return the bare value of an annotation (not the wrapper), or default."""
    a = get_annotation(manifest, image_id, metric_id)
    return default if a is None else a.value
