"""Core protocols and types for lookbook.

This module is the open-closed boundary of the package. Every extension point
is a `typing.Protocol` here. Heavy ML dependencies must NOT be imported in
this module — keep it pure-Python so the laptop tier of the package stays
import-light.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, Tuple, runtime_checkable


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


ImageId = str
MetricId = str
ManifestKey = Tuple[ImageId, MetricId]


@dataclass(frozen=True)
class Annotation:
    """A single annotation produced by a Scorer or Embedder for one image.

    The (image_id, metric_id) pair is the manifest key. `config_hash` is the
    cache key — re-running with the same config yields a cache hit; changing
    a threshold or model invalidates.
    """

    image_id: ImageId
    metric_id: MetricId
    value: Any
    config_hash: str = ""
    cost_tier: int = 0
    timestamp: datetime = field(default_factory=_utcnow)
    backend: str = ""

    @property
    def key(self) -> ManifestKey:
        return (self.image_id, self.metric_id)


Manifest = MutableMapping[ManifestKey, Annotation]


@runtime_checkable
class ImageRef(Protocol):
    """Anything that knows its identity and how to be opened lazily."""

    image_id: ImageId
    metadata: Mapping[str, Any]

    def open(self) -> Any:  # returns PIL.Image.Image when Pillow is loaded
        ...

    def bytes(self) -> bytes: ...


@runtime_checkable
class Scorer(Protocol):
    """Computes one annotation per image. Idempotent w.r.t. config_hash."""

    metric_id: MetricId
    cost_tier: int
    requires: Tuple[MetricId, ...]
    config_hash: str

    def score(self, ref: ImageRef, manifest: Manifest) -> Any: ...


@runtime_checkable
class Filter(Protocol):
    """Predicate over annotations that decides whether to keep an image."""

    def keep(self, ref: ImageRef, manifest: Manifest) -> bool: ...


@runtime_checkable
class Embedder(Protocol):
    """Computes a vector representation in some named embedding space."""

    space_id: str
    cost_tier: int

    def embed(self, ref: ImageRef) -> Any: ...  # np.ndarray


@runtime_checkable
class Selector(Protocol):
    """Chooses a subset of size up-to-K from candidates given the manifest."""

    def select(
        self,
        candidates: Iterable[ImageRef],
        manifest: Manifest,
        k: int,
        constraints: Mapping[str, Any] = None,  # type: ignore[assignment]
    ) -> list[ImageRef]: ...
