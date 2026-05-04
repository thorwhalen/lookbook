"""Phase 1 filters — cheap funnel.

Filters consume the manifest and return True/False per image. Two of them
(`NoExactDuplicate`, `NoNearDuplicate`) carry per-pipeline state and must
be re-instantiated for each new run; the rest are stateless.

Each filter exposes a `name` attribute (the filter id) so the run report
can attribute drops back to the rule that fired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from lookbook.base import ImageRef, Manifest
from lookbook.manifest import value_of
from lookbook.registry import filters
from lookbook.scorers.technical import phash_distance


# ---------------------------------------------------------------------------
# Stateless filters
# ---------------------------------------------------------------------------


@dataclass
class MinResolution:
    """Drop images whose long side is below `min_long_side` pixels.

    Reads the `resolution` annotation. If absent, the filter keeps the image
    (under the principle that filters never drop on missing evidence).
    """

    min_long_side: int = 1024
    name: str = "min_resolution"

    def keep(self, ref: ImageRef, manifest: Manifest) -> bool:
        res = value_of(manifest, ref.image_id, "resolution")
        if not res:
            return True
        return int(res.get("long_side", 0)) >= self.min_long_side


filters.register("min_resolution", MinResolution())


@dataclass
class MinBlur:
    """Drop images whose Laplacian variance is below `threshold` (blurry)."""

    threshold: float = 100.0
    name: str = "min_blur"

    def keep(self, ref: ImageRef, manifest: Manifest) -> bool:
        v = value_of(manifest, ref.image_id, "blur")
        if v is None:
            return True
        return float(v) >= self.threshold


filters.register("min_blur", MinBlur())


@dataclass
class ExposureRange:
    """Drop images that are mostly black or mostly white.

    Defaults: drop if more than 50% of pixels are below 16 or above 240.
    """

    max_underexposed_frac: float = 0.5
    max_overexposed_frac: float = 0.5
    name: str = "exposure_range"

    def keep(self, ref: ImageRef, manifest: Manifest) -> bool:
        e = value_of(manifest, ref.image_id, "exposure")
        if not e:
            return True
        return (
            e.get("frac_underexposed", 0) <= self.max_underexposed_frac
            and e.get("frac_overexposed", 0) <= self.max_overexposed_frac
        )


filters.register("exposure_range", ExposureRange())


# ---------------------------------------------------------------------------
# Stateful set-aware filters: dedup
# ---------------------------------------------------------------------------


@dataclass
class NoExactDuplicate:
    """Keep at most one image per `file_hash`.

    The first image in iteration order to claim a hash wins; later images
    with the same hash are dropped.
    """

    name: str = "no_exact_duplicate"
    _seen: set = field(default_factory=set, repr=False)

    def keep(self, ref: ImageRef, manifest: Manifest) -> bool:
        h = value_of(manifest, ref.image_id, "file_hash")
        if not h:
            return True
        if h in self._seen:
            return False
        self._seen.add(h)
        return True


filters.register("no_exact_duplicate", NoExactDuplicate())


@dataclass
class NoNearDuplicate:
    """Keep at most one image per near-duplicate cluster.

    Clusters are formed greedily: the first image to be visited is the
    cluster representative, and any later image whose `phash` is within
    `max_distance` Hamming bits is dropped.
    """

    max_distance: int = 5
    name: str = "no_near_duplicate"
    _seen_hashes: list = field(default_factory=list, repr=False)

    def keep(self, ref: ImageRef, manifest: Manifest) -> bool:
        h = value_of(manifest, ref.image_id, "phash")
        if not h:
            return True
        for prior in self._seen_hashes:
            if phash_distance(h, prior) <= self.max_distance:
                return False
        self._seen_hashes.append(h)
        return True


filters.register("no_near_duplicate", NoNearDuplicate())


def fresh_filter(filter_id: str, **overrides):
    """Return a fresh instance of a registered filter.

    Stateful filters (dedup) must be re-instantiated per pipeline; the
    registry holds a singleton, so call this when building a fresh pipeline.
    """
    inst = filters.get(filter_id)
    cls = type(inst)
    if overrides:
        return cls(**overrides)
    return cls()
