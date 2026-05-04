"""Quota-aware selection: pick K from N with per-bin budgets.

The QuotaSelector wraps an inner selector and runs it once per bin. Bins
are derived from a manifest annotation (`bin_metric_id`); quotas are a
mapping from bin label to a count.

Typical use for character LoRAs:

    QuotaSelector(
        bin_metric_id="head_pose.yaw_bin",   # or any nested-key path
        inner_selector_id="top_k",
        inner_overrides={"metric_id": "face_quality"},
    )

    # via curate(...)
    constraints = {"quotas": {"front": 8, "three_quarter": 5, "profile": 3, "back": 1}}

If a bin doesn't reach its quota (not enough survivors there), the deficit
falls back to the highest-quality survivors from any bin so the kept-set
size still hits K when possible. Set `strict=True` to disable that fallback.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from lookbook.base import ImageRef, Manifest
from lookbook.manifest import value_of
from lookbook.registry import selectors


def _resolve_path(value: Any, path: str) -> Any:
    """Traverse a dotted path into a nested dict-like value."""
    if value is None:
        return None
    parts = path.split(".")
    cur = value
    for p in parts:
        if not isinstance(cur, Mapping) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _bin_for(manifest: Manifest, image_id: str, bin_metric_id: str) -> Any:
    """Read the bin label for an image.

    `bin_metric_id` may be a plain metric_id (`"size_bin"`) or a dotted path
    into a nested-dict annotation (`"head_pose.yaw_bin"`).
    """
    if "." in bin_metric_id:
        head, _, tail = bin_metric_id.partition(".")
        ann_value = value_of(manifest, image_id, head)
        return _resolve_path(ann_value, tail)
    return value_of(manifest, image_id, bin_metric_id)


@dataclass
class QuotaSelector:
    """Per-bin quota selection wrapping an inner Selector."""

    selector_id: str = "quota"
    bin_metric_id: str = "head_pose.yaw_bin"
    inner_selector_id: str = "top_k"
    inner_overrides: dict = field(default_factory=dict)
    strict: bool = False
    embedding_space: Optional[str] = None  # forwarded if the inner needs it

    def select(
        self,
        candidates: Iterable[ImageRef],
        manifest: Manifest,
        k: int,
        constraints: Mapping[str, Any] = None,  # type: ignore[assignment]
    ) -> list[ImageRef]:
        candidates = list(candidates)
        if k <= 0 or not candidates:
            return []
        constraints = dict(constraints or {})
        quotas = constraints.get("quotas") or {}

        # Bin candidates.
        by_bin: dict[Any, list[ImageRef]] = {}
        for r in candidates:
            b = _bin_for(manifest, r.image_id, self.bin_metric_id)
            by_bin.setdefault(b, []).append(r)

        # Resolve the inner selector fresh so any internal state is clean.
        inner_inst = selectors.get(self.inner_selector_id)
        inner_cls = type(inner_inst)
        inner = (
            inner_cls(**self.inner_overrides) if self.inner_overrides else inner_cls()
        )

        kept: list[ImageRef] = []
        kept_ids: set = set()

        # Iterate quotas in declaration order (Python 3.7+ dicts preserve order).
        for bin_label, quota in quotas.items():
            if quota <= 0:
                continue
            pool = by_bin.get(bin_label, [])
            if not pool:
                continue
            chosen = inner.select(pool, manifest, k=quota, constraints=constraints)
            for r in chosen:
                if r.image_id not in kept_ids:
                    kept.append(r)
                    kept_ids.add(r.image_id)

        # Backfill if quotas under-filled K (and not strict).
        if not self.strict and len(kept) < k:
            remaining = [r for r in candidates if r.image_id not in kept_ids]
            backfill = inner.select(
                remaining, manifest, k=(k - len(kept)), constraints=constraints
            )
            for r in backfill:
                if r.image_id not in kept_ids:
                    kept.append(r)
                    kept_ids.add(r.image_id)

        return kept[:k]


selectors.register("quota", QuotaSelector())
