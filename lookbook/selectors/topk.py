"""Top-K selector: pick the K highest-scoring images by a single metric."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from lookbook.base import ImageRef, Manifest
from lookbook.manifest import value_of
from lookbook.registry import selectors


@dataclass
class TopK:
    """Sort candidates by `metric_id` (descending) and take the first K."""

    metric_id: str = "random_score"
    selector_id: str = "top_k"
    descending: bool = True

    def select(
        self,
        candidates: Iterable[ImageRef],
        manifest: Manifest,
        k: int,
        constraints: Mapping[str, Any] = None,  # type: ignore[assignment]
    ) -> list[ImageRef]:
        cands = list(candidates)
        ranked = sorted(
            cands,
            key=lambda r: (
                value_of(manifest, r.image_id, self.metric_id, default=float("-inf"))
            ),
            reverse=self.descending,
        )
        return ranked[: max(0, k)]


selectors.register("top_k", TopK())
