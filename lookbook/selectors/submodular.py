"""Submodular selectors over embedding spaces.

Phase 2 ships a greedy facility-location selector implemented in pure
numpy. It's the right default for "best K from N" when diversity matters:

  f(S) = sum over y in V of max over x in S of similarity(x, y)
       + alpha * sum over x in S of quality(x)

The greedy algorithm gives a (1 - 1/e) ≈ 0.63 approximation guarantee.
For typical curation sizes (N ≤ 1000, K ≤ 50), pure numpy is fast enough
and avoids the `apricot` dependency. For larger pools, swap in `apricot`'s
optimized lazy-greedy via the same Selector protocol.

Embeddings are not read from the manifest directly. The pipeline pre-fetches
them from `stores.embeddings[space_id]` and passes them via
`constraints["embeddings"]` as a `dict[image_id, np.ndarray]`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from lookbook.base import ImageRef, Manifest
from lookbook.manifest import value_of
from lookbook.registry import selectors


@dataclass
class FacilityLocation:
    """Greedy facility-location subset selection.

    The objective is:
        f(S) = sum_y max_{x in S} sim(x, y)  +  weight_quality * sum_{x in S} q(x)

    where `sim` is cosine similarity (assumes embeddings are L2-normalized;
    if they are not, the selector normalizes them first) and `q(x)` is the
    annotation `quality_metric_id` for `x`, defaulting to 0 when absent.

    `weight_quality=0` gives a pure-diversity selection. `weight_quality`
    in 0.1–0.5 typically works well for "diverse but high-quality."
    """

    selector_id: str = "facility_location"
    embedding_space: str = "dinov2_base"  # which embeddings to read
    quality_metric_id: str = ""  # optional per-image quality score
    weight_quality: float = 0.0
    weight_diversity: float = 1.0

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
        k = min(k, len(candidates))

        constraints = constraints or {}
        embeddings_map = constraints.get("embeddings") or {}
        missing = [r.image_id for r in candidates if r.image_id not in embeddings_map]
        if missing:
            raise ValueError(
                f"FacilityLocation: missing embeddings for {len(missing)} candidates "
                f"in space {self.embedding_space!r}. Run the embedder first. "
                f"First missing id: {missing[0]!r}"
            )

        X = np.stack([np.asarray(embeddings_map[r.image_id], dtype=np.float32)
                      for r in candidates])
        # Normalize defensively so dot product is cosine.
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

        # NxN similarity matrix, clipped to [0, 1] so anti-correlations
        # don't contribute to the facility-location objective and the empty
        # set has f({}) = 0.
        sim = np.clip(X @ X.T, 0.0, 1.0)

        # Per-image quality (zero by default).
        if self.quality_metric_id:
            q = np.array(
                [
                    float(value_of(manifest, r.image_id, self.quality_metric_id) or 0.0)
                    for r in candidates
                ],
                dtype=np.float32,
            )
        else:
            q = np.zeros(len(candidates), dtype=np.float32)

        # Greedy selection: at each step, pick the candidate maximizing the
        # marginal gain in the facility-location + quality objective.
        N = len(candidates)
        chosen: list[int] = []
        # `best_so_far[y]` holds max_{x in S} sim(x, y) for the current S.
        # Start at 0 so the first round's gain is `sum_y sim[i, y]` — i.e.
        # the most "central" candidate wins ties from quality.
        best_so_far = np.zeros(N, dtype=np.float32)

        for _ in range(k):
            # Marginal facility-location gain for each candidate i:
            # sum_y max(best_so_far[y], sim[i, y]) - sum_y best_so_far[y]
            # which equals sum_y max(0, sim[i, y] - best_so_far[y]).
            gains = np.maximum(0.0, sim - best_so_far[None, :]).sum(axis=1)
            gains = self.weight_diversity * gains + self.weight_quality * q
            # Block already-chosen indices.
            for i in chosen:
                gains[i] = -np.inf
            i_star = int(np.argmax(gains))
            chosen.append(i_star)
            best_so_far = np.maximum(best_so_far, sim[i_star])

        return [candidates[i] for i in chosen]


selectors.register("facility_location", FacilityLocation())
