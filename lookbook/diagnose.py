"""Set-level diagnosis: what does the kept set cover, what's missing.

Phase 2 ships cluster-coverage. The function takes the candidate pool and
the kept subset, clusters the candidates by their embeddings, and reports
how many clusters the kept set populates plus which clusters are
empty / underfilled.

This is the read-only counterpart to the selector — same data, different
question. Selectors *choose*; diagnosers *describe*.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Optional

import numpy as np

from lookbook.base import ImageRef


def cluster_coverage(
    candidates: Iterable[ImageRef],
    kept: Iterable[ImageRef],
    embeddings: Mapping[str, np.ndarray],
    *,
    n_clusters: int = 12,
    random_state: int = 0,
) -> dict:
    """Cluster the candidate embeddings into `n_clusters` and report coverage.

    Returns a dict with:
      - `n_clusters`: actual number of clusters used (capped at len(candidates))
      - `n_clusters_filled`: clusters with at least one kept image
      - `cluster_sizes_candidates`: list of int, candidate count per cluster
      - `cluster_sizes_kept`: list of int, kept count per cluster
      - `empty_clusters`: list of int (indexes of clusters with zero kept)
      - `underrepresented_clusters`: clusters with kept_count == 0 and
        candidate_count >= 2 (the meaningful gaps)
    """
    candidates = list(candidates)
    kept = list(kept)
    if not candidates:
        return {
            "n_clusters": 0,
            "n_clusters_filled": 0,
            "cluster_sizes_candidates": [],
            "cluster_sizes_kept": [],
            "empty_clusters": [],
            "underrepresented_clusters": [],
        }

    X = np.stack(
        [np.asarray(embeddings[r.image_id], dtype=np.float32) for r in candidates]
    )
    n = X.shape[0]
    n_clusters_eff = max(1, min(n_clusters, n))

    if n_clusters_eff == 1:
        # Degenerate case — everyone in cluster 0.
        labels = np.zeros(n, dtype=int)
    else:
        from sklearn.cluster import KMeans  # local import: optional dep

        km = KMeans(
            n_clusters=n_clusters_eff,
            n_init="auto",
            random_state=random_state,
        )
        labels = km.fit_predict(X)

    cand_id_to_label = {r.image_id: int(labels[i]) for i, r in enumerate(candidates)}

    cand_counts = Counter(int(l) for l in labels)
    kept_counts = Counter(
        cand_id_to_label[r.image_id] for r in kept if r.image_id in cand_id_to_label
    )

    cluster_sizes_candidates = [cand_counts.get(c, 0) for c in range(n_clusters_eff)]
    cluster_sizes_kept = [kept_counts.get(c, 0) for c in range(n_clusters_eff)]
    empty = [c for c in range(n_clusters_eff) if cluster_sizes_kept[c] == 0]
    underrep = [c for c in empty if cluster_sizes_candidates[c] >= 2]

    return {
        "n_clusters": n_clusters_eff,
        "n_clusters_filled": int(sum(1 for v in cluster_sizes_kept if v > 0)),
        "cluster_sizes_candidates": cluster_sizes_candidates,
        "cluster_sizes_kept": cluster_sizes_kept,
        "empty_clusters": empty,
        "underrepresented_clusters": underrep,
    }
