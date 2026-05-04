"""Pipeline orchestrator.

A `Pipeline` is a list of scorers (and, later, filters/embedders) plus a
final selector. Running it:

1. Topologically orders the scorers by their `requires` declarations.
2. Walks them in cost-tier order, writing results to the manifest.
3. Hands the surviving candidates plus the read-only manifest to the
   selector.

For Phase 0 the orchestration is deliberately simple — a hand-rolled topo
walk. Phase 1+ will swap in `meshed.DAG` for richer dependency graphs and
caching.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from lookbook.base import (
    Annotation,
    Embedder,
    Filter,
    ImageRef,
    Manifest,
    Scorer,
    Selector,
)
from lookbook.manifest import has_annotation, put_annotation
from lookbook.report import Report, attribute_drops
from lookbook.store import Stores, get_stores


# Stable manifest metric_id used to mark "this image has been embedded in
# space `space_id` with this config_hash." The vector itself lives in
# `stores.embeddings[space_id][image_id]`; the manifest only tracks
# presence so cache lookups don't have to read the vector.
def _embedder_metric_id(space_id: str) -> str:
    return f"emb:{space_id}"


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """The result of running a pipeline."""

    run_id: str
    kept: list[ImageRef]
    candidates: list[ImageRef]
    selector_id: str
    scorer_ids: list[str]
    started_at: datetime
    finished_at: datetime
    report: dict = field(default_factory=dict)

    def to_record(self) -> dict:
        return {
            "run_id": self.run_id,
            "kept": [r.image_id for r in self.kept],
            "candidates": [r.image_id for r in self.candidates],
            "selector_id": self.selector_id,
            "scorer_ids": self.scorer_ids,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "report": self.report,
        }


# ---------------------------------------------------------------------------
# Topological ordering
# ---------------------------------------------------------------------------


def _topo_sort(scorers: Sequence[Scorer]) -> list[Scorer]:
    """Order scorers so that each runs after its `requires` predecessors.

    Within a tier, ties are broken by `cost_tier` (cheaper first) and then
    by registration order.
    """
    by_id: dict[str, Scorer] = {s.metric_id: s for s in scorers}
    visited: set[str] = set()
    ordered: list[Scorer] = []
    visiting: set[str] = set()

    def visit(s: Scorer):
        if s.metric_id in visited:
            return
        if s.metric_id in visiting:
            raise ValueError(f"cyclic dependency among scorers near {s.metric_id!r}")
        visiting.add(s.metric_id)
        for dep in getattr(s, "requires", ()):
            if dep in by_id:
                visit(by_id[dep])
            # Missing deps are not an error here — they may have been pre-
            # populated in the manifest by an earlier run.
        visiting.discard(s.metric_id)
        visited.add(s.metric_id)
        ordered.append(s)

    # Visit cheaper tiers first so a cheap dep of an expensive scorer runs
    # earlier when the topological order is otherwise free.
    for s in sorted(scorers, key=lambda x: getattr(x, "cost_tier", 0)):
        visit(s)
    return ordered


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class Pipeline:
    """An ordered set of scorers + embedders + filters + a final selector."""

    scorers: list[Scorer] = field(default_factory=list)
    embedders: list[Embedder] = field(default_factory=list)
    filters: list[Filter] = field(default_factory=list)
    selector: Optional[Selector] = None
    diagnose_clusters: int = 0  # 0 = skip diagnosis, >0 = run cluster_coverage

    def run(
        self,
        candidates: Iterable[ImageRef],
        *,
        k: int,
        stores: Optional[Stores] = None,
        constraints: Optional[Mapping[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> RunResult:
        """Score, embed, filter, and select.

        `stores` defaults to in-memory. Pass `get_stores()` for the user's
        app data folder defaults.
        """
        if self.selector is None:
            raise ValueError("Pipeline.selector must be set before run()")

        candidates = list(candidates)
        stores = (
            stores
            if stores is not None
            else get_stores(
                images_store={},
                manifest_store={},
                runs_store={},
                embeddings={},
            )
        )
        manifest: Manifest = stores.manifest

        started = datetime.now(timezone.utc)

        ordered = _topo_sort(self.scorers)
        for scorer in ordered:
            for ref in candidates:
                # Cache hit: same (image_id, metric_id) already present.
                # config_hash mismatches are treated as misses.
                existing = manifest.get((ref.image_id, scorer.metric_id))
                if existing is not None and existing.config_hash == getattr(
                    scorer, "config_hash", ""
                ):
                    continue
                value = scorer.score(ref, manifest)
                put_annotation(
                    manifest,
                    Annotation(
                        image_id=ref.image_id,
                        metric_id=scorer.metric_id,
                        value=value,
                        config_hash=getattr(scorer, "config_hash", ""),
                        cost_tier=getattr(scorer, "cost_tier", 0),
                        backend=getattr(scorer, "backend", ""),
                    ),
                )

        # Run embedders. Vectors go to stores.embeddings[space_id][image_id];
        # the manifest gets a presence flag so cache hits don't reload the
        # vector from disk.
        for emb in self.embedders:
            space_id = emb.space_id
            metric_id = _embedder_metric_id(space_id)
            cfg = getattr(emb, "config_hash", "")
            # Auto-create the per-space store on first use so callers don't
            # have to pre-populate `stores.embeddings`.
            if space_id not in stores.embeddings:
                stores.embeddings[space_id] = {}
            space_store = stores.embeddings[space_id]
            for ref in candidates:
                existing = manifest.get((ref.image_id, metric_id))
                if (
                    existing is not None
                    and existing.config_hash == cfg
                    and ref.image_id in space_store
                ):
                    continue
                vec = emb.embed(ref)
                # Store as plain list so the default JSON codec works.
                space_store[ref.image_id] = (
                    vec.tolist() if hasattr(vec, "tolist") else list(vec)
                )
                put_annotation(
                    manifest,
                    Annotation(
                        image_id=ref.image_id,
                        metric_id=metric_id,
                        value=True,
                        config_hash=cfg,
                        cost_tier=getattr(emb, "cost_tier", 0),
                        backend=getattr(emb, "backend", ""),
                    ),
                )

        survivors, drops = attribute_drops(candidates, self.filters, manifest)

        # Pre-fetch embeddings for the selector when it asks for a space.
        sel_constraints = dict(constraints or {})
        sel_space = getattr(self.selector, "embedding_space", None)
        if sel_space and sel_space in stores.embeddings:
            import numpy as np  # local: numpy is a core dep but optional here

            space_store = stores.embeddings[sel_space]
            sel_constraints["embeddings"] = {
                r.image_id: np.asarray(space_store[r.image_id], dtype=np.float32)
                for r in survivors
                if r.image_id in space_store
            }

        kept = self.selector.select(
            survivors, manifest, k=k, constraints=sel_constraints
        )

        selector_id = getattr(
            self.selector, "selector_id", type(self.selector).__name__
        )
        report = Report(
            n_candidates=len(candidates),
            n_survivors=len(survivors),
            n_kept=len(kept),
            dropped_by_filter=drops,
            scorer_ids=[s.metric_id for s in ordered],
            selector_id=selector_id,
            notes={"embedder_ids": [e.space_id for e in self.embedders]}
            if self.embedders
            else {},
        )

        # Cluster-coverage diagnosis. Skipped when `diagnose_clusters == 0`
        # or when there's no embedding space to cluster over.
        if (
            self.diagnose_clusters > 0
            and sel_space
            and sel_constraints.get("embeddings")
        ):
            from lookbook.diagnose import cluster_coverage

            coverage = cluster_coverage(
                survivors,
                kept,
                sel_constraints["embeddings"],
                n_clusters=self.diagnose_clusters,
            )
            report.notes["cluster_coverage"] = coverage

        finished = datetime.now(timezone.utc)
        rid = run_id or f"run-{started.strftime('%Y%m%dT%H%M%S')}"
        result = RunResult(
            run_id=rid,
            kept=list(kept),
            candidates=candidates,
            selector_id=selector_id,
            scorer_ids=[s.metric_id for s in ordered],
            started_at=started,
            finished_at=finished,
            report={
                **report.to_dict(),
            },
        )

        # Persist the run record. JSON-able dict by construction.
        try:
            stores.runs[f"{rid}.json"] = result.to_record()
        except Exception:
            stores.runs[rid] = result.to_record()

        return result
