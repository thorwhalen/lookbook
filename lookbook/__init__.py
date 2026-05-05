"""lookbook — distill image pools into reference sets for personalized model training.

Public facade. The package is split across:

- `lookbook.base`      — Protocols and core types (Annotation, Manifest)
- `lookbook.store`     — Repository pattern (Stores) over `dol`
- `lookbook.refs`      — ImageRef implementations
- `lookbook.manifest`  — Manifest helpers
- `lookbook.registry`  — Plugin registries (scorers, filters, embedders, selectors)
- `lookbook.pipeline`  — Pipeline orchestrator
- `lookbook.scorers`   — Per-image scorer modules (registered on import)
- `lookbook.selectors` — Selector modules (registered on import)
- `lookbook.io`        — Ingest, export

Plugin registries are accessed as `lookbook.registry.scorers` etc. (not
`lookbook.scorers`, which is the submodule that *holds* scorers).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

# Trigger registration of built-in plugins. Submodule imports happen first
# so the registry attributes referenced below are populated.
from lookbook import scorers as _scorers_pkg  # noqa: F401
from lookbook import selectors as _selectors_pkg  # noqa: F401
from lookbook import filters as _filters_pkg  # noqa: F401
from lookbook import embedders as _embedders_pkg  # noqa: F401

from lookbook import registry
from lookbook.base import (
    Annotation,
    Embedder,
    Filter,
    ImageRef,
    Manifest,
    Scorer,
    Selector,
)
from lookbook.io import ingest
from lookbook.manifest import value_of
from lookbook.pipeline import Pipeline, RunResult
from lookbook.refs import BytesImageRef, PathImageRef, UrlImageRef, to_local_path
from lookbook.interactive import InteractiveDecision, curate_interactive
from lookbook.store import Stores, get_stores


__all__ = [
    "Annotation",
    "BytesImageRef",
    "Embedder",
    "Filter",
    "ImageRef",
    "Manifest",
    "PathImageRef",
    "Pipeline",
    "RunResult",
    "Scorer",
    "Selector",
    "Stores",
    "UrlImageRef",
    "InteractiveDecision",
    "curate",
    "curate_interactive",
    "get_stores",
    "ingest",
    "registry",
    "to_local_path",
    "score",
    "value_of",
]


PluginSpec = Union[str, tuple]  # "name" or ("name", {"kw": value})


def _resolve(reg, spec: PluginSpec, *, fresh: bool = False):
    """Look up a plugin from a registry, optionally with config overrides.

    `spec` is either a registered name or a `(name, kwargs)` tuple. With
    `fresh=True` (used for filters with internal state), returns a brand
    new instance even when no overrides are given.
    """
    if isinstance(spec, str):
        name, overrides = spec, {}
    else:
        name, overrides = spec[0], dict(spec[1] or {})
    inst = reg.get(name)
    if not overrides and not fresh:
        return inst
    cls = type(inst)
    return cls(**overrides) if overrides else cls()


def curate(
    source,
    *,
    k: int = 20,
    scorer_ids: Sequence[PluginSpec] = ("random_score",),
    embedder_ids: Sequence[PluginSpec] = (),
    filter_ids: Sequence[PluginSpec] = (),
    selector_id: PluginSpec = "top_k",
    diagnose_clusters: int = 0,
    stores: Optional[Stores] = None,
    constraints: Optional[Mapping[str, Any]] = None,
) -> RunResult:
    """High-level facade: ingest a source, run a pipeline, return the result.

    Each plugin id may be either a string ("blur") or a (name, kwargs)
    tuple (("blur", {"max_side": 256})) to override the default config.

    `diagnose_clusters > 0` runs cluster-coverage diagnosis after selection
    and writes the result into the report's `notes`.
    """
    refs = ingest(source) if not isinstance(source, list) else source
    pipeline = Pipeline(
        scorers=[_resolve(registry.scorers, sp) for sp in scorer_ids],
        embedders=[_resolve(registry.embedders, sp) for sp in embedder_ids],
        # Filters always get fresh instances so stateful ones (dedup) don't
        # leak across runs.
        filters=[_resolve(registry.filters, sp, fresh=True) for sp in filter_ids],
        selector=_resolve(registry.selectors, selector_id),
        diagnose_clusters=diagnose_clusters,
    )
    return pipeline.run(refs, k=k, stores=stores, constraints=constraints)


def score(
    ref: Union[ImageRef, str],
    *,
    metric_id: str,
    stores: Optional[Stores] = None,
) -> Any:
    """Score one image against one metric. Returns the bare value.

    Caches into the manifest so repeated calls are free.
    """
    if isinstance(ref, str):
        ref = PathImageRef(path=ref)
    if stores is None:
        stores = get_stores(
            images_store={},
            manifest_store={},
            runs_store={},
            embeddings={},
        )
    s = registry.scorers.get(metric_id)
    pipeline = Pipeline(scorers=[s], selector=registry.selectors.get("top_k"))
    pipeline.run([ref], k=1, stores=stores)
    return value_of(stores.manifest, ref.image_id, metric_id)
