"""lookbook — distill image pools into reference sets for personalized model training.

Public facade. The package is split across:

- `lookbook.base`      — Protocols and core types (Annotation, Manifest)
- `lookbook.store`     — Repository pattern (Stores) over `dol`
- `lookbook.refs`      — ImageRef implementations
- `lookbook.manifest`  — Manifest helpers
- `lookbook.registry`  — Plugin registries (scorers, filters, embedders, selectors)
- `lookbook.pipeline`  — Pipeline orchestrator
- `lookbook.facade`    — curate / score / curate_for_* entry points
- `lookbook.interactive` — human-in-the-loop curate loop
- `lookbook.scorers`   — Per-image scorer modules (registered on import)
- `lookbook.selectors` — Selector modules (registered on import)
- `lookbook.io`        — Ingest, export

Plugin registries are accessed as `lookbook.registry.scorers` etc. (not
`lookbook.scorers`, which is the submodule that *holds* scorers).

This module only re-exports; it defines nothing of its own — not even a
`from __future__ import annotations`, so `dir(lookbook)` stays the public API
plus the submodules.
"""

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
from lookbook.facade import (
    PluginSpec,
    curate,
    curate_for_character,
    curate_for_environment,
    score,
)
from lookbook.io import ingest
from lookbook.manifest import value_of
from lookbook.pipeline import Pipeline, RunResult
from lookbook.refs import BytesImageRef, PathImageRef, UrlImageRef, to_local_path
from lookbook.interactive import InteractiveDecision, curate_interactive
from lookbook.scorers.identity import (
    IdentitySimilarity,
    SimilarityResult,
    compare_to_reference,
)
from lookbook.store import Stores, get_stores


__all__ = [
    "Annotation",
    "BytesImageRef",
    "Embedder",
    "Filter",
    "IdentitySimilarity",
    "ImageRef",
    "Manifest",
    "PathImageRef",
    "Pipeline",
    "RunResult",
    "Scorer",
    "Selector",
    "SimilarityResult",
    "Stores",
    "UrlImageRef",
    "InteractiveDecision",
    "compare_to_reference",
    "curate",
    "curate_for_character",
    "curate_for_environment",
    "curate_interactive",
    "get_stores",
    "ingest",
    "registry",
    "to_local_path",
    "score",
    "value_of",
]
