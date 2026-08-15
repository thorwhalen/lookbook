"""High-level facades — the one-call entry points into the pipeline.

`curate` is the general one: ingest a source, run scorers/embedders/filters,
select K. `curate_for_character` and `curate_for_environment` are opinionated
presets over it, and `score` is the single-image, single-metric shortcut.

Everything here is re-exported from `lookbook`, which is where callers should
import it from. Plugins are named, never subclassed: each `*_ids` argument
takes registry names, optionally paired with config overrides as a
`(name, kwargs)` tuple.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from lookbook.base import ImageRef
from lookbook.io import ingest
from lookbook.manifest import value_of
from lookbook.pipeline import Pipeline, RunResult
from lookbook.refs import PathImageRef
from lookbook.store import Stores, get_stores
from lookbook import registry


PluginSpec = str | tuple  # "name" or ("name", {"kw": value})


def resolve_plugin(reg, spec: PluginSpec, *, fresh: bool = False):
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
        scorers=[resolve_plugin(registry.scorers, sp) for sp in scorer_ids],
        embedders=[resolve_plugin(registry.embedders, sp) for sp in embedder_ids],
        # Filters always get fresh instances so stateful ones (dedup) don't
        # leak across runs.
        filters=[resolve_plugin(registry.filters, sp, fresh=True) for sp in filter_ids],
        selector=resolve_plugin(registry.selectors, selector_id),
        diagnose_clusters=diagnose_clusters,
    )
    return pipeline.run(refs, k=k, stores=stores, constraints=constraints)


def score(
    ref: ImageRef | str,
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


def curate_for_character(
    source,
    *,
    k: int = 1,
    face_detector: PluginSpec = "insightface",
    stores: Optional[Stores] = None,
    constraints: Optional[Mapping[str, Any]] = None,
) -> RunResult:
    """Curate the best reference image(s) of a *known* character from a pool.

    Opinionated facade over :func:`curate`, tuned for the "pick the
    reference image of this one character" job (IP-adapter conditioning,
    model-sheet seeding): every image is scored for resolution, sharpness,
    exposure and face quality, then the top ``k`` are taken by the
    composite ``face_quality`` metric.

    Identity is *not* scored — the pool is assumed to already be one
    character, so what matters is which frame shows them most usably: in
    focus, well exposed, face clearly visible and well sized. To check
    whether a *generation* still matches a locked reference, use
    :func:`lookbook.compare_to_reference` instead.

    `face_detector` names the registered face-box scorer:

    - ``"insightface"`` — real RetinaFace detection; needs
      ``pip install lookbook[person]``. The default.
    - ``"mock_face"`` — the deterministic centred-box detector, for tests
      and demos without the ML dependency.

    Images with no detected face score ``0`` and sort last; the pool is
    never filtered down to empty, so a small or awkward pool still yields
    a best-effort pick.
    """
    return curate(
        source,
        k=k,
        scorer_ids=(
            "resolution",
            "blur",
            "exposure",
            face_detector,
            "face_area",
            "face_quality",
        ),
        selector_id=("top_k", {"metric_id": "face_quality"}),
        stores=stores,
        constraints=constraints,
    )


def curate_for_environment(
    source,
    *,
    k: int = 1,
    stores: Optional[Stores] = None,
    constraints: Optional[Mapping[str, Any]] = None,
) -> RunResult:
    """Curate the best reference image(s) for a non-person subject.

    Opinionated facade over :func:`curate` for pools where face detection
    is moot — environment plates, prop references, style boards. Each
    image is scored for resolution, sharpness and exposure, folded into
    the composite ``technical_quality`` metric, and the top ``k`` are
    taken.

    Unlike :func:`curate_for_character` this needs no ML dependency — the
    scorers run on Pillow + numpy (cv2 is used for sharpness when present,
    with a numpy fallback otherwise).
    """
    return curate(
        source,
        k=k,
        scorer_ids=("resolution", "blur", "exposure", "technical_quality"),
        selector_id=("top_k", {"metric_id": "technical_quality"}),
        stores=stores,
        constraints=constraints,
    )
