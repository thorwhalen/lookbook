"""CLI entry point. `python -m lookbook curate ./photos --k 5`."""

from __future__ import annotations

import json
import sys
from typing import Sequence

import argh

from lookbook import curate as _curate, get_stores, registry
from lookbook import profiles as _profiles


# A handful of named recipes — sensible bundles of scorers + filters that
# users (and agents) can invoke without having to know the metric ids.
RECIPES = {
    "random": {
        "scorers": ["random_score"],
        "embedders": [],
        "filters": [],
        "selector": "top_k",
        "diagnose_clusters": 0,
    },
    "funnel": {
        "scorers": ["resolution", "file_hash", "phash", "blur", "exposure"],
        "embedders": [],
        "filters": [
            "min_resolution",
            "exposure_range",
            "min_blur",
            "no_exact_duplicate",
            "no_near_duplicate",
        ],
        "selector": "top_k",
        "diagnose_clusters": 0,
    },
    "funnel_relaxed": {
        # Same as funnel but with looser thresholds for typical phone photos.
        "scorers": ["resolution", "file_hash", "phash", "blur", "exposure"],
        "embedders": [],
        "filters": [
            ["min_resolution", {"min_long_side": 512}],
            "exposure_range",
            ["min_blur", {"threshold": 30.0}],
            "no_exact_duplicate",
            ["no_near_duplicate", {"max_distance": 8}],
        ],
        "selector": "top_k",
        "diagnose_clusters": 0,
    },
    "diverse": {
        # Phase 2: cheap funnel + DINOv2 embeddings + facility-location.
        # Pulls torch + transformers; downloads ~350MB on first use.
        "scorers": ["resolution", "file_hash", "phash", "blur", "exposure"],
        "embedders": ["dinov2"],
        "filters": [
            "min_resolution",
            "exposure_range",
            "min_blur",
            "no_exact_duplicate",
            "no_near_duplicate",
        ],
        "selector": [
            "facility_location",
            {"embedding_space": "dinov2_base",
             "quality_metric_id": "blur",
             "weight_quality": 0.05,
             "weight_diversity": 1.0},
        ],
        "diagnose_clusters": 12,
    },
    "diverse_clip": {
        # Same as `diverse` but uses CLIP (semantic similarity) instead of
        # DINOv2 (visual similarity). ~150MB download on first use.
        "scorers": ["resolution", "file_hash", "phash", "blur", "exposure"],
        "embedders": ["clip"],
        "filters": [
            "min_resolution",
            "exposure_range",
            "min_blur",
            "no_exact_duplicate",
            "no_near_duplicate",
        ],
        "selector": [
            "facility_location",
            {"embedding_space": "clip_vit_b32",
             "quality_metric_id": "blur",
             "weight_quality": 0.05,
             "weight_diversity": 1.0},
        ],
        "diagnose_clusters": 12,
    },
    "diverse_mock": {
        # No-download recipe used by tests / CI sanity checks.
        "scorers": ["random_score"],
        "embedders": ["mock"],
        "filters": [],
        "selector": [
            "facility_location",
            {"embedding_space": "mock",
             "quality_metric_id": "random_score",
             "weight_quality": 0.1,
             "weight_diversity": 1.0},
        ],
        "diagnose_clusters": 4,
    },
}


def _normalize_filter_specs(items):
    """Convert mixed list of strings / [name, kwargs] entries into specs."""
    out = []
    for it in items:
        if isinstance(it, list) and len(it) == 2 and isinstance(it[1], dict):
            out.append((it[0], it[1]))
        else:
            out.append(it)
    return out


def _normalize_selector_spec(spec):
    """Selector: either a name (str) or a [name, kwargs] list."""
    if isinstance(spec, list) and len(spec) == 2 and isinstance(spec[1], dict):
        return (spec[0], spec[1])
    return spec


def _resolve_recipe(name: str) -> dict:
    """Look up a recipe by name. In-code RECIPES win; YAML profiles fall in
    after that.
    """
    if name in RECIPES:
        return RECIPES[name]
    try:
        return _profiles.load(name)
    except KeyError:
        known = sorted(set(RECIPES) | set(_profiles.list_profiles()))
        raise SystemExit(
            f"Unknown recipe/profile: {name!r}. Known: {known}"
        )


def curate(
    source: str,
    *,
    k: int = 20,
    recipe: str = "random",
    scorer: Sequence[str] = (),
    embedder: Sequence[str] = (),
    filter: Sequence[str] = (),  # noqa: A002 (CLI flag wins over builtin)
    selector: str = "",
    diagnose_clusters: int = -1,
    in_memory: bool = False,
) -> None:
    """Curate `source` into K reference images.

    Recipes (see `lookbook list-recipes`):
      - `random`: placeholder (Phase 0).
      - `funnel`, `funnel_relaxed`: Phase 1 cheap funnel, no GPU.
      - `diverse`, `diverse_clip`: Phase 2 — funnel + embeddings + facility-
        location selection. Downloads model weights on first use.
      - `diverse_mock`: same shape as `diverse`, no-download (tests / sanity).

    Prints a JSON record of the run to stdout. The manifest persists to
    the user's app data folder unless `--in-memory` is passed.
    """
    spec = _resolve_recipe(recipe)
    scorers = list(scorer) if scorer else list(spec["scorers"])
    embedders = list(embedder) if embedder else list(spec.get("embedders", []))
    filters = (
        _normalize_filter_specs(list(filter)) if filter
        else _normalize_filter_specs(spec["filters"])
    )
    sel = selector or _normalize_selector_spec(spec["selector"])
    diag = (
        diagnose_clusters
        if diagnose_clusters >= 0
        else spec.get("diagnose_clusters", 0)
    )
    constraints = spec.get("constraints") or None

    if in_memory:
        stores = get_stores(
            images_store={}, manifest_store={}, runs_store={}, embeddings={},
        )
    else:
        stores = get_stores()

    result = _curate(
        source,
        k=k,
        scorer_ids=tuple(scorers),
        embedder_ids=tuple(embedders),
        filter_ids=tuple(filters),
        selector_id=sel,
        diagnose_clusters=diag,
        constraints=constraints,
        stores=stores,
    )
    print(json.dumps(result.to_record(), indent=2))


def list_plugins() -> None:
    """List all registered scorers / selectors / filters / embedders."""
    print("scorers:   " + ", ".join(registry.scorers.names()))
    print("selectors: " + ", ".join(registry.selectors.names()))
    print("filters:   " + ", ".join(registry.filters.names()))
    print("embedders: " + ", ".join(registry.embedders.names()))


def list_recipes() -> None:
    """List built-in curation recipes and YAML profiles."""
    seen: set = set()

    def _show(name: str, spec: dict, source: str):
        print(f"{name}  [{source}]")
        if spec.get("description"):
            for line in str(spec["description"]).strip().splitlines():
                print(f"  {line}")
        print(f"  scorers:   {', '.join(spec['scorers'])}")
        print(f"  embedders: {', '.join(spec.get('embedders', [])) or '—'}")
        fnames = [f if isinstance(f, str) else f[0] for f in spec["filters"]]
        print(f"  filters:   {', '.join(fnames) or '—'}")
        sel = spec["selector"]
        sel_name = sel if isinstance(sel, str) else sel[0]
        print(f"  selector:  {sel_name}")
        if spec.get("diagnose_clusters", 0):
            print(f"  diagnose:  cluster_coverage(n={spec['diagnose_clusters']})")

    for name, spec in RECIPES.items():
        _show(name, spec, "in-code")
        seen.add(name)
    for name in _profiles.list_profiles():
        if name in seen:
            continue
        try:
            _show(name, _profiles.load(name), "profile")
        except Exception as e:
            print(f"{name}  [profile]  (failed to load: {e})")


def main():
    parser = argh.ArghParser(prog="lookbook")
    argh.add_commands(parser, [curate, list_plugins, list_recipes])
    argh.dispatch(parser)


if __name__ == "__main__":
    main()
