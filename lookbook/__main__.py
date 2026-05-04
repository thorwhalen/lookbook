"""CLI entry point. `python -m lookbook curate ./photos --k 5`."""

from __future__ import annotations

import json
import sys
from typing import Sequence

import argh

from lookbook import curate as _curate, get_stores, registry


# A handful of named recipes — sensible bundles of scorers + filters that
# users (and agents) can invoke without having to know the metric ids.
RECIPES = {
    "random": {
        "scorers": ["random_score"],
        "filters": [],
        "selector": "top_k",
    },
    "funnel": {
        "scorers": ["resolution", "file_hash", "phash", "blur", "exposure"],
        "filters": [
            "min_resolution",
            "exposure_range",
            "min_blur",
            "no_exact_duplicate",
            "no_near_duplicate",
        ],
        "selector": "top_k",
    },
    "funnel_relaxed": {
        # Same as funnel but with looser thresholds for typical phone photos.
        "scorers": ["resolution", "file_hash", "phash", "blur", "exposure"],
        "filters": [
            ["min_resolution", {"min_long_side": 512}],
            "exposure_range",
            ["min_blur", {"threshold": 30.0}],
            "no_exact_duplicate",
            ["no_near_duplicate", {"max_distance": 8}],
        ],
        "selector": "top_k",
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


def curate(
    source: str,
    *,
    k: int = 20,
    recipe: str = "random",
    scorer: Sequence[str] = (),
    filter: Sequence[str] = (),  # noqa: A002 (CLI flag wins over builtin)
    selector: str = "",
    in_memory: bool = False,
) -> None:
    """Curate `source` into K reference images.

    By default uses the `random` recipe (placeholder, Phase 0). Pass
    `--recipe funnel` to apply the Phase 1 cheap-funnel pipeline. Manual
    overrides via `--scorer`, `--filter`, `--selector` win when provided.

    Prints a JSON record of the run to stdout. The manifest persists to
    the user's app data folder unless `--in-memory` is passed.
    """
    if recipe not in RECIPES:
        raise SystemExit(
            f"Unknown recipe: {recipe!r}. Known: {sorted(RECIPES)}"
        )
    spec = RECIPES[recipe]
    scorers = list(scorer) if scorer else list(spec["scorers"])
    filters = (
        _normalize_filter_specs(list(filter)) if filter
        else _normalize_filter_specs(spec["filters"])
    )
    sel = selector or spec["selector"]

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
        filter_ids=tuple(filters),
        selector_id=sel,
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
    """List built-in curation recipes."""
    for name, spec in RECIPES.items():
        print(f"{name}:")
        print(f"  scorers:  {', '.join(spec['scorers'])}")
        fnames = [f if isinstance(f, str) else f[0] for f in spec["filters"]]
        print(f"  filters:  {', '.join(fnames) or '—'}")
        print(f"  selector: {spec['selector']}")


def main():
    parser = argh.ArghParser(prog="lookbook")
    argh.add_commands(parser, [curate, list_plugins, list_recipes])
    argh.dispatch(parser)


if __name__ == "__main__":
    main()
