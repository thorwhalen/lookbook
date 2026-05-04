"""HTTP surface for lookbook, built with `qh` over FastAPI.

Each HTTP route is a thin function that takes JSON-able args and returns
a JSON-able dict. The route functions are the *only* thing this module
adds — all real work is delegated to the existing facade.

Build the app with `mk_lookbook_app()` and run it with `serve()` (or use
the `lookbook serve` CLI command).
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Sequence

from lookbook import (
    PathImageRef,
    Pipeline,
    curate as _curate,
    get_stores,
    registry,
)
from lookbook import profiles as _profiles
from lookbook.io.ingest import ingest_to_store
from lookbook.manifest import iter_annotations_for, value_of


# ---------------------------------------------------------------------------
# Server-wide stores singleton
# ---------------------------------------------------------------------------

_stores = None


def _get_stores():
    """Return the server's `Stores` singleton, lazy-initialized.

    The HTTP server uses a single Stores instance backed by the user's app
    data folder. Override the data root by setting `LOOKBOOK_DATA_ROOT`
    before starting the server.
    """
    global _stores
    if _stores is None:
        root = os.environ.get("LOOKBOOK_DATA_ROOT") or None
        _stores = get_stores(root=root)
    return _stores


def reset_stores(stores=None):
    """Replace the server-wide stores. Used by tests; rare otherwise."""
    global _stores
    _stores = stores


# ---------------------------------------------------------------------------
# Route functions
# ---------------------------------------------------------------------------


def list_recipes() -> dict:
    """All available recipes / profiles, with their specs."""
    out = {}
    # In-code RECIPES dict from __main__.
    from lookbook.__main__ import RECIPES as _RECIPES

    for name, spec in _RECIPES.items():
        out[name] = {**spec, "source": "in-code"}
    for name in _profiles.list_profiles():
        if name in out:
            continue
        try:
            spec = _profiles.load(name)
            out[name] = {**spec, "source": "profile"}
        except Exception as e:
            out[name] = {"source": "profile", "error": str(e)}
    return {"recipes": out}


def list_plugins() -> dict:
    """All registered scorers / embedders / filters / selectors."""
    return {
        "scorers": registry.scorers.names(),
        "embedders": registry.embedders.names(),
        "filters": registry.filters.names(),
        "selectors": registry.selectors.names(),
    }


def ingest_source(source_path: str) -> dict:
    """Ingest a directory or file. Records id->path in `stores.images`.

    Returns the list of resolved image ids.
    """
    stores = _get_stores()
    refs = ingest_to_store(source_path, stores)
    return {
        "image_ids": [r.image_id for r in refs],
        "n": len(refs),
    }


def curate_source(
    source_path: str,
    *,
    k: int = 20,
    recipe: str = "funnel",
) -> dict:
    """Run a recipe on a directory of images. Returns the run record."""
    spec = _resolve_recipe(recipe)
    stores = _get_stores()
    # Ensure the id->path map is populated before running.
    ingest_to_store(source_path, stores)

    selector_spec = _normalize_selector_spec(spec["selector"])
    filters = _normalize_filter_specs(spec["filters"])

    result = _curate(
        source_path,
        k=k,
        scorer_ids=tuple(spec["scorers"]),
        embedder_ids=tuple(spec.get("embedders", [])),
        filter_ids=tuple(filters),
        selector_id=selector_spec,
        diagnose_clusters=spec.get("diagnose_clusters", 0),
        constraints=spec.get("constraints") or None,
        stores=stores,
    )
    return result.to_record()


def score_image(*, source_path: str = "", image_id: str = "",
                metric_id: str) -> dict:
    """Score a single image by metric id.

    Pass either `source_path` (file path on the server) or `image_id`
    (must already have a path recorded via ingest).
    """
    stores = _get_stores()
    ref = _ref_from_request(stores, source_path=source_path, image_id=image_id)
    scorer = registry.scorers.get(metric_id)
    pipeline = Pipeline(scorers=[scorer], selector=registry.selectors.get("top_k"))
    pipeline.run([ref], k=1, stores=stores)
    v = value_of(stores.manifest, ref.image_id, metric_id)
    return {"image_id": ref.image_id, "metric_id": metric_id, "value": v}


def get_annotations(image_id: str) -> dict:
    """All annotations for one image (drawn from the manifest)."""
    stores = _get_stores()
    out = []
    for ann in iter_annotations_for(stores.manifest, image_id):
        out.append({
            "metric_id": ann.metric_id,
            "value": ann.value,
            "config_hash": ann.config_hash,
            "cost_tier": ann.cost_tier,
            "backend": ann.backend,
            "timestamp": ann.timestamp.isoformat() if ann.timestamp else None,
        })
    return {"image_id": image_id, "annotations": out}


def list_runs() -> dict:
    """All run ids stored on the server."""
    stores = _get_stores()
    keys = list(stores.runs)
    # Strip a trailing .json suffix when present, for cleaner response.
    ids = [k[: -len(".json")] if k.endswith(".json") else k for k in keys]
    return {"runs": sorted(ids)}


def get_run(run_id: str) -> dict:
    """One run record by id."""
    stores = _get_stores()
    # Try id, then id+".json".
    for key in (run_id, f"{run_id}.json"):
        if key in stores.runs:
            return stores.runs[key]
    raise KeyError(f"unknown run_id: {run_id!r}")


def get_image(image_id: str):
    """Return the raw bytes of an image by id.

    The server reads from `stores.images[image_id]["path"]`. If the entry
    is missing, returns a 404-style error dict (FastAPI surfaces this as a
    JSON response; a future iteration may switch to a streaming binary
    response via FastAPI's `Response`).
    """
    stores = _get_stores()
    rec = stores.images.get(image_id)
    if not rec or "path" not in rec:
        return {"error": "unknown image_id", "image_id": image_id}
    path = rec["path"]
    if not os.path.isfile(path):
        return {"error": "path no longer present on disk", "path": path}
    with open(path, "rb") as f:
        data = f.read()
    return {
        "image_id": image_id,
        "path": path,
        "n_bytes": len(data),
        # Bytes themselves are returned via a separate streaming route in
        # production. For Phase 4 we return only metadata so the response
        # stays JSON-friendly. The client can fetch the file directly when
        # the server is local; for remote operation, see Phase 5.
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_recipe(name: str) -> dict:
    from lookbook.__main__ import RECIPES as _RECIPES

    if name in _RECIPES:
        return _RECIPES[name]
    try:
        return _profiles.load(name)
    except KeyError:
        known = sorted(set(_RECIPES) | set(_profiles.list_profiles()))
        raise ValueError(f"Unknown recipe/profile: {name!r}. Known: {known}")


def _normalize_filter_specs(items):
    out = []
    for it in items:
        if isinstance(it, list) and len(it) == 2 and isinstance(it[1], dict):
            out.append((it[0], it[1]))
        else:
            out.append(it)
    return out


def _normalize_selector_spec(spec):
    if isinstance(spec, list) and len(spec) == 2 and isinstance(spec[1], dict):
        return (spec[0], spec[1])
    return spec


def _ref_from_request(stores, *, source_path: str = "", image_id: str = ""):
    """Build an ImageRef from the HTTP request inputs."""
    if source_path:
        return PathImageRef(path=source_path)
    if image_id:
        rec = stores.images.get(image_id)
        if not rec or "path" not in rec:
            raise KeyError(
                f"image_id {image_id!r} is not in the images store. "
                f"Call /ingest_source first."
            )
        return PathImageRef(path=rec["path"], image_id=image_id)
    raise ValueError("either source_path or image_id must be provided")


# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------


def mk_lookbook_app(*, app=None, **kwargs):
    """Build a FastAPI app exposing the lookbook HTTP surface.

    Each function becomes a POST endpoint with a JSON body matching its
    keyword arguments. Use the auto-generated `/docs` (Swagger) or
    `/redoc` for interactive exploration. Disabling qh's convention-based
    routing keeps the path scheme uniform — every route is `POST /<verb>`
    and takes a JSON body — which is the pattern most agent clients
    (and our future MCP layer) expect.
    """
    import qh  # local: keep qh out of `import lookbook` cost

    funcs = [
        list_recipes,
        list_plugins,
        ingest_source,
        curate_source,
        score_image,
        get_annotations,
        list_runs,
        get_run,
        get_image,
    ]
    return qh.mk_app(funcs, app=app, use_conventions=False, **kwargs)


def serve(*, host: str = "127.0.0.1", port: int = 8000, **kwargs):
    """Run the lookbook HTTP server (uvicorn under the hood)."""
    try:
        import uvicorn  # type: ignore
    except ImportError as e:
        raise ImportError(
            "lookbook serve requires uvicorn. "
            "`pip install lookbook[http]` or `pip install uvicorn fastapi`."
        ) from e

    app = mk_lookbook_app()
    uvicorn.run(app, host=host, port=port, **kwargs)
