---
name: lookbook-http
description: Use when working on lookbook's HTTP surface — adding a new endpoint, debugging a route, configuring the server (host/port/data-root), or wiring a frontend/agent client. Triggers on lookbook.http questions, "add an endpoint", "new HTTP route", "lookbook serve", "FastAPI app", or work in lookbook/http.py.
---

# lookbook — HTTP surface

`lookbook/http.py` exposes the package over HTTP via FastAPI, built with
`qh`. The design is deliberately small: each verb is one Python function
that takes JSON-able args and returns a JSON-able dict. `qh.mk_app` wraps
that list of functions into a FastAPI app.

Read `lookbook-dev` first if you haven't.

## Route layout

Every route is `POST /<verb>` with a JSON body. No path params, no GET
verbs that aren't really getters — agents prefer one consistent shape.

| Route | Body fields | Returns |
|---|---|---|
| `POST /list_recipes` | — | `{recipes: {name: spec, ...}}` |
| `POST /list_plugins` | — | `{scorers, embedders, filters, selectors}` |
| `POST /ingest_source` | `source_path` | `{image_ids, n}` |
| `POST /curate_source` | `source_path, k, recipe` | run record |
| `POST /score_image` | `source_path \| image_id, metric_id` | `{image_id, metric_id, value}` |
| `POST /get_annotations` | `image_id` | `{image_id, annotations}` |
| `POST /list_runs` | — | `{runs: [...]}` |
| `POST /get_run` | `run_id` | run record |
| `POST /get_image` | `image_id` | `{image_id, path, n_bytes}` |

Auto-generated Swagger UI at `/docs`; ReDoc at `/redoc`.

## Server-wide stores

The HTTP server uses a single `Stores` instance for the whole process,
lazily initialized on first request. Resolution order:

1. `LOOKBOOK_DATA_ROOT` env var, if set
2. `config2py.get_app_data_folder("lookbook")` (default)

Tests substitute the singleton via `lookbook.http.reset_stores(stores)`.
The fixture in `tests/test_phase4.py` is the canonical pattern:

```python
@pytest.fixture
def memory_server():
    stores = get_stores(images_store={}, manifest_store={},
                        runs_store={}, embeddings={})
    reset_stores(stores)
    yield stores
    reset_stores(None)
```

Rule of thumb: never let a test hit `get_stores()` defaults (= the user's
app folder). Always reset the singleton.

## Adding a new endpoint

The 4-step recipe:

### 1. Write the route function

```python
def my_verb(*, image_id: str, threshold: float = 0.5) -> dict:
    """One-line docstring becomes the OpenAPI summary."""
    stores = _get_stores()
    # ... do real work, returning a JSON-able dict ...
    return {"result": "..."}
```

Rules:
- All args keyword-only with sensible defaults where possible.
- Type-annotate every arg — qh / FastAPI generates the JSON schema from
  the signature.
- Return a JSON-able dict (no numpy arrays, datetimes, etc. unless you
  serialize them yourself).
- Lazy-import any heavy deps inside the function.
- Read `_get_stores()` for the singleton; do NOT instantiate fresh
  `Stores` per request unless the user explicitly asked for it.

### 2. Register it in the function list

In `mk_lookbook_app()`:

```python
funcs = [
    list_recipes, list_plugins, ingest_source, curate_source,
    score_image, get_annotations, list_runs, get_run, get_image,
    my_verb,   # ← add here
]
```

That's it — no decorator, no path declaration. qh handles the routing.

### 3. Test via qh's TestClient

```python
def test_my_verb(client):
    r = client.post("/my_verb", json={"image_id": "abc", "threshold": 0.7})
    assert r.status_code == 200
    body = r.json()
    assert "result" in body
```

The `client` fixture (in `tests/test_phase4.py`) wraps the app with
`qh.testing.run_app`. No real port is bound; everything is in-process.

### 4. Document it

Add a row to the route table at the top of this skill, and add the
endpoint to README's CLI/curl examples if it's user-facing.

## Common patterns

### Building an `ImageRef` from request inputs

The `_ref_from_request(stores, source_path=..., image_id=...)` helper
handles both forms:

- `source_path` — direct path on the server's filesystem
- `image_id` — looked up via `stores.images[image_id]["path"]` (which
  was populated by an earlier `ingest_source` call)

If both are present, `source_path` wins. If neither is, raises.

### Returning errors

For "expected" errors (unknown ids, bad input), return a dict with an
`"error"` key — the test pattern is `assert "error" in r.json()`. For
"unexpected" errors, raise a regular Python exception — FastAPI surfaces
it as a 500.

For "user supplied bad recipe name" we currently raise `ValueError`,
which FastAPI turns into a 500. A future improvement is to register a
custom exception handler so it's a 400 instead. Until then, tests assert
`status_code != 200` rather than 400.

### Serving large image bytes

`get_image` currently returns metadata only (`path`, `n_bytes`). Returning
raw bytes via FastAPI requires a `Response` object instead of a dict —
qh's default JSON wrapping doesn't support that. When the time comes:

```python
from fastapi import Response

def get_image_bytes(image_id: str) -> Response:
    stores = _get_stores()
    rec = stores.images.get(image_id)
    with open(rec["path"], "rb") as f:
        return Response(content=f.read(), media_type="image/png")
```

This needs to bypass `qh.mk_app`'s wrapping — register the route directly
on the FastAPI app returned by `mk_lookbook_app()`.

## Anti-patterns to avoid

- **Convention-based routing.** Don't pass `use_conventions=True` to
  `qh.mk_app`. The convention-derived paths are inconsistent (some GET,
  some POST, some path-param'd) and confuse agents. Stick to uniform
  POST.
- **Per-request `get_stores()`.** Each call hits the filesystem to set
  up codecs. Use the singleton.
- **Side effects outside the function.** Don't mutate module-level state
  from inside a route. Use the stores singleton for shared state.
- **Raising on cache misses.** A route that "looks up X" should return
  a JSON error dict on miss, not 500. Reserve exceptions for user-input
  errors (which become 422 via FastAPI's pydantic validation).
- **Blocking on slow recipes.** `curate_source` with `recipe=person`
  pulls insightface and may take 30s+. For Phase 5, expose this as an
  async task via `qh.async_funcs` so the client can poll.
