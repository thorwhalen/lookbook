---
name: lookbook-dev
description: Use when developing or modifying the lookbook package — extending the curation pipeline, adding scorers/selectors/embedders, working with the manifest, swapping storage backends. Triggers on lookbook architecture questions, "how do I add a scorer", "where does the manifest live", "what's the registry pattern", or any work inside /Users/thorwhalen/Dropbox/py/proj/t/lookbook/.
---

# lookbook — developer's reference

`lookbook` curates image pools into reference sets for training personalized
models (e.g. character LoRAs). The package is built on a small, opinionated
architecture; everything below assumes you've read it.

For the long-form context, see:
- `misc/docs/lookbook_design_report.md` (problem framing, metrics catalogue)
- `misc/docs/lookbook_development_plan.md` (phasing, decisions, status)

## Current state at a glance

Always cross-check against `misc/docs/lookbook_development_plan.md` (which
carries the authoritative status table), but as a quick orientation:

- **Phase 0 (skeleton + stores)** — done. Protocols, manifest, registry,
  pipeline, JSON-on-disk persistence via `dol`, app-folder paths via
  `config2py`, CLI dispatch. `RandomScore` + `TopK`.
- **Phase 1 (cheap funnel)** — done. Real scorers (`resolution`,
  `file_hash`, `phash`, `blur`, `exposure`), filters (`min_resolution`,
  `min_blur`, `exposure_range`, `no_exact_duplicate`, `no_near_duplicate`),
  drop-attributing `Report`, named CLI recipes (`funnel`, `funnel_relaxed`).
- **Phase 2 (embeddings + submodular)** — not started. Will pull `torch`,
  add CLIP/DINOv2 embedders, LAION-Aesthetic scorer, an `apricot`-based
  facility-location selector, and cluster-coverage diagnosis.
- **Phase 3+ (person profile, HTTP via qh, MCP)** — not started.

## The five-layer architecture

```
Interface (CLI, HTTP via qh, MCP via py2mcp, Python lib)
   ↓
Recipe / facade   (lookbook.curate, profiles)
   ↓
Orchestration    (lookbook.pipeline, budget, manifest)
   ↓
Plugin layer     (Scorer, Filter, Embedder, Selector — Protocols in base.py)
   ↓
Backend          (CLIP, DINOv2, InsightFace, pyiqa, apricot — wrapped)
```

**No `import torch` above the backend layer.** Heavy ML deps must be
lazy-imported inside scorer/embedder methods, never at module top. This is
what keeps the package laptop-installable and lets the planning/inspection
layers work without GPU.

## The four protocols are the extension surface

Defined in `lookbook/base.py`:

- `ImageRef` — has `image_id`, `metadata`, `open()`, `bytes()`. Three concrete
  impls in `refs.py`: `PathImageRef`, `BytesImageRef`, `UrlImageRef`.
- `Scorer` — `metric_id`, `cost_tier`, `requires`, `config_hash`, `score(ref, manifest) -> Any`.
- `Filter` — `keep(ref, manifest) -> bool`.
- `Embedder` — `space_id`, `cost_tier`, `embed(ref) -> ndarray`.
- `Selector` — `select(candidates, manifest, k, constraints) -> list[ImageRef]`.

New plugins are **registered**, never subclassed. Use the registry decorators
or call `register(...)` directly:

```python
from lookbook.registry import scorers, register_scorer

@dataclass
class MyScorer:
    metric_id: str = "my_metric"
    cost_tier: int = 1
    requires: tuple = ()
    @property
    def config_hash(self): return "v1"
    def score(self, ref, manifest): return ...

scorers.register("my_metric", MyScorer())
```

For the eight-step recipe to add a scorer end-to-end, see the
`lookbook-add-scorer` skill.

### Tunable plugins via the (name, kwargs) tuple

Every plugin id accepted by the facade may be either a string or a
`(name, kwargs)` tuple:

```python
curate(
    source,
    k=20,
    scorer_ids=("resolution", ("blur", {"max_side": 256})),
    filter_ids=(("min_resolution", {"min_long_side": 800}), "no_exact_duplicate"),
    selector_id=("top_k", {"metric_id": "blur"}),
)
```

The facade resolves the tuple by re-instantiating the registered class
with the overrides. Stateful filters always get fresh instances regardless
of overrides, so dedup state never leaks across runs.

The CLI `--recipe` named bundles use the same shape; see
`lookbook list-recipes` for the built-in ones.

## The manifest is the SSOT

The manifest is a `MutableMapping[(image_id, metric_id), Annotation]`. That's
the whole abstraction. Everywhere we annotate `manifest: Manifest`, we mean
"any mapping with that key/value shape." This is what lets the same code run
against `dict`, `JsonFiles`, `mongodol`, `s3dol`, etc.

Helpers live in `lookbook.manifest`: `get_annotation`, `put_annotation`,
`has_annotation`, `iter_annotations_for`, `image_ids`, `value_of`. Use them
instead of poking the mapping directly when you can — they survive future
changes to the codec.

### config_hash is the cache key

When a scorer runs against an image, the orchestrator first checks the
manifest for `(image_id, metric_id)`. If found AND `existing.config_hash ==
scorer.config_hash`, it skips the scorer. Changing a threshold, model, or
parameter must change the `config_hash` to invalidate.

The standard pattern is `config_hash = sha1(repr(sorted(config.items())))[:12]`
exposed as a `@property` on the scorer. See `scorers/technical.py:RandomScore`
for the canonical example.

## Stores: the repository pattern via `dol`

`lookbook.store.Stores` bundles three (and a half) MutableMappings:

| Store | Keys | Values |
|---|---|---|
| `images` | `image_id` | bytes / path / url payload |
| `manifest` | `(image_id, metric_id)` | `Annotation` |
| `runs` | `run_id` | run record (JSON-able dict) |
| `embeddings` | `space_id -> { image_id: vector }` | per-space vector index |

`get_stores()` is the factory. With no args, it points at the user's app data
folder via `config2py.get_app_data_folder("lookbook")`. For tests, pass
`dict()` to every slot:

```python
get_stores(images_store={}, manifest_store={}, runs_store={}, embeddings={})
```

The manifest goes through a codec (`lookbook.store.manifest_codec`) that:
- Translates `(image_id, metric_id)` ↔ `"image_id::metric_id.json"` filenames.
- Serializes `Annotation` ↔ JSON-able dicts.

Swapping to S3/Mongo/SQLite is a one-line change — pass an alternate
`manifest_store` to `get_stores` and the rest of the package is unaffected.
Details in the `lookbook-storage` skill.

## Pipeline orchestration

`lookbook.pipeline.Pipeline.run(...)`:

1. Topo-sorts scorers by their `requires` declarations (within ties, cheaper
   `cost_tier` first).
2. For each scorer × image, checks the manifest cache. On miss, calls
   `scorer.score(ref, manifest)` and writes the result.
3. Applies filters (all must return True for an image to survive).
4. Calls `selector.select(survivors, manifest, k, constraints)`.
5. Persists a run record to `stores.runs`.

For Phase 0 the orchestrator is a hand-rolled topo walk. Phase 1+ swaps in
`meshed.DAG` for richer dependency graphs and persistent caching. When you
make that switch: the `Scorer.requires` field is already the dependency
graph — just feed it to meshed.

## Folder & path defaults

Everything goes through `lookbook._paths`:
- `default_data_root()` → `~/.local/share/lookbook` on Linux, `~/Library/Application Support/lookbook` on macOS.
- `default_cache_root()` for regeneratable model weights / embeddings.
- `default_config_root()` for user-edited recipes/profiles.

Do **not** hardcode paths anywhere else; route them through this module so
swapping app names is a one-place change.

## Conventions for new code

- Dataclasses for data; functions over methods; `Protocol` over ABC.
- Modules in `scorers/`, `selectors/` register their plugins on import. The
  package `__init__` imports those subpackages first to populate the registry.
- The submodule names `scorers/`, `selectors/`, `filters/`, `embedders/`
  collide with the registry attribute names, so the registries are accessed
  as `lookbook.registry.scorers` etc. — never `lookbook.scorers`.
- Tests pass `dict()` for every store. Don't hit the user's app folder in
  tests. Use `tmp_path` only when verifying disk persistence specifically.
- Keyword-only arguments after the third position. No magic numbers.

## Common tasks → which skill

| Task | Skill |
|---|---|
| Add a scorer (any tier) | `lookbook-add-scorer` |
| Add a selector | `lookbook-add-selector` |
| Swap storage backend, debug persistence | `lookbook-storage` |
| Write a subject profile (person, product, …) | `lookbook-profile` (Phase 3+) |
| Help an agent USE lookbook to curate | `lookbook-curate` (Phase 4+) |
