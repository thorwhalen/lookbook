---
name: lookbook-dev
description: Use when developing or modifying the lookbook package — extending the curation pipeline, adding scorers/selectors/embedders, working with the manifest, swapping storage backends. Triggers on lookbook architecture questions, "how do I add a scorer", "where does the manifest live", "what's the registry pattern", or any work inside the lookbook repo.
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
- **Phase 2 (embeddings + submodular)** — done. New `embedders/` module
  with `MockEmbedder` (deterministic, no deps), `CLIPEmbedder` and
  `DINOv2Embedder` (lazy-imported via `transformers`, MPS-aware).
  `FacilityLocation` selector — pure-numpy greedy, no apricot dep — plus
  `lookbook.diagnose.cluster_coverage` (sklearn KMeans). New CLI recipes:
  `diverse`, `diverse_clip`, `diverse_mock`. The pipeline now pre-fetches
  embeddings into `constraints["embeddings"]` so the Selector Protocol
  stays unchanged.
- **Phase 3 (person profile)** — done. Person scorers (`MockFaceDetect`,
  `InsightFaceDetect`, `FaceArea`, `MockHeadPose`, `SixDRepNetHeadPose`,
  `FaceQualityProxy`), person filters (`HasFace`, `SingleFaceOnly`,
  `MinFaceArea`, `MinFaceConfidence`), `MockArcFaceEmbedder` +
  `InsightFaceArcFace`, `QuotaSelector` (binning supports dotted paths
  like `head_pose.yaw_bin`), and YAML profile loader at
  `lookbook/profiles/{__init__.py, *.yaml}`. Shipped profiles: `person`
  (real backends) and `person_mock` (no-download). The CLI's `--recipe`
  resolves through in-code RECIPES first, then YAML profiles. New skill:
  `lookbook-profile`.
- **Phase 4 (HTTP via qh)** — done. `lookbook/http.py` exposes 10 verbs
  as uniform `POST /<verb>` endpoints with JSON bodies (no convention-
  based routing). Server-wide stores singleton, lazy-init to the user's
  app data folder; override with `LOOKBOOK_DATA_ROOT`. CLI: `lookbook
  serve --port 8000`. Swagger at `/docs`. New skill: `lookbook-http`.
- **Phase 5 (MCP + usage skills)** — done. `lookbook/mcp.py` exposes
  every HTTP verb as a FastMCP tool. `lookbook mcp` runs over stdio.
  Three new usage skills written for agent callers: `lookbook-curate`,
  `lookbook-diagnose`, `lookbook-recipe`. (The plan named `py2mcp`;
  `fastmcp` is the community-standard substitute.)

### Shipped after the five phases

The five-phase plan is done; these landed on top of it and are the parts
federation callers actually reach for:

- **`local_path()` on every ImageRef** — `PathImageRef` / `BytesImageRef` /
  `UrlImageRef` all answer `local_path(*, cache_dir=None) -> str`, and the
  free function `lookbook.to_local_path(ref)` dispatches across them. Bytes/url
  refs materialize once into a content-addressed cache (honors
  `$LOOKBOOK_REFS_CACHE_DIR`). This is what downstream tools that need a real
  file (ffmpeg, uploads) call.
- **`curate_interactive` + `InteractiveDecision`** (`lookbook/interactive.py`)
  — round-based human-in-the-loop curation. `on_decision` is either a callable
  or a pre-recorded list of decisions (headless replay, tests).
- **`technical_quality` scorer** (`scorers/technical.py:TechnicalQuality`) — a
  derived 0..1 composite of `blur` + `exposure` + `resolution`; the non-face
  counterpart of `face_quality`, giving `top_k` one rankable metric for
  environment / prop / style pools.
- **`curate_for_character` / `curate_for_environment`** (`lookbook/facade.py`)
  — opinionated one-call presets over `curate`, ranking by `face_quality` and
  `technical_quality` respectively. `curate_for_character(face_detector=...)`
  swaps the real `insightface` detector for `mock_face` in tests/demos.
- **Cross-image identity scorer** (`scorers/identity.py`) — `IdentitySimilarity`
  (the `identity_similarity` registry entry), the `compare_to_reference(...)
  -> SimilarityResult` facade, and its HTTP + MCP verb. Answers "does this
  generation still match the locked reference?" with a `[0, 1]` score and an
  *advisory* `passed` flag. The embedder is injected (registry name, `Embedder`
  object, or bare `embed_fn`), so importing the module pulls in no torch /
  insightface and the cosine math is unit-tested with a fake embedder.

Every function and class named above is a top-level `lookbook` export except
`TechnicalQuality`, which is reached as the `technical_quality` registry name
through `registry.scorers`. `compare_to_reference` is additionally an HTTP verb
and an MCP tool; the `curate_for_*` presets are deliberately Python-only.

`lookbook/__init__.py` is now pure re-export — no logic, and no
`from __future__ import annotations`, so `dir(lookbook)` is the public API
plus submodules. New facade functions belong in `lookbook/facade.py`.

## The five-layer architecture

```
Interface (CLI, HTTP via qh, MCP via fastmcp, Python lib)
   ↓
Recipe / facade   (lookbook.facade, profiles)
   ↓
Orchestration    (lookbook.pipeline, manifest, drop attribution, run records)
   ↓
Plugin layer     (Scorer, Filter, Embedder, Selector — Protocols in base.py)
   ↓
Backend          (CLIP, DINOv2, InsightFace, 6DRepNet — wrapped, lazy-imported)
```

**No `import torch` above the backend layer.** Heavy ML deps must be
lazy-imported inside scorer/embedder methods, never at module top. This is
what keeps the package laptop-installable and lets the planning/inspection
layers work without GPU.

## The five protocols are the extension surface

Defined in `lookbook/base.py`. `ImageRef` is the input type; the other four
are the plugin layer:

- `ImageRef` — has `image_id`, `metadata`, `open()`, `bytes()`. Three concrete
  impls in `refs.py`: `PathImageRef`, `BytesImageRef`, `UrlImageRef`.
- `Scorer` — `metric_id`, `cost_tier`, `requires`, `config_hash`, `score(ref, manifest) -> Any`.
- `Filter` — `keep(ref, manifest) -> bool`.
- `Embedder` — `space_id`, `cost_tier`, `embed(ref) -> ndarray`.
- `Selector` — `select(candidates, manifest, k, constraints) -> list[ImageRef]`.

A Selector that needs embeddings does **not** read the manifest or stores
directly — it declares an `embedding_space` attribute, and the pipeline
pre-fetches the corresponding vectors into `constraints["embeddings"]`
before calling `select()`. This keeps the Selector Protocol stable while
giving selectors clean access to set-level data.

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

`lookbook.store.Stores` bundles four MutableMappings (plus the filesystem
`root` they came from, when there is one):

| Store | Keys | Values |
|---|---|---|
| `images` | `image_id` | metadata record, e.g. `{"path": ...}` |
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
- Translates `(image_id, metric_id)` ↔ `"image_id--metric_id.json"` filenames.
  The separator is `lookbook.store._KEY_SEP`, and it is `--` rather than `::`
  because Windows disallows `:` in filenames.
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

The orchestrator is still a hand-rolled topo walk — the planned swap to
`meshed.DAG` (richer dependency graphs, persistent caching) has never been
taken up, and nothing in the package imports `meshed`. When someone does make
that switch: the `Scorer.requires` field is already the dependency graph —
just feed it to meshed.

## Folder & path defaults

Everything goes through `lookbook._paths`, which wraps `config2py`. On POSIX —
macOS included, it does *not* use `~/Library/Application Support` — the XDG
convention applies, and the `$XDG_*_HOME` env vars override it:
- `default_data_root()` → `~/.local/share/lookbook`.
- `default_cache_root()` → `~/.cache/lookbook`, for regeneratable model weights / embeddings.
- `default_config_root()` → `~/.config/lookbook`, for user-edited recipes/profiles.

On Windows the same three funnel into `%LOCALAPPDATA%` / `%APPDATA%` instead;
`config2py` owns that branch, which is why nothing here hardcodes a path.

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
| Write a subject profile (person, product, …) | `lookbook-profile` |
| Add an HTTP endpoint or wire a frontend | `lookbook-http` |
| Help an agent USE lookbook to curate | `lookbook-curate` |
| Interpret a curation result / explain drops | `lookbook-diagnose` |
| Customize recipes / quotas / weights | `lookbook-recipe` |
