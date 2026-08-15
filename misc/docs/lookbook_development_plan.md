# Lookbook — Development Plan

**Companion to:** `lookbook_design_report.md`
**Status:** All five v1 phases shipped, plus the post-v1 work listed below.
**Goal:** A flexible, extensible Python library for image-set curation, with a thin
HTTP surface (FastAPI via `qh`), and a set of Claude Code skills that (a) accelerate
development and (b) let AI agents use lookbook well once it ships.

## Status snapshot

| Phase | What | Status |
|---|---|---|
| 0 | Skeleton + stores | ✅ Done |
| 1 | Cheap funnel (resolution / dedup / blur / exposure / phash) | ✅ Done |
| 2 | Embeddings (CLIP, DINOv2) + facility-location + cluster diagnosis | ✅ Done |
| 3 | Person profile (InsightFace, ArcFace, head pose, quotas, YAML loader) | ✅ Done |
| 4 | HTTP surface via `qh` — 10 verbs | ✅ Done |
| 5 | MCP via `fastmcp` + usage skills — 10 tools | ✅ Done |
| — | `local_path()` on every ImageRef + `to_local_path` | ✅ Shipped post-v1 |
| — | `curate_interactive` (human in the loop) | ✅ Shipped post-v1 |
| — | `technical_quality` scorer + `curate_for_character` / `curate_for_environment` | ✅ Shipped post-v1 |
| — | Cross-image identity scorer + `compare_to_reference` (HTTP + MCP verb) | ✅ Shipped post-v1 |
| — | Curation decisions as lacing annotations (`annot://schema/curation-decision/v1`) | ⬜ Deferred (on-disk format change) |
| — | `meshed.DAG` orchestration (planned in §4) | ⬜ Not taken up |

### Phase 0 deliverables (done)

- `lookbook/{base,refs,manifest,registry,pipeline,store,_paths}.py`
- `dol`-backed `Stores` bundle; default location via `config2py.get_app_data_folder("lookbook")`
- `RandomScore` placeholder + `TopK` selector
- CLI: `lookbook curate` / `list-plugins`
- 15 tests covering protocols, codecs, filesystem persistence, end-to-end curate
- Skills: `lookbook-dev`, `lookbook-storage`

### Phase 1 deliverables (done)

- Scorers: `resolution`, `file_hash`, `phash` (via `imagehash`), `blur`
  (cv2 with numpy fallback), `exposure`
- Filters: `min_resolution`, `min_blur`, `exposure_range`, plus stateful
  `no_exact_duplicate`, `no_near_duplicate` with `fresh_filter()` for state isolation
- `Report` dataclass with drop attribution; integrated into `RunResult`
- Facade `(name, kwargs)` tuple form for tunable plugins
- CLI named recipes: `random`, `funnel`, `funnel_relaxed`
- 18 new tests (33 total, all passing)
- **Performance check:** 200 random 1100×1100 images → 20 kept in 17.9s on
  laptop CPU (target <30s).
- Skills: `lookbook-add-scorer`, `lookbook-add-selector`

### Phase 2 deliverables (done)

- Embedders submodule: `MockEmbedder` (deterministic, no deps), `CLIPEmbedder`,
  `DINOv2Embedder` (both via `transformers`, lazy-imported, MPS-aware)
- Pipeline orchestration extended:
  - Embedders run after scorers; vectors stored in `stores.embeddings[space_id]`
  - Manifest gets a `emb:<space_id>` presence annotation for cache lookups
  - Selector pre-fetch: pipeline reads `selector.embedding_space` and passes
    `constraints["embeddings"]` as `dict[image_id, vector]`
  - Selector Protocol *unchanged* — embedding access is a constraint key,
    not a signature change
- `FacilityLocation` selector — pure-numpy greedy implementation (no apricot
  dep). Objective: `weight_diversity * facility-location + weight_quality *
  per-image quality`. Cosine similarity clipped to [0, 1] so the empty-set
  baseline is well-defined.
- `lookbook.diagnose.cluster_coverage` — sklearn KMeans, written into the
  Report's `notes` so users see "you filled 9/12 visual clusters."
- New CLI recipes: `diverse` (DINOv2), `diverse_clip`, `diverse_mock`
- Defensive `USE_TF=0` / `USE_FLAX=0` env vars in `embedders/__init__.py`
  to side-step transformers' optional-backend autodetection
- 16 new tests (49 total) plus 2 opt-in real-model smokes gated by
  `LOOKBOOK_TEST_MODELS=1`
- **Real-model smoke check:** CLIP ViT-B/32 + DINOv2 base both load and
  produce L2-normalized 512-D / 768-D vectors via the full pipeline; 9
  images → 3 kept with cluster coverage in <10s after weights are cached
- Skill update: `lookbook-dev` now lists the embedders module and Phase 2
  status; the existing `lookbook-add-selector` already covers the
  Selector Protocol contract (embedding access via `constraints`)

### Phase 3 deliverables (done)

- Person scorers (`lookbook/scorers/person.py`):
  - `MockFaceDetect`, `InsightFaceDetect` — both write the same
    `face_box` annotation, distinguished by `config_hash`
  - `FaceArea` — derived from `face_box` + `resolution`, with size bins
  - `MockHeadPose`, `SixDRepNetHeadPose` — yaw/pitch/roll + bin labels
  - `FaceQualityProxy` — confidence + area + sharpness; stand-in for FIQA
- Person filters (`lookbook/filters/person.py`):
  - `HasFace`, `SingleFaceOnly`, `MinFaceArea`, `MinFaceConfidence`
- ArcFace embedder (`lookbook/embedders/arcface.py`):
  - `MockArcFaceEmbedder` (deterministic, no deps)
  - `InsightFaceArcFace` (lazy-imported)
- `QuotaSelector` (`lookbook/selectors/quota.py`):
  - Bins candidates by a manifest annotation (supports dotted paths
    like `head_pose.yaw_bin`)
  - Wraps any registered selector as the inner; respects per-bin quotas
    from `constraints["quotas"]`
  - Backfills from the global pool when bins under-fill (or strict mode)
- YAML profiles (`lookbook/profiles/`):
  - `lookbook.profiles.load(name)` — loads `lookbook/profiles/*.yaml`
    or `<user_config>/profiles/*.yaml`; user wins
  - Shipped: `person.yaml` (real backends), `person_mock.yaml` (mocks)
  - `pyyaml` added to runtime deps; YAMLs force-included in the wheel
- CLI: `lookbook curate --recipe person`/`person_mock` (recipes resolve
  through in-code RECIPES first, then YAML profiles)
- 26 new tests (75 total, all passing); 2 opt-in real-model smokes
- New skill: `lookbook-profile` — the recipe for adding subject profiles
  (product, scene, style, brand, font, …)

### Phase 4 deliverables (done)

- `lookbook/http.py` — JSON-friendly HTTP surface built with `qh.mk_app`:
  - `list_recipes`, `list_plugins`, `ingest_source`, `curate_source`,
    `score_image`, `get_annotations`, `list_runs`, `get_run`, `get_image`
    (a 10th verb, `compare_to_reference`, landed post-v1 — see below)
  - Uniform `POST /<verb>` routes with JSON bodies (no convention-based
    routing — keeps the API consistent for agent clients and the future
    MCP layer)
  - `mk_lookbook_app()` builder; `serve(host, port)` runs uvicorn
- Server-wide stores singleton, lazy-initialized to the user's app data
  folder. Override with `LOOKBOOK_DATA_ROOT=...` env var. `reset_stores()`
  swaps it out for tests
- `ingest_to_store()` — writes `image_id -> {"path": ...}` records into
  `stores.images` so `get_image` can serve bytes by id. The `images` slot
  default switched from `Files` (bytes-only) to `JsonFiles` (metadata
  records); zero callers were affected
- CLI: `lookbook serve --port 8000 --host 127.0.0.1`
- 14 new tests via `qh.testing.run_app` TestClient (89 total, all passing).
  Live server smoke-tested with curl: `POST /list_recipes` and
  `POST /list_plugins` returned correct JSON over HTTP/1.1
- `[http]` extras: `qh`, `fastapi`, `uvicorn` (already declared in Phase 0)

### Phase 5 deliverables (done)

- `lookbook/mcp.py` — MCP surface via `fastmcp`. Each verb wraps the
  matching `lookbook.http` route function (same JSON-shaped args /
  returns), exposed as a FastMCP tool with an LLM-tuned description
- 9 tools: `list_recipes`, `list_plugins`, `ingest_source`,
  `curate_source`, `score_image`, `get_annotations`, `list_runs`,
  `get_run`, `get_image` (a 10th, `compare_to_reference`, landed
  post-v1 — see below)
- Note on substitution: the original plan named `py2mcp` as the MCP
  framework. `py2mcp` isn't on this user's machine and `fastmcp` is the
  community-standard choice; the protocol is identical, so the
  substitution is invisible to clients.
- `mk_lookbook_mcp()` builder; `serve(transport="stdio")` is the
  Claude-Desktop-compatible default
- CLI: `lookbook mcp` (stdio); the same Stores singleton as `lookbook.http`
  so HTTP and MCP servers backed by the same data folder share the
  manifest and runs index
- 12 tests via `fastmcp.Client(server)` in-memory transport (101 total,
  all passing). Tool descriptions are sanity-checked for length to
  catch regressions where the LLM-facing description gets truncated
- `[mcp]` extras: `fastmcp`
- Three new usage skills (the agent-facing ones):
  - `lookbook-curate` — the headline curation tool for agents
  - `lookbook-diagnose` — interpreting reports, querying the manifest
  - `lookbook-recipe` — per-call overrides + user YAML profiles

### Post-v1 deliverables (done)

Shipped after the five phases closed, each driven by a federation consumer:

- **`ImageRef.local_path()` + `lookbook.to_local_path(ref)`** — every ref
  type answers "give me a real file on disk"; bytes/url refs materialize
  once into a content-addressed cache (`$LOOKBOOK_REFS_CACHE_DIR`). What
  downstream tools that shell out to ffmpeg or upload files need.
- **`curate_interactive` + `InteractiveDecision`** (`lookbook/interactive.py`)
  — round-based human-in-the-loop curation; `on_decision` is a callable or a
  pre-recorded list of decisions (headless replay). `muvid` drives this.
- **`technical_quality` scorer** — derived 0..1 composite of `blur` +
  `exposure` + `resolution`; the non-face counterpart of `face_quality`,
  giving `top_k` one rankable metric for environment / prop / style pools.
- **`curate_for_character` / `curate_for_environment`** (`lookbook/facade.py`)
  — opinionated `k=1`-by-default presets over `curate`, ranking by
  `face_quality` and `technical_quality` respectively. Python-only; they are
  deliberately *not* HTTP/MCP verbs.
- **Cross-image identity scorer** (`lookbook/scorers/identity.py`) —
  `IdentitySimilarity` (registered as `identity_similarity`),
  `compare_to_reference(...) -> SimilarityResult`, and the matching HTTP verb
  + MCP tool. The measurement substrate for a reference supervisor: "does
  this generation still match the locked reference?" as a `[0, 1]` score with
  an advisory `passed` flag. The embedder is injected, so the module imports
  without torch and the cosine math is unit-tested offline with a fake
  embedder.
- `lookbook/__init__.py` reduced to pure re-exports; the `curate*` / `score`
  entry points live in `lookbook/facade.py`.

## All shipped skills (9 total)

**Dev skills** (used while building / extending lookbook):
- `lookbook-dev` — overall architecture, cross-references
- `lookbook-storage` — repository pattern, dol stores, codecs
- `lookbook-add-scorer` — adding a new per-image metric
- `lookbook-add-selector` — adding a new set-selection algorithm
- `lookbook-profile` — adding a subject profile (YAML)
- `lookbook-http` — HTTP route layout + adding endpoints

**Usage skills** (used by agents driving lookbook to curate):
- `lookbook-curate` — the headline workflow
- `lookbook-diagnose` — making sense of run reports
- `lookbook-recipe` — customizing recipes per-call or via user YAML

---

**Everything below this line is the original plan of record**, kept for the
rationale. Where it disagrees with the status sections above, the status
sections win — notably: the MCP layer is `fastmcp`, not `py2mcp`; the usage
skill shipped as `lookbook-curate`, not `lookbook-use`; and the orchestrator
is still a hand-rolled topo walk, not `meshed.DAG`.

## 0. Guiding Principles

Three non-negotiable architectural commitments that shape every decision below:

1. **Open-closed by Protocols, not classes.** The extension surface is a tiny set
   of `typing.Protocol`s — `ImageRef`, `Scorer`, `Filter`, `Embedder`, `Selector`,
   `Reporter`. New metrics, models, selectors are *registered*, never *subclassed*.
2. **The manifest is the SSOT, the manifest is a `dol` store.** Every annotation
   ever computed for any image lives in a `MutableMapping`. Persistence is pluggable
   (filesystem, SQLite, S3, Mongo) by swapping the underlying `dol` store. The
   default is the user's app data folder via `config2py`.
3. **No `import torch` above the backend layer.** The orchestration, manifest,
   selection, and interface layers must work on a laptop with zero ML deps. Heavy
   models live behind facades and are imported lazily inside backend modules.

Everything that follows is a consequence of these three.

---

## 1. Repository Pattern via `dol`

This is the single most important design choice and it cuts across every layer.

### 1.1 Three stores, one shape

Lookbook persists three kinds of state. All three are `MutableMapping`s:

| Store | Keys | Values | Default backing |
|---|---|---|---|
| **`images`** | `image_id` (str) | `ImageRef`-compatible bytes/path/url record | `Files` over `<data>/images/` (or symlink farm) |
| **`manifest`** | `(image_id, metric_id)` | `Annotation` dataclass | JSON-per-image in `<data>/manifest/<image_id>.json`, or SQLite |
| **`runs`** | `run_id` (str) | run record (recipe, kept set, report) | JSON in `<data>/runs/` |

Internally, each is built from `dol` primitives:
- `Files` / `JsonFiles` for filesystem (default)
- `wrap_kvs` / `filt_iter` / `cached_keys` for transformations
- A user can substitute `mongodol`, `s3dol`, `sqldol`, `redisdol`, etc. without
  any other code change. This is the whole point of the pattern.

### 1.2 Default location via `config2py`

```python
# lookbook/_paths.py
from config2py import get_app_data_folder

def default_data_root() -> str:
    return get_app_data_folder("lookbook", folder_kind="data", ensure_exists=True)
```

Additional folders the package will set up the same way:
- `cache/`  →  `folder_kind="cache"` for model weights, intermediate embeddings
- `config/` →  `folder_kind="config"` for user-edited recipes & profiles
- `state/`  →  `folder_kind="state"` for run logs

Layout under the data root:
```
<data>/
  images/          # ingested image content (or refs)
  manifest/        # JSON-per-image annotations
  runs/            # per-run records
  embeddings/      # vector index (CLIP, DINOv2, ArcFace), one shard per space
```

### 1.3 The store-factory pattern

A single function returns a fully-wired set of stores. Users override pieces:

```python
# lookbook/store.py
@dataclass
class Stores:
    images:    MutableMapping
    manifest:  MutableMapping
    runs:      MutableMapping
    embeddings: Mapping[str, MutableMapping]   # space_id → vector store

def get_stores(
    *,
    root: str | None = None,
    images_store: MutableMapping | None = None,
    manifest_store: MutableMapping | None = None,
    runs_store: MutableMapping | None = None,
    embeddings_factory: Callable[[str], MutableMapping] | None = None,
) -> Stores: ...
```

Test code passes `dict()` everywhere. Production code passes nothing and gets the
local defaults. A hosted deployment swaps in S3/Mongo/Redis-backed stores.

---

## 2. Package Layout

```
lookbook/
  __init__.py                # facade: curate, score, score_set, diagnose, export
  base.py                    # Protocols + Annotation + Manifest type
  _paths.py                  # config2py-backed default folders
  store.py                   # Stores dataclass, get_stores()
  refs.py                    # ImageRef impls (path, url, bytes, in-memory)
  manifest.py                # Manifest helpers (CRUD, queries, provenance)
  pipeline.py                # Orchestrator (built on meshed)
  budget.py                  # Cost accounting, dynamic skip
  registry.py                # Plugin registry: scorers, filters, embedders, selectors
  scorers/                   # All implement Scorer Protocol
    technical.py             # Phase 1: blur, exposure, dedup, resolution
    aesthetic.py             # Phase 2: LAION, NIMA, MUSIQ, TOPIQ, CLIP-IQA
    embeddings.py            # Phase 2: CLIP, DINOv2, ArcFace
    person.py                # Phase 3: face detect, FIQA, head pose, identity
    object.py                # Phase 4+
    scene.py                 # Phase 4+
  selectors/
    threshold.py             # Phase 1
    topk.py                  # Phase 1
    submodular.py            # Phase 2: apricot wrapper
    constrained.py           # Phase 3: quota-aware
    dpp.py                   # Phase 4: DPPy
  profiles/
    person.yaml              # Phase 3
    product.yaml             # Phase 4
    scene.yaml               # Phase 4
    style.yaml               # Phase 4
  io/
    ingest.py                # dir, zip, url-list, cloud
    export.py                # kohya-style folders
  report.py                  # Diagnose / human-readable report
  __main__.py                # argh CLI dispatch
  http.py                    # qh / FastAPI surface (Phase 4)
  mcp.py                     # py2mcp surface (Phase 5)
tests/                       # pytest, top-level
misc/docs/                   # design + plan docs
.claude/
  skills/                    # see §6
```

A few conventions worth flagging:
- `base.py` holds Protocols and dataclasses only; zero imports of heavy ML libs.
- `scorers/`, `selectors/` modules import their heavy deps **inside the function
  body**, not at module top, so a `from lookbook.scorers import technical` does
  not pull in torch. The registry uses entry-point-style late binding.
- Profiles are YAML files shipped with the package via `importlib.resources`,
  copied into `<config>/profiles/` on first run so users can edit them. (This
  is the seed-on-missing pattern — `user-data-folder` skill applies.)

---

## 3. Core Type Sketches

These are concrete enough to start coding from but small enough to stay flexible.

```python
# lookbook/base.py
from typing import Protocol, runtime_checkable, Any, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime

@runtime_checkable
class ImageRef(Protocol):
    image_id: str
    metadata: Mapping[str, Any]
    def open(self) -> "PIL.Image.Image": ...
    def bytes(self) -> bytes: ...

@dataclass(frozen=True)
class Annotation:
    image_id: str
    metric_id: str
    value: Any                  # number | vector | dict | label
    config_hash: str            # which model / params produced this
    cost_tier: int
    timestamp: datetime
    backend: str = ""           # provenance: e.g. "pyiqa:musiq", "insightface:arcface"

# Manifest is the canonical type for "everything we know about every image".
# It is a MutableMapping[(image_id, metric_id) -> Annotation].
# In code, helpers below provide the ergonomics.

class Scorer(Protocol):
    metric_id: str
    cost_tier: int
    requires: tuple[str, ...] = ()
    config_hash: str
    def score(self, ref: ImageRef, manifest: "Manifest") -> Any: ...

class Filter(Protocol):
    def keep(self, ref: ImageRef, manifest: "Manifest") -> bool: ...

class Embedder(Protocol):
    space_id: str
    cost_tier: int
    def embed(self, ref: ImageRef) -> "np.ndarray": ...

class Selector(Protocol):
    def select(
        self,
        candidates: Iterable[ImageRef],
        manifest: "Manifest",
        k: int,
        constraints: Mapping[str, Any] = (),
    ) -> list[ImageRef]: ...
```

Two design notes worth dwelling on:

- **Manifest is a `MutableMapping`, not a class.** Anywhere we annotate a
  parameter `manifest: Manifest`, we mean "any mapping with that key/value
  shape." This is what lets the same code run against `dict`, `JsonFiles`,
  `mongodol`, etc.
- **`config_hash` is the cache key.** Re-running with the same scorer config
  re-uses the cached annotation; changing a threshold or model invalidates.
  The Scorer is responsible for computing its own hash from its config.

---

## 4. Orchestration via `meshed`

A `Pipeline` is a list of stages plus a final selector. Each stage is a
`Scorer | Filter | Embedder`. The orchestrator:

1. Builds a DAG from each Scorer's `requires` field (e.g. `fiqa` requires
   `face_box`).
2. Walks the DAG in cost-tier order, gating on filters between tiers.
3. Writes every Scorer/Embedder output to the manifest via the standard SSOT.
4. Hands the surviving candidates plus the read-only manifest to the Selector.

`meshed` does the wiring; `lookbook.pipeline` is a thin layer that knows about
cost tiers, the manifest, and the budget.

```python
# Conceptual usage
from lookbook import Pipeline, profiles

pipeline = Pipeline.from_profile(profiles.person)
result = pipeline.run(images, k=20, stores=stores, budget="2min")
# result has: kept (list[ImageRef]), manifest (Manifest), report (Report)
```

---

## 5. Phasing — Build Order

Five phases. Each phase keeps the package shippable.

### Phase 0 — Skeleton + Stores (~1 week)

- `pyproject.toml` deps: `Pillow`, `numpy`, `dol`, `meshed`, `config2py`,
  `argh`, `pydantic` (no torch yet).
- Implement: `base.py`, `_paths.py`, `store.py`, `refs.py`, `manifest.py`,
  `registry.py`, `pipeline.py`, `__main__.py`.
- One trivial scorer (`random_score`) and one trivial selector (`top_k`) so the
  pipeline runs end-to-end on 10 test images.
- Tests for the store layer using `dict()` substitutes.
- **Deliverable:** `lookbook curate ./photos --k 5` returns 5 images with
  random scores. Manifest persists between runs to the user data folder.
- **Skill output:** establish `.claude/skills/lookbook-dev/` (see §6).

### Phase 1 — Cheap Funnel (1–2 weeks)

- T0/T1 scorers: resolution, file hash, perceptual hash (`imagededup`),
  blur (Variance of Laplacian), exposure histogram.
- Threshold and top-K selectors with sensible defaults.
- A first real `Report` (kept N, dropped M, reasons by counts).
- **Deliverable:** "clean my photo dump" works on 200 random images in <30s
  on a laptop, no GPU, no AI deps. Already a useful tool for non-LoRA users.

### Phase 2 — Embeddings + Submodular (~2 weeks)

- Add `torch`, `open_clip_torch`, `transformers`, `apricot-select` as optional
  deps under `[project.optional-dependencies] embed`.
- Scorers: CLIP and DINOv2 embedders, LAION-Aesthetic-V2, one `pyiqa` head
  (MUSIQ or TOPIQ).
- Selector: `apricot`-based facility-location.
- Diagnose: cluster-coverage report ("you filled 9/12 visual clusters").
- **Deliverable:** the headline 200→20 generic workflow, subject-agnostic.

### Phase 3 — Person Profile (~2–3 weeks)

- Optional dep group `[person]`: `insightface`, `sixdrepnet`, `mediapipe`.
- Scorers: face detection/count, ArcFace identity, head pose, FIQA wrapper
  (CR-FIQA initially).
- Constrained / quota selector for pose bins.
- `profiles/person.yaml` with the recipe from design report §4.2.
- **Deliverable:** `lookbook curate ./photos --profile person --k 20` end-to-end.
  This is the version that pulls ahead of every existing tool.

### Phase 4 — HTTP Surface via `qh` (~1 week)

- Define a `lookbook.http` module that wraps the eight verbs (Ingest, Probe,
  Score, Filter, Embed, Select, Diagnose, Export) plus three resource verbs
  (`list_runs`, `get_run`, `get_image`) into a `qh.mk_app` tree.
- Every HTTP route is a thin wrapper around the same Python function — no
  duplicated logic, just dispatch.
- Returns JSON manifests; image bytes via a separate route.
- **Deliverable:** `lookbook serve --port 8000` starts a FastAPI app. A simple
  HTML/JS frontend (later, separate concern) can call it.
- A minimal browser UI is *not* part of this plan — that's a separate
  follow-up project. The HTTP layer is designed so the frontend is
  uncontroversial whenever someone builds it.

### Phase 5 — Agent Surfaces (~1 week)

- `lookbook.mcp` via `py2mcp`: each verb becomes an MCP tool.
- Profile templates: product, scene, style.
- `.claude/skills/lookbook-use/` skill (see §6) for end users / agents.
- **Deliverable:** an LLM agent can call `score_one`, `select`, `diagnose`
  iteratively on a candidate pool.

### Non-goals (for v1)

Lifted from design report §7.4: no GUI, no captioning, no LoRA training, no
reimplementation of dedup/IQA/clustering. Wrap, don't rebuild.

---

## 6. Claude Code Skills

Two categories. Development skills accelerate the build; usage skills help
agents (and the user, in Claude Code) drive the package once it ships. Both
live under `.claude/skills/` in the repo so they travel with the project.

### 6.1 Development skills (used during the build)

#### `lookbook-dev` — overall project knowledge
Triggers: working in `/lookbook`, asking "how do I add a scorer", "where does
manifest live", "what's the registry pattern".
Contents:
- The five-layer architecture (interface → recipe → orchestration → plugin → backend).
- How `dol` stores wire up; how to swap them for tests vs production.
- Where `config2py` lives in the path resolution.
- The Scorer/Filter/Embedder/Selector Protocols with concrete examples.
- The `requires`/`config_hash`/`cost_tier` contract a Scorer must honor.
- A worked example: "adding a new scorer end to end" (file location, registry
  registration, test).

#### `lookbook-add-scorer` — one-screen recipe
Triggers: "add a scorer for X", "wrap pyiqa model Y", "add a new metric".
Contents:
- The 5-step recipe: (1) decide cost tier, (2) declare `requires`, (3) compute
  `config_hash`, (4) write the function, (5) register and test.
- Templates for T0, T1, T2, T3 scorers.
- How to wrap a `pyiqa` model in 12 lines.
- Cache invalidation rules.

#### `lookbook-add-selector` — same shape for selectors
Triggers: "add a selector", "wrap a new submodular function", "add quota selector".
Contents:
- The Selector Protocol expectations.
- How to read from manifest without mutating it.
- How to add constraints (quotas, must-include, must-exclude).
- When to prefer `apricot` vs writing a greedy by hand vs DPPy.

#### `lookbook-storage` — repository pattern reference
Triggers: "swap to S3 storage", "the manifest store", "test with in-memory store",
"persistence for runs".
Contents:
- The three stores (images, manifest, runs) and their key shape.
- How to swap each one. Where `config2py` is involved.
- How `wrap_kvs` is used to add codecs (e.g. JSON codec on values).
- Common pitfalls: key collisions, codec round-tripping, embedded vector stores.

#### `lookbook-profile` — adding a subject profile
Triggers: "add product profile", "support style LoRA", "new subject type".
Contents:
- Anatomy of a YAML profile (default scorers, selector, weights, quotas).
- The seed-on-missing pattern that copies the YAML to user config on first run.
- The minimum metric set per subject family (person, product, scene, style).

### 6.2 Usage skills (used after lookbook ships)

#### `lookbook-curate` — the headline tool, for agents
Triggers: "curate these images", "pick K best for LoRA training", "dataset for
character LoRA", "select images for training".
Contents:
- The eight verbs and when to call each.
- The standard recipe per subject type.
- How to read a Report and act on its diagnose output.
- The "don't recompute" principle: re-running with a different selector but
  same scorers is free.
- When to call the MCP tools individually vs the high-level `curate` facade.

#### `lookbook-diagnose` — making sense of a curation result
Triggers: "why did this image get rejected", "what's missing from my set",
"explain the curation report".
Contents:
- How to query the manifest by image_id to get the full annotation history.
- Reading coverage charts (which pose bins are empty, which clusters underfilled).
- Common failure modes: identity drift, redundancy collapse, low-FIQA bias.

#### `lookbook-recipe` — writing a custom recipe
Triggers: "I want different defaults", "custom scorer weights", "skip aesthetic
scoring", "build my own pipeline".
Contents:
- How to override profile defaults from the CLI / Python / HTTP.
- The "what-if reweighting" pattern: cache scores once, sweep weights.
- Recipe portability (the YAML is the spec).

### 6.3 Skill priorities

The build sequence drives skill creation:

| Phase | Dev skills written | Usage skills written |
|---|---|---|
| 0 | `lookbook-dev`, `lookbook-storage` | — |
| 1 | `lookbook-add-scorer` | — |
| 2 | `lookbook-add-selector` | — |
| 3 | `lookbook-profile` | (drafts) |
| 4 | — | `lookbook-curate`, `lookbook-recipe` |
| 5 | — | `lookbook-diagnose` |

Each skill is created/updated using the `skill-creator` skill, kept in sync
with code via `skill-sync`, and packaged for distribution via `skill-build` /
`skill-enable` so they ship with the pip-installable wheel.

---

## 7. HTTP Interface (FastAPI via `qh`)

`qh` builds a FastAPI app from a tree of Python functions. The lookbook HTTP
surface is a one-file module (`lookbook/http.py`) that:

1. Imports the eight verbs as plain functions from the facade.
2. Adds three resource endpoints: `GET /images/{id}`, `GET /runs`, `GET /runs/{id}`.
3. Wraps everything with `qh.mk_app`.

A sample call shape:

```
POST /curate          { "source": "...", "profile": "person", "k": 20 }
                    →  { "run_id": "...", "kept": [...], "report_url": "..." }

POST /score           { "image_id": "...", "metric_id": "blur" }
                    →  { "value": 137.2, "config_hash": "..." }

POST /select          { "candidates": [...], "k": 20, "selector": "facility_location" }
                    →  { "kept": [...] }

GET  /diagnose/{run}  →  { "missing": ["yaw>30"], "overrepresented": [...] }
```

Key design points:
- The HTTP layer **is generated from the Python functions**, not parallel to
  them. No business logic in HTTP code.
- A simple browser frontend is a follow-up; this plan does not block on it.
- For long-running curation jobs, `qh.AuTaskStore`/`AuTaskExecutor` provides
  async task semantics out of the box. Phase 4 should wire this in.

---

## 8. Open Questions Worth Aligning On Before Coding

These are decisions where I want a quick yes/no/redirect before locking in:

1. **Default storage codec for the manifest.** JSON-per-image is simple and
   git-friendly. SQLite is faster for queries. I'd start with JSON-per-image
   (one file under `<data>/manifest/`) and add SQLite later via a `dol` codec
   swap. OK?
2. **`pyiqa` as a Phase 2 dependency or Phase 3.** It pulls torch. If we want
   Phase 1 to stay laptop-friendly we keep `pyiqa` in an `[embed]` extra.
   Confirming the split is acceptable.
3. **Should the HTTP surface live in this repo or a sibling?** I have it inline
   here but it's a clean cut to put it in `lookbook-http` if you prefer the
   core pure-Python.
4. **Default user-app-folder name.** `get_app_data_folder("lookbook")` →
   `~/Library/Application Support/lookbook` on macOS. Confirming the app name.
5. **License posture.** Design report §6.2 flags AGPL deps (CleanVision,
   Ultralytics). Plan above sticks to MIT/Apache-2.0; confirm we want to keep
   that posture even if it costs us CleanVision integration.

I will not start any code beyond reading until we agree on at least
questions 1, 2, and 5.

---

## 9. First Concrete Tickets (Phase 0)

The literal first PRs, in order:

1. **`lookbook/base.py`** — Protocols, `Annotation`, type aliases. ~80 lines.
2. **`lookbook/_paths.py`** — `default_data_root`, `default_cache_root`,
   `default_config_root`. ~30 lines, all `config2py`.
3. **`lookbook/store.py`** — `Stores` dataclass, `get_stores()` factory.
   Uses `dol.Files` and `dol.JsonFiles` over the paths from `_paths.py`. ~120 lines.
4. **`lookbook/refs.py`** — `PathImageRef`, `BytesImageRef`, `UrlImageRef`. ~60 lines.
5. **`lookbook/manifest.py`** — Helpers: `get_annotation`, `put_annotation`,
   `iter_annotations_for`, `image_ids`. ~80 lines.
6. **`lookbook/registry.py`** — In-memory registry of scorers/filters/embedders/
   selectors keyed by id. ~40 lines.
7. **`lookbook/pipeline.py`** — Pipeline class wired with `meshed`. ~150 lines.
8. **`lookbook/scorers/technical.py`** — Just `random_score` for Phase 0; real
   scorers land in Phase 1.
9. **`lookbook/__init__.py`** — Facade re-exporting `curate`, `score`,
   `score_set`, `diagnose`, `Pipeline`, `Stores`.
10. **`lookbook/__main__.py`** — `argh` dispatch over the facade.
11. **Tests** — One end-to-end test that runs `curate` with `random_score` and
    `top_k` against a `dict()` store, plus targeted unit tests per module.
12. **`.claude/skills/lookbook-dev/SKILL.md`** — written alongside the code.
13. **`.claude/skills/lookbook-storage/SKILL.md`** — written alongside `store.py`.

Each ticket is small enough to land in a single, reviewable commit. The whole
of Phase 0 should be ~700 lines of code plus tests.
