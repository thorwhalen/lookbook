# lookbook

Distill raw image pools into optimized, high-diversity reference sets for
training personalized models (character LoRAs, product LoRAs, style LoRAs).

> **Status:** All five v1 phases are shipped. The package can: clean a
> photo dump (200→20 in <30s, no GPU); run the full embeddings +
> facility-location workflow with CLIP or DINOv2; curate a person /
> character LoRA with face detection, ArcFace identity diversity, and
> pose-bin quotas; serve every verb over HTTP (FastAPI via `qh`); and
> expose every verb over MCP (via `fastmcp`) so an LLM agent can drive
> curation directly. See
> [`misc/docs/lookbook_development_plan.md`](misc/docs/lookbook_development_plan.md)
> for the full roadmap and [`misc/docs/lookbook_design_report.md`](misc/docs/lookbook_design_report.md)
> for the design rationale.

## Why

Training a personalized image model needs ~15–30 carefully chosen reference
images, where each adds new context — pose, lighting, expression, distance.
"Top-K by score" collapses to near-duplicates; this is fundamentally a
*set-selection* problem with diversity as a constraint, not a tiebreaker.

`lookbook` separates **per-image scoring** (is this image individually good?)
from **set-level selection** (does this collection cover the concept?), and
makes both extensible.

## Install

```bash
pip install lookbook                  # core: Pillow, numpy, dol, meshed, config2py
pip install lookbook[funnel]          # + cv2 / imagededup for the cheap funnel
pip install lookbook[embed]           # + torch, CLIP, DINOv2, pyiqa, apricot
pip install lookbook[person]          # + InsightFace, head pose, mediapipe
pip install lookbook[http]            # + FastAPI / qh server
pip install lookbook[mcp]             # + fastmcp (Anthropic MCP server)
```

The base install has no ML dependencies — Phase 1 (cheap funnel) works on a
plain laptop. `[embed]` and beyond pull torch.

## Quickstart

CLI:

```bash
# Phase 1: clean up a photo dump (drops blurry, dark, duplicate, tiny).
lookbook curate ./photos --k 20 --recipe funnel

# Phase 2: same funnel + DINOv2 embeddings + facility-location selection
# for "diverse but sharp" picks. Downloads ~350MB on first run; subsequent
# runs are fast and cached. Use --recipe diverse_clip for CLIP semantic
# embeddings instead.
lookbook curate ./photos --k 20 --recipe diverse

# Phase 3: full character/person LoRA curation. Detects faces, embeds
# with ArcFace, applies pose-bin quotas, and runs cluster-coverage
# diagnosis. Pulls insightface + sixdrepnet (lazy on first use).
lookbook curate ./photos --k 20 --recipe person

# Same shape with no model downloads (mock backends; useful for tests):
lookbook curate ./photos --k 8 --recipe person_mock

# Phase 4: HTTP server. Every verb is POST /<verb> with a JSON body;
# Swagger UI at /docs.
lookbook serve --port 8000 --host 127.0.0.1
# curl examples:
#   curl -X POST localhost:8000/list_recipes -H 'Content-Type: application/json' -d '{}'
#   curl -X POST localhost:8000/curate_source -H 'Content-Type: application/json' \
#        -d '{"source_path":"/abs/photos","k":20,"recipe":"funnel"}'

# Phase 5: MCP server (stdio transport — for Claude Desktop, Anthropic SDK).
# Each verb is exposed as an MCP tool an agent can call.
lookbook mcp

# See available scorers, embedders, filters, selectors, recipes / profiles:
lookbook list-plugins
lookbook list-recipes
```

Python:

```python
from lookbook import curate

result = curate(
    "./photos",
    k=20,
    scorer_ids=("resolution", "file_hash", "phash", "blur", "exposure"),
    filter_ids=(
        ("min_resolution", {"min_long_side": 800}),
        "exposure_range",
        "min_blur",
        "no_exact_duplicate",
        "no_near_duplicate",
    ),
    selector_id=("top_k", {"metric_id": "blur"}),
)
print(result.report)        # drop counts attributed to each filter
print([r.image_id for r in result.kept])
```

### Interactive curate (keep the human in the loop)

Pipeline-only top-k can rank "boring but sharp" above "stylistically
perfect but slightly blurry". `curate_interactive` invites the caller
into the loop, one round at a time:

```python
from lookbook import curate_interactive, InteractiveDecision

def decide(presented, info):
    # show `presented` to the user; collect their picks
    return InteractiveDecision(keep=("img-abc",), reject=("img-xyz",))

result = curate_interactive(
    "./photos", on_decision=decide, k=20, present=8,
    scorer_ids=("blur", "exposure"),
)
```

Pre-recorded decisions (a list of `InteractiveDecision`) are accepted in
place of the callable — handy for tests, headless replays, and
agent-scripted flows.

### `local_path()` on every ImageRef

`PathImageRef`, `BytesImageRef`, and `UrlImageRef` all expose
`local_path(cache_dir=None) -> str`. Path refs return their existing
path; bytes/url refs materialize once into a content-addressed cache
(honors `$LOOKBOOK_REFS_CACHE_DIR`). The free function
`lookbook.to_local_path(ref)` dispatches across all subtypes — useful
when downstream tools (ffmpeg, fal upload) need a real file.

## Architecture

Five layers; the heavy ML libs live only at the bottom so the upper layers
stay laptop-installable.

```
Interface       (CLI, HTTP via qh, MCP via py2mcp, Python lib)
   ↓
Recipe / facade (lookbook.curate, named recipes, profiles)
   ↓
Orchestration  (lookbook.pipeline, manifest, drop attribution, run records)
   ↓
Plugin layer   (Scorer | Filter | Embedder | Selector — Protocols)
   ↓
Backend        (CLIP, DINOv2, InsightFace, pyiqa, apricot — wrapped, lazy-imported)
```

The **manifest** — `MutableMapping[(image_id, metric_id), Annotation]` — is
the SSOT. Persistence is pluggable via `dol`: filesystem (default), SQLite,
S3, Mongo, Redis. The default location is the user's app data folder via
`config2py` (`~/Library/Application Support/lookbook` on macOS).

New scorers/selectors/filters are *registered*, never subclassed. See the
`.claude/skills/` directory for developer skills (`lookbook-dev`,
`lookbook-add-scorer`, `lookbook-add-selector`, `lookbook-storage`).

## Project layout

```
lookbook/
  base.py               Protocols + Annotation + Manifest type
  store.py              dol-backed Stores bundle, manifest codec
  _paths.py             config2py-backed default folders
  refs.py               PathImageRef, BytesImageRef, UrlImageRef
  manifest.py           Manifest helpers
  registry.py           Plugin registries
  pipeline.py           Orchestrator (topo-sorted scorers + filters + selector)
  report.py             Drop-attributing Report
  scorers/              random, resolution, file_hash, phash, blur, exposure
                          + person.py: face detection, area, head pose, quality
  embedders/            mock, clip, dinov2, arcface (lazy-imported)
  filters/              min_resolution, min_blur, exposure_range, dedup
                          + person.py: has_face, single_face_only, min_face_*
  selectors/            top_k, facility_location, quota
  profiles/             person.yaml, person_mock.yaml (+ user-edited via config2py)
  diagnose.py           cluster_coverage (set-level diagnosis)
  io/                   ingest, ingest_to_store
  http.py               qh-built FastAPI surface; mk_lookbook_app, serve
  mcp.py                fastmcp-built MCP server; mk_lookbook_mcp, serve
  __main__.py           CLI (argh): curate, list-plugins, list-recipes

.claude/skills/         Claude Code skills for development & agent use
misc/docs/              Design report + development plan
tests/                  pytest, hermetic
```

## Claude Code skills

The `.claude/skills/` directory ships nine skills covering both ends of
the workflow:

**Dev skills** (used while building / extending lookbook):
- `lookbook-dev` — overall architecture, cross-references between modules
- `lookbook-storage` — repository pattern, swapping storage backends
- `lookbook-add-scorer` — adding a new per-image metric
- `lookbook-add-selector` — adding a new set-selection algorithm
- `lookbook-profile` — adding a subject profile (YAML)
- `lookbook-http` — HTTP route layout + adding endpoints

**Usage skills** (for agents driving lookbook to curate):
- `lookbook-curate` — the headline workflow
- `lookbook-diagnose` — interpreting reports, querying the manifest
- `lookbook-recipe` — customizing recipes per-call or via user YAML
