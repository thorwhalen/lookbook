# lookbook

Distill raw image pools into optimized, high-diversity reference sets for
training personalized models (character LoRAs, product LoRAs, style LoRAs).

> **Status:** Early development. Phases 0, 1, and 2 are shipped — the
> package can clean a photo dump (200→20 in <30s on a laptop, no GPU) and
> run the full embeddings + facility-location workflow with CLIP or DINOv2
> for "diverse but high-quality K from N." Phase 3 (person-LoRA-specific
> profile: InsightFace, FIQA, head-pose-aware quotas) is next. See
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

# See available scorers, embedders, filters, selectors, recipes:
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
  embedders/            mock, clip, dinov2 (lazy-imported)
  filters/              min_resolution, min_blur, exposure_range, dedup
  selectors/            top_k, facility_location (pure-numpy greedy)
  diagnose.py           cluster_coverage (set-level diagnosis)
  io/                   ingest
  __main__.py           CLI (argh): curate, list-plugins, list-recipes

.claude/skills/         Claude Code skills for development & agent use
misc/docs/              Design report + development plan
tests/                  pytest, hermetic
```
