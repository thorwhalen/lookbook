---
name: lookbook-curate
description: Use when the user wants to curate a folder of images into a smaller training set — for character LoRAs, product LoRAs, style LoRAs, or any "pick K best from N candidates" image-selection task. Triggers on "curate these images", "pick K for LoRA training", "choose images for character LoRA", "filter my photo dump", "select diverse training images", "build a training set". Also use to drive lookbook over MCP/HTTP from an agent.
---

# Curating images with lookbook

`lookbook` distills a pool of N candidate images into K well-chosen ones.
The selection is **two-level** — every image gets per-image scores
(blur, exposure, face quality), and the chosen *set* gets a coverage
score (diversity, pose balance, identity uniformity).

This skill is for **using** lookbook to curate. If you're building or
extending lookbook itself, use `lookbook-dev` and friends.

## When to reach for which recipe

The recipe is the single most important parameter. Pick it from the
user's intent:

| User says | Recipe |
|---|---|
| "clean up these photos", "drop the bad ones" | `funnel` |
| "phone-camera quality, less strict" | `funnel_relaxed` |
| "pick 20 diverse training images" | `diverse` (DINOv2) |
| "match by content, not look" | `diverse_clip` (CLIP) |
| "character LoRA", "person LoRA", "headshots" | `person` |
| "smoke test", "no model downloads" | `funnel` or `person_mock` |

You can always start with `list_recipes` to see what's available — new
profiles (product, scene, style) may have been added.

## The standard workflow

### 1. Confirm what the user wants

The two questions that matter:
- **K** — target subset size. Defaults vary; 20 is typical for character
  LoRAs, 30 for products.
- **Subject type** — drives the recipe choice. If unclear, ask.

If the user dropped a folder path with no further instruction, default
to `recipe="funnel"` and `k=20`, run it, and let them iterate.

### 2. Run `curate_source`

Via MCP:
```python
result = await client.call_tool("curate_source", {
    "source_path": "/abs/path/to/photos",
    "k": 20,
    "recipe": "person",
})
```

Via HTTP:
```bash
curl -X POST http://localhost:8000/curate_source \
  -H 'Content-Type: application/json' \
  -d '{"source_path":"/abs/photos","k":20,"recipe":"person"}'
```

Via Python:
```python
from lookbook import curate
result = curate("/abs/path", k=20, scorer_ids=("resolution", "blur", "phash"),
                filter_ids=("min_resolution", "no_exact_duplicate"),
                selector_id="top_k")
```

### 3. Read the run record

Every `curate_source` returns a `RunResult` shaped like:

```python
{
  "run_id": "run-20260504T154932",
  "kept": ["abc123…", "def456…", …],         # K image_ids
  "candidates": [...],                         # all N input ids
  "selector_id": "facility_location",
  "scorer_ids": ["resolution", "blur", …],
  "report": {
    "n_candidates": N,
    "n_survivors": M,                          # passed all filters
    "n_kept": K,
    "dropped_by_filter": {"min_blur": 12, …},
    "notes": {
      "embedder_ids": ["dinov2_base"],
      "cluster_coverage": {…}
    }
  }
}
```

The two fields that matter for follow-up:
- **`report.dropped_by_filter`** — *why* images were rejected. Each drop
  is attributed to the *first* filter that rejected it.
- **`report.notes.cluster_coverage`** (when present) — which visual
  clusters were filled / missed. See `lookbook-diagnose`.

### 4. If the user wants to iterate

Common follow-ups, with the right tool to call:

| User asks | Action |
|---|---|
| "Why was this one rejected?" | `get_annotations(image_id)` — full provenance |
| "Show me the kept set" | The `kept` list of ids; if they want bytes, call `get_image(id)` |
| "Try a different K" | Re-run `curate_source`; the manifest cache is honored |
| "Try with a different recipe" | Re-run; scorer cache hits where configs match |
| "I want fewer profile shots" | Override quotas — see `lookbook-recipe` |
| "What does the kept set look like in coverage?" | See `lookbook-diagnose` |

Re-running with the same scorers but a different selector is **free** —
the heavy work (CLIP/DINOv2/InsightFace) is cached against
`(image_id, metric_id, config_hash)`.

## How to interpret a curated set

A few things that surprise users the first time:

1. **The kept set is in selection order, not score order.** Facility-
   location picks the most "central" candidate first, then iteratively
   the one that adds the most diversity. The 5th pick is typically the
   most "different" image — that's normal.
2. **Drop counts attribute to the *first* filter that rejected.** If
   `min_blur` and `exposure_range` would both have rejected an image,
   only the first-listed one in the recipe gets the credit. To re-attribute,
   run `get_annotations(image_id)` and read the underlying values.
3. **Cluster coverage uses the candidate-pool's clusters, not a fixed
   ontology.** "9 of 12 clusters filled" means "9 of the 12 visual modes
   I saw in *your* candidates" — the missing 3 don't necessarily map to
   semantically meaningful gaps.
4. **`person` profile auto-applies pose-bin quotas.** The default quotas
   are in the YAML (`person.yaml`). See `lookbook-recipe` to override.

## Anti-patterns to avoid

- **Don't keep re-running with `recipe="random"` and expecting useful
  results.** It exists for end-to-end orchestration testing, not
  curation.
- **Don't run `recipe="diverse"` or `recipe="person"` without warning the
  user about the model download.** First run pulls 350MB+. Subsequent
  runs are fast.
- **Don't conflate `n_survivors` and `n_kept`.** Survivors passed all
  filters; kept is the size of the chosen subset (`n_kept ≤ n_survivors`,
  often less when K < survivors).
- **Don't send raw image bytes back to the user when ids would do.** All
  the lookbook tools speak in `image_id`; the user can resolve to bytes
  via `get_image(id)` only when they really need them.
