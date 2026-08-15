---
name: lookbook-diagnose
description: Use after a lookbook curation run to interpret the result, explain why specific images were dropped or kept, query the manifest for provenance, or read cluster-coverage diagnoses. Triggers on "why was this image rejected", "what's missing from my training set", "explain the curation report", "this image was dropped — why", "what coverage gaps does my set have".
---

# Diagnosing a lookbook curation result

The point of lookbook is not just to pick K images, but to *explain*
what it picked and why. Every curation run leaves a complete trail in
the manifest: the per-image annotations that produced each decision,
the filter that fired, and the cluster coverage of the kept set.

This skill is the playbook for answering follow-up questions after a
`curate_source` call. Use `lookbook-curate` first if you haven't curated
yet.

## The three diagnostic questions, mapped to tools

| Question | Tool to call |
|---|---|
| Why was *this* image dropped? | `get_annotations(image_id)` |
| What did the run as a whole reject? | `get_run(run_id)` → `report.dropped_by_filter` |
| What's missing / underrepresented in the kept set? | `get_run(run_id)` → `report.notes.cluster_coverage` |
| Is *this* image the same person/subject as my reference? | `compare_to_reference(reference_paths, candidate_path)` |

That last one is a different question from the other three: it compares two
images rather than reading a run's trail, so it works on any image, curated
or not. Its `passed` flag is advisory — report the `score` alongside it.

## Per-image diagnosis: `get_annotations`

`get_annotations(image_id)` returns every annotation in the manifest for
that image. Each annotation carries the metric_id, the value, the
config_hash, and the backend that produced it.

The standard reading flow:

1. Look at `resolution.long_side` — was the image too small?
2. Look at `blur` — was Laplacian variance below the threshold? (A
   typical rejection cutoff is 100; "blurry" images often score < 50.)
3. Look at `exposure.frac_underexposed` and `frac_overexposed` — was
   the image mostly black or white? (>50% rejection threshold is the
   default.)
4. For person profiles: `face_box` — was a face detected? `face_area.fraction`
   — was the face big enough? `face_box.confidence` — did the detector
   trust the box?
5. `phash` — duplicate detection. Two images with `phash` Hamming distance
   ≤ 5 are flagged as near-duplicates by `no_near_duplicate`.

The drop attribution in the run report tells you **which filter**
rejected an image, but the manifest tells you **what value caused it**.
Use both.

### Worked example

User: "Why was image abc123 rejected?"

```
get_annotations({"image_id": "abc123"})
→ {
    "annotations": [
      {"metric_id": "resolution", "value": {"long_side": 412, ...}},
      {"metric_id": "file_hash", "value": "..."},
      {"metric_id": "blur", "value": 87.3},
      {"metric_id": "exposure", "value": {...}}
    ]
  }
```

Then `get_run(run_id)` shows `dropped_by_filter: {"min_resolution": 1, ...}`.
Cross-referencing: long_side=412 < default 1024 → caught by `min_resolution`.
Tell the user: "Rejected by `min_resolution` (long side 412px, threshold
1024px). To keep images this size, re-run with
`recipe=funnel_relaxed` (threshold 512) or with `(min_resolution, {min_long_side: 400})`."

## Run-level diagnosis: drop attributions

`report.dropped_by_filter` is a `{filter_name: count}` dict — each drop
is attributed to the **first filter that rejected it**, not to the union
of all filters that would have. This matters when reading reports:

- A high `min_blur` count usually means real blurry images, but if
  `exposure_range` runs *after* `min_blur` you might miss seeing that
  many of those "blurry" images were actually solid colors (Laplacian
  variance ≈ 0 in both cases). Reorder filters in the recipe to get
  finer attribution.
- The default funnel order — `min_resolution → exposure_range → min_blur
  → no_exact_duplicate → no_near_duplicate` — is intentional. Solids
  hit `exposure_range`, true blur hits `min_blur`.

## Set-level diagnosis: cluster coverage

When the recipe has `diagnose_clusters > 0` and ran a real embedder, the
report includes a `cluster_coverage` block:

```python
{
  "n_clusters": 12,                     # how many clusters KMeans formed
  "n_clusters_filled": 9,               # clusters with ≥ 1 kept image
  "cluster_sizes_candidates": [4, 7, 5, 3, 2, 6, 8, 5, 4, 3, 9, 6],
  "cluster_sizes_kept":       [2, 3, 0, 1, 0, 2, 1, 2, 0, 0, 4, 5],
  "empty_clusters": [2, 4, 8, 9],
  "underrepresented_clusters": [2, 4, 8]   # empty AND ≥2 candidates
}
```

How to read this:

- **`empty_clusters`** is the raw "not represented at all" list.
- **`underrepresented_clusters`** filters that to clusters with ≥2
  candidates — the ones the selector *could have* covered but chose not
  to. These are the meaningful gaps; empties with only 1 candidate are
  probably noise.
- The cluster *indexes* are arbitrary (KMeans labels). They don't map to
  semantic categories like "front vs profile." For semantic gaps in a
  person-LoRA, look at the manifest's `head_pose.yaw_bin` distribution
  in the kept set instead.

### When clusters under-fill

If the user is unhappy with a curated set, a high `underrepresented_clusters`
count usually means one of:

1. **The selector's `weight_quality` is too high.** It's prioritizing
   high-quality images over diverse ones; lower it (try 0.1–0.3 instead
   of 0.5+).
2. **The candidate pool is genuinely lopsided.** Most candidates are in
   one or two clusters, so the selector picks from there. The user needs
   more variety in their input, not a different selector.
3. **K is too small.** If you're picking K=10 from 200 candidates with
   12 natural clusters, you can't fill all 12 — that's mathematics, not
   a bug.

## Person-profile-specific diagnosis

For person/character LoRAs, the meaningful coverage axis is `head_pose.yaw_bin`,
not KMeans clusters. The `person` profile bins yaw into:

```
front (|yaw|≤10) | three_quarter (10–30) | profile (30–60) | back (>60)
```

The scheme is symmetric, so every bin except `front` has a left-hand twin:
the actual labels are `front`, `three_quarter` / `three_quarter_left`,
`profile` / `profile_left`, `back` / `back_left`. Tally the exact labels —
`three_quarter` alone is half the bin.

To diagnose pose coverage manually after a `person` curation:

1. Get each kept image's `head_pose` annotation via `get_annotations(id)`.
2. Tally `yaw_bin` values in the kept set.
3. Compare to the recipe's `quotas` block (the YAML default is
   `front: 8, three_quarter: 4, …`).

If the count is `< quota` for a bin, the candidate pool didn't have
enough images of that pose type — tell the user to capture more.

## Anti-patterns to avoid

- **Don't trust drop counts as a complete attribution.** Filter
  short-circuits — an image rejected by `min_resolution` won't have
  even been *checked* by `min_blur`. Use `get_annotations` for the
  ground truth.
- **Don't report cluster ids as if they were semantic.** Cluster 7 in
  one run is unrelated to cluster 7 in the next; KMeans labels are
  permutation-arbitrary.
- **Don't hand the user a raw run record.** Translate. "Rejected by
  min_resolution: long side 412px (threshold 1024px); fixable by lowering
  the threshold or capturing higher-resolution shots" beats "your
  `report.dropped_by_filter.min_resolution` is 7."
- **Don't skip the manifest when the report seems off.** The report is a
  summary; the manifest is the source of truth.
