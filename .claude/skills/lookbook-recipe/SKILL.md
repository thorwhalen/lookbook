---
name: lookbook-recipe
description: Use when the user wants to customize a lookbook curation pipeline beyond what the built-in recipes / profiles provide — overriding filter thresholds, swapping selectors, adjusting quotas, building a one-off recipe inline. Triggers on "different defaults", "custom scorer weights", "skip aesthetic scoring", "build my own pipeline", "tweak the funnel", "override the threshold", "what-if reweighting".
---

# Customizing a lookbook curation recipe

Lookbook ships a handful of built-in recipes (`funnel`, `diverse`, `person`,
…). For most users they're enough; for the ones who want to tune, this
skill is the playbook.

There are three ways to customize, in increasing order of permanence:

1. **Per-call overrides** — pass tunable plugin specs as `(name, kwargs)`
   tuples through `curate(...)`.
2. **A user YAML profile** — drop a `*.yaml` in `<user_config>/profiles/`.
3. **A new shipped profile** — add `*.yaml` to `lookbook/profiles/`.
   This is the dev path; see `lookbook-profile` instead.

This skill covers (1) and (2). For dev-time work see `lookbook-profile`,
`lookbook-add-scorer`, `lookbook-add-selector`.

Before tuning anything: if the job is "pick the best reference image(s) of
this character / this environment", `curate_for_character` and
`curate_for_environment` are already that recipe, pre-tuned. See
`lookbook-curate`.

## Per-call overrides — the (name, kwargs) tuple form

Every `*_ids` argument to `curate(...)` accepts either a string or a
`(name, kwargs)` tuple. The tuple re-instantiates the registered class
with the kwargs you supply.

```python
from lookbook import curate

result = curate(
    source,
    k=20,
    scorer_ids=(
        "resolution",  # string form, defaults
        ("blur", {"max_side": 256}),  # tuned form
        "exposure",
    ),
    filter_ids=(
        ("min_resolution", {"min_long_side": 800}),
        "exposure_range",
        ("min_blur", {"threshold": 30.0}),
        "no_exact_duplicate",
        ("no_near_duplicate", {"max_distance": 8}),
    ),
    selector_id=(
        "facility_location",
        {
            "embedding_space": "dinov2_base",
            "quality_metric_id": "blur",
            "weight_quality": 0.05,
            "weight_diversity": 1.0,
        },
    ),
)
```

The CLI accepts the same forms via the `--filter` / `--scorer` flags
when paired with JSON-shaped values. For more than one or two overrides,
write a YAML profile instead — the readability is worth it.

## What's tunable per plugin

The most-tuned defaults, by plugin id:

### Filters

| Filter | Most-tuned kwargs |
|---|---|
| `min_resolution` | `min_long_side=1024` — drop to 512 for phones, 1536 for SDXL strict |
| `min_blur` | `threshold=100` — drop to 30 for phones, 200 for studio |
| `exposure_range` | `max_underexposed_frac=0.5`, `max_overexposed_frac=0.5` |
| `no_near_duplicate` | `max_distance=5` (hash bits) — 8 catches more, 3 fewer |
| `min_face_area` | `min_fraction=0.05` — 0.10 for tight portraits, 0.02 for full-body |
| `min_face_confidence` | `threshold=0.5` — bump to 0.7 for strictness |

### Scorers

| Scorer | Most-tuned kwargs |
|---|---|
| `blur` | `max_side=512` (downsample for speed) |
| `exposure` | `max_side=512` |
| `phash` | `algorithm="phash"` — also `ahash`, `dhash`, `whash` |
| `face_quality` | `blur_normalization=500.0` — divisor for the sharpness term |
| `technical_quality` | `blur_normalization=500.0`, `target_long_side=1024`, `clipping_penalty=2.0` |
| `identity_similarity` | `threshold=0.85`, `aggregation="max"`, `embedder="arcface"`, `normalization="rescale"` |

### Embedders

| Embedder | Most-tuned kwargs |
|---|---|
| `clip` | `model_name="openai/clip-vit-large-patch14"` (~600 MB, slower, better) |
| `dinov2` | `model_name="facebook/dinov2-small"` (~85 MB) or `"facebook/dinov2-large"` (~1.2 GB) |

### Selectors

| Selector | Most-tuned kwargs |
|---|---|
| `top_k` | `metric_id` (which annotation to rank by), `descending=True` |
| `facility_location` | `embedding_space`, `quality_metric_id`, `weight_quality` (0.0–1.0), `weight_diversity` (typically 1.0) |
| `quota` | `bin_metric_id` (e.g. `"head_pose.yaw_bin"`), `inner_selector_id`, `inner_overrides`, `strict` |

## The "what-if reweighting" pattern

The cheapest customization is to keep the scorers / embedders the same
and only change the selector. The pipeline cache means the heavy work
is *not* redone.

Sequence:

1. First run with `recipe="diverse"` and the default selector weights.
2. User wants more diversity → re-run with
   `selector_id=("facility_location", {"weight_quality": 0.0, "weight_diversity": 1.0})`.
3. Cache hits on every scorer and the embedder; only the selector
   re-runs (~50 ms for 200 images).

This is the pattern to recommend when the user is dialing in their
preferences interactively. **Don't blow the cache** by changing scorer
configs unless the user is genuinely changing what's being measured.

## A user YAML profile (the persistent way)

For settings the user wants to keep around, write a profile YAML.
Drop it under `<user_config>/profiles/` (resolved by `config2py` —
typically `~/.config/lookbook/profiles/` or
`~/Library/Application Support/lookbook/profiles/`). User profiles
override shipped profiles with the same name.

Minimal example — a "phone photos" funnel with relaxed thresholds:

```yaml
# ~/.config/lookbook/profiles/phone_funnel.yaml
name: phone_funnel
description: Cheap funnel tuned for typical phone photos.

scorers:
  - resolution
  - file_hash
  - phash
  - blur
  - exposure

filters:
  - [min_resolution, {min_long_side: 600}]
  - exposure_range
  - [min_blur, {threshold: 30}]
  - no_exact_duplicate
  - [no_near_duplicate, {max_distance: 8}]

selector:
  - top_k
  - {metric_id: blur}

diagnose_clusters: 0
```

Run as: `lookbook curate ./photos --recipe phone_funnel`.

The schema is documented in detail in `lookbook-profile`; that skill is
about *adding* profile types (product, scene, etc.). This skill is
about *tuning* an existing recipe for a particular user / dataset.

## Quotas — the person-profile knob

The `person` profile uses `QuotaSelector` with default per-pose-bin
quotas in the YAML:

```yaml
constraints:
  quotas:
    front: 8
    three_quarter: 4
    three_quarter_left: 4
    profile: 2
    profile_left: 2
```

To override quotas without editing the profile, pass `constraints` to
`curate(...)`:

```python
result = curate(
    source,
    k=10,
    # ... the rest of the person recipe ...
    constraints={
        "quotas": {
            "front": 5,
            "three_quarter": 3,
            "three_quarter_left": 1,
            "profile": 1,
        }
    },
)
```

If quotas under-fill, `QuotaSelector` backfills from any bin to reach
K. Set `inner_overrides.strict=true` (in the selector spec) to disable
backfill — useful when the user really wants empty bins to stay empty.

## Anti-patterns to avoid

- **Don't change the scorer config and the selector config in the same
  iteration.** You'll lose the cache and won't know which change moved
  the needle. Tune in stages.
- **Don't hand-roll a "filter then top-K" when `funnel_relaxed` would
  do.** Built-ins are tuned defaults; reach for them first.
- **Don't pass the same scorer twice with different configs hoping for
  both annotations.** The manifest is keyed by `(image_id, metric_id)`,
  not by config_hash, so the second one overwrites the first. To keep
  multiple variants, use distinct `metric_id`s (register a new scorer
  with a different name; see `lookbook-add-scorer`).
- **Don't disable `diagnose_clusters` to "speed things up."** It runs
  KMeans on the candidate embeddings — typically <1s for N≤500 and
  zero cost when no embeddings are present.
