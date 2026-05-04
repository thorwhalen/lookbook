---
name: lookbook-profile
description: Use when adding a new subject profile to lookbook (product, scene, style, brand, font, …). A profile is a YAML file that bundles scorers, embedders, filters, a selector, and constraints into a named recipe. Triggers on "add a profile for X", "support product LoRAs", "scene profile", "style LoRA recipe", "new subject type", or work in lookbook/profiles/.
---

# Adding a subject profile to lookbook

A profile is a declarative recipe — a YAML file that says how to curate
images for a specific subject type (person, product, scene, style, brand,
font, etc.). It bundles scorers, embedders, filters, a selector, and
default constraints into a single named unit.

The reason profiles exist as their own concept (rather than just being
in-code RECIPES dicts): a profile is **the user-visible plugin point**.
Adding a new subject type — `lookbook curate ./photos --recipe scene` —
should NOT require touching core code. Just drop a YAML in
`lookbook/profiles/` (or in the user's config folder) and it works.

Read `lookbook-dev` first if you haven't.

## Where profiles live

Two locations, in priority order (user wins over shipped):

1. `<user_config>/profiles/*.yaml` — user-edited profiles, found via
   `config2py.get_app_config_folder("lookbook")`.
2. `lookbook/profiles/*.yaml` — profiles shipped with the package.

The loader (`lookbook.profiles.load(name)`) tries (1) first, then (2),
then raises `KeyError`.

## Profile YAML schema

Every profile file follows the same shape:

```yaml
name: my_profile
description: |
  One or two sentences explaining when to use this profile.

scorers:
  - scorer_id_1
  - [scorer_id_2, {param: value}]   # tunable variant

embedders:
  - embedder_id

filters:
  - filter_id_1
  - [filter_id_2, {threshold: 0.5}]

selector:
  - selector_id
  - {selector_param_1: value, selector_param_2: value}
  # OR just `selector: top_k` for a no-config selector

diagnose_clusters: 8

constraints:
  quotas:
    bin_label_1: 3
    bin_label_2: 5
```

Notes on the YAML conventions:

- The `[name, kwargs]` form is the YAML-friendly version of the
  `(name, kwargs)` tuple used in code. The runtime accepts either.
- `embedders`, `filters`, `constraints`, `diagnose_clusters` are all
  optional — omit them if they don't apply.
- `description` is shown by `lookbook list-recipes` so make it useful.

## The 7-step recipe for a new profile

### 1. Decide what the bin attribute is

A profile's selector will almost always be a **quota** selector wrapping
something diversity-aware (top-K, facility_location). The bin attribute
determines what dimensions of variety the user gets to control:

| Subject | Natural bin axis |
|---|---|
| Person / character | `head_pose.yaw_bin` (front / 3-quarter / profile / back) |
| Product / object | viewpoint sphere bins (front / side / top / detail) |
| Scene / environment | time-of-day / weather / focal length |
| Style | dummy bin: subject-class for diversity-of-content within style |
| Brand | logo-position bin (top-left / centered / corner / inverted) |
| Font | glyph-coverage bin (uppercase / lowercase / digits / symbols) |

If your subject doesn't have an obvious axis, the safe default is to bin
on the dominant cluster from the embedding space — but at that point you
might as well skip the quota and just use facility_location directly.

### 2. Pick the embedding space

Three choices in lookbook today:
- `clip_vit_b32` — semantic ("same concept, different look")
- `dinov2_base` — visual ("same shape/scene, different style")
- `arcface` — identity ("same person") — only useful for person

For most non-person subjects, **DINOv2 is the default**. CLIP wins when
the user wants to dedupe by *meaning* (keep one shot per concept) rather
than by visual similarity.

### 3. List the scorers needed

Always include the cheap funnel — they're free:
- `resolution`, `file_hash`, `phash`, `blur`, `exposure`

Then add subject-specific:
- For person: `insightface` (face_box), `face_area`, `head_pose`, `face_quality`
- For product: `yolo` (bbox of subject) — not yet shipped, see
  `lookbook-add-scorer` for adding new ones
- For style: an "aesthetic" scorer (LAION-Aesthetic-V2) — Phase 2.5+

### 4. List the filters needed

Always include the cheap-funnel filters:
- `min_resolution` (with subject-appropriate threshold)
- `exposure_range`, `min_blur`
- `no_exact_duplicate`, `no_near_duplicate`

Then add subject-specific:
- For person: `has_face`, `single_face_only`, `min_face_area`, `min_face_confidence`
- For product: `subject_present` (writes a "subject_present: bool" annotation)

### 5. Configure the selector

Standard pattern: `quota` wrapping `facility_location`:

```yaml
selector:
  - quota
  - bin_metric_id: <annotation path of the bin>
    inner_selector_id: facility_location
    inner_overrides:
      embedding_space: <embedding to read>
      quality_metric_id: <per-image quality score id>
      weight_quality: 0.4
      weight_diversity: 1.0
    embedding_space: <same as inner; pipeline pre-fetches via this>
```

The duplication of `embedding_space` (once on QuotaSelector, once on
inner_overrides) is intentional: the pipeline reads the outer one to
decide which space to pre-fetch; the inner uses it to look the vectors
up in `constraints["embeddings"]`.

### 6. Set default quotas

Make the quotas total exactly the typical `k`. For person/character
LoRAs that's 20; for products it's often 30. Quotas under-fill silently
when there aren't enough candidates per bin (the QuotaSelector backfills
from any bin to hit `k`); set `strict: true` on the selector if you
want to disable that fallback.

### 7. Test

Two tests, in order:

```python
def test_my_profile_loads():
    spec = load("my_profile")
    assert spec["name"] == "my_profile"
    # Check it parses what you expect.

def test_my_profile_end_to_end_via_mock():
    """If your profile uses heavy backends, write a `my_profile_mock`
    sibling that swaps in MockFaceDetect / MockHeadPose / MockArcFace
    and run end-to-end against synthetic images."""
    spec = load("my_profile_mock")
    refs = [_ref(i) for i in range(10)]
    result = curate(refs, k=5, **_unpack(spec), stores=memory_stores)
    assert len(result.kept) == 5
```

The `_mock` sibling is the canonical Phase 3 pattern. It lets the test
suite exercise the full orchestration without downloading model weights.
See `lookbook/profiles/person_mock.yaml` for the template.

## Anti-patterns to avoid

- **Hardcoding paths to models.** If a scorer needs a model file, the
  scorer is responsible for downloading and caching it (lazy import +
  `transformers.from_pretrained` style). Profiles are pure config.
- **Putting per-user thresholds in the shipped YAML.** Defaults should
  work for the typical case. Users override via `(name, kwargs)` from
  the facade or by dropping a YAML in their config folder.
- **Inventing new bin schemes per profile.** If the bin scheme generalizes,
  put the binning logic in the upstream scorer (e.g. `_bin_yaw` lives in
  `scorers/person.py`). Profiles only reference bin labels by string.
- **Mixing concerns.** A profile says *what* runs; the scorers/selectors
  say *how*. If your profile YAML is starting to look like Python, you
  should be writing a new selector (see `lookbook-add-selector`).

## Worked example: a `product` profile (sketch)

```yaml
name: product
description: |
  Object / product LoRA curation. Detects the subject with an open-vocab
  detector, embeds with DINOv2 for visual similarity, selects across
  viewpoint bins.
scorers:
  - resolution
  - file_hash
  - phash
  - blur
  - exposure
  - [grounding_dino, {prompt: "{user_subject}"}]   # writes subject_box
  - subject_area
  - viewpoint                                       # writes viewpoint.bin
embedders:
  - dinov2
filters:
  - [min_resolution, {min_long_side: 1024}]
  - exposure_range
  - min_blur
  - no_exact_duplicate
  - no_near_duplicate
  - has_subject
  - [min_subject_area, {min_fraction: 0.10}]
selector:
  - quota
  - bin_metric_id: viewpoint.bin
    inner_selector_id: facility_location
    inner_overrides:
      embedding_space: dinov2_base
      quality_metric_id: subject_confidence
      weight_quality: 0.3
      weight_diversity: 1.0
    embedding_space: dinov2_base
diagnose_clusters: 12
constraints:
  quotas:
    front: 6
    side: 4
    side_left: 4
    top: 3
    detail: 3
```

The scorers `grounding_dino`, `subject_area`, `viewpoint`, the filters
`has_subject`, `min_subject_area`, and the bin scheme `viewpoint.bin`
don't ship yet — they would be the work of "extending lookbook to
products." But the profile YAML itself is stable: the day someone adds
those plugins, this YAML works.
