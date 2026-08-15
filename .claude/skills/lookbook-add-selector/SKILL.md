---
name: lookbook-add-selector
description: Use when adding a new selector to lookbook — wrapping an apricot submodular function, building a quota/constraint selector, integrating DPPy, or implementing a custom set-selection algorithm. Triggers on "add a selector", "wrap a submodular function", "quota-aware selection", "DPP selector", "custom set selection".
---

# Adding a selector to lookbook

A selector picks a subset of size up-to-K from N candidates given the
manifest. It implements the `Selector` protocol from `lookbook/base.py`:

```python
class Selector(Protocol):
    def select(
        self,
        candidates: Iterable[ImageRef],
        manifest: Manifest,
        k: int,
        constraints: Mapping[str, Any] = (),
    ) -> list[ImageRef]: ...
```

Read `lookbook-dev` first if you haven't.

## When to write a new selector vs reuse an existing one

| Want | Use |
|---|---|
| "Pick K highest-scoring" | `top_k` (shipped) |
| "Pick K diverse, well-represented" | `facility_location` (shipped, pure-numpy greedy) |
| "Pick K under quotas (3 close-up, 5 medium, …)" | `quota` (shipped, wraps any inner selector) |
| "Pick K with hard must-include / must-exclude" | constrained wrapper |
| "Pick K probabilistically diverse" | DPPy-based |
| "Faster than the greedy at N > 10k" | `apricot` wrapper |

If the answer is `top_k` with a different scoring metric, **don't write a
new selector** — pass `("top_k", {"metric_id": "your_metric"})` from the
facade. New selectors are for new *algorithms*, not new metric choices.

## The 6-step recipe

### 1. Pick a `selector_id`

Stable, snake_case. Don't change once shipped — it's what users put in
recipes.

Good: `"facility_location"`, `"k_dpp"`, `"quota"`, `"submodular_mix"`.

### 2. Decide the input shape

Selectors usually need one of:
- A **per-image score** (top-K, threshold).
- A **vector embedding** (submodular over similarity).
- A **discrete attribute** (quota, e.g. head-pose bin).
- A **mix** (the realistic case).

Read these from the manifest with `value_of(manifest, image_id, metric_id)`.
If the dep isn't there, raise a clear error pointing at the missing scorer:

```python
emb = value_of(manifest, ref.image_id, "dinov2_emb")
if emb is None:
    raise ValueError(
        "facility_location selector requires the 'dinov2_emb' annotation. "
        "Add it to the pipeline's scorer_ids."
    )
```

### 3. Implement `select`

Pattern:

```python
@dataclass
class FacilityLocation:
    selector_id: str = "facility_location"
    embedding: str = "dinov2_emb"
    weight_quality: float = 0.3
    weight_diversity: float = 1.0

    def select(self, candidates, manifest, k, constraints=None):
        candidates = list(candidates)
        if k <= 0 or not candidates:
            return []
        embeddings = np.stack(
            [
                np.asarray(value_of(manifest, r.image_id, self.embedding))
                for r in candidates
            ]
        )
        # Heavy import inside the method:
        from apricot import FacilityLocationSelection

        sel = FacilityLocationSelection(min(k, len(candidates)))
        sel.fit(embeddings)
        idx = sel.ranking[:k]
        return [candidates[i] for i in idx]
```

Rules:
- **Lazy-import heavy deps.**
- **Always return a list, never a generator.** Callers index it.
- **Cap `k` to `len(candidates)`** — never raise on small candidate sets.
- **Read manifest read-only.** Selectors don't write annotations.
- **Honor `constraints`** when set; default to ignoring unknown keys.

### 4. Constraints

`constraints` is a free-form `Mapping[str, Any]` that lets the user pass
selector-specific settings without changing the protocol. Examples:

```python
constraints = {
    "must_include": ["abc123", "def456"],
    "must_exclude": ["bad1"],
    "quotas": {"head_pose_yaw_bin": {"left": 3, "center": 8, "right": 3}},
}
```

A selector that doesn't recognize a key should ignore it silently. A
selector that requires a key should validate at the top of `select` with
a clear error.

### 5. Register

```python
selectors.register("facility_location", FacilityLocation())
```

The facade resolves tunable variants on the fly:
```python
curate(..., selector_id=("facility_location", {"weight_quality": 0.5}))
```

### 6. Test

Three layers of test, in order:

```python
def test_selector_returns_at_most_k():
    refs = [BytesImageRef(payload=b"x", image_id=f"r{i}") for i in range(5)]
    sel = MySelector()
    out = sel.select(refs, {}, k=3)
    assert len(out) <= 3


def test_selector_picks_high_score_first(memory_stores):
    # Set up manifest with known scores; assert the right items come back.
    ...


def test_selector_end_to_end_via_curate(image_dir, memory_stores):
    result = curate(
        image_dir,
        k=3,
        selector_id="my_selector",
        scorer_ids=("random_score",),
        stores=memory_stores,
    )
    assert len(result.kept) == 3
```

## Common patterns

### How embeddings reach the selector

The pipeline reads `selector.embedding_space` (a string attribute) and
pre-fetches all vectors for survivor candidates from
`stores.embeddings[space_id]`, passing them as `constraints["embeddings"]`
— a `dict[image_id, np.ndarray]`. Your selector reads from there:

```python
def select(self, candidates, manifest, k, constraints=None):
    embs = (constraints or {}).get("embeddings") or {}
    missing = [r.image_id for r in candidates if r.image_id not in embs]
    if missing:
        raise ValueError(
            f"missing embeddings for {len(missing)} candidates in "
            f"space {self.embedding_space!r} — run the embedder first"
        )
    X = np.stack([np.asarray(embs[r.image_id], dtype=np.float32) for r in candidates])
    ...
```

This is what `lookbook.selectors.submodular.FacilityLocation` does. Read
that module for the canonical pattern — it's pure numpy, no apricot
dependency, and adequate up to N ≈ a few thousand candidates.

### Wrapping `apricot`

Apricot's API is consistent: instantiate, fit on a numpy matrix, read
`.ranking`. The wrapper is small. Available submodular functions:

- `FacilityLocationSelection` — best general-purpose; "represent the pool."
- `FeatureBasedSelection` — encourages saturation across feature dimensions.
- `GraphCutSelection` — balances representativeness and diversity.
- `SaturatedCoverageSelection` — diminishing returns past a threshold.
- `SumRedundancySelection` — explicit redundancy avoidance.
- `MaxCoverageSelection` — set-cover style.
- `MixtureSelection` — convex combination of any of the above.

Use `MixtureSelection` when the user has multiple objectives (e.g. quality
× diversity) and you want one knob per objective.

### Quota / constraint selector

The standard pattern: bin candidates by a discrete attribute, run a
sub-selector per bin up to that bin's quota, concatenate.

```python
def select(self, candidates, manifest, k, constraints=None):
    bin_attr = self.bin_metric_id
    quotas = (constraints or {}).get("quotas", {}).get(bin_attr, {})
    by_bin = {}
    for r in candidates:
        b = value_of(manifest, r.image_id, bin_attr)
        by_bin.setdefault(b, []).append(r)
    out = []
    for bin_name, quota in quotas.items():
        candidates_in_bin = by_bin.get(bin_name, [])
        sub = self.inner_selector.select(
            candidates_in_bin, manifest, k=quota, constraints=constraints
        )
        out.extend(sub)
    return out[:k]
```

This composes well with any inner selector — top-K within a bin, facility
location within a bin, etc.

### k-DPP via DPPy

```python
def select(self, candidates, manifest, k, constraints=None):
    from dppy.finite_dpps import FiniteDPP

    embeddings = np.stack([...])
    L = embeddings @ embeddings.T  # similarity kernel
    dpp = FiniteDPP("likelihood", **{"L": L})
    dpp.sample_exact_k_dpp(size=k)
    return [candidates[i] for i in dpp.list_of_samples[-1]]
```

DPPs are slower than apricot but give principled diversity. Nothing in the
package uses them today; reserve for small candidate pools (< 1000).

## Anti-patterns to avoid

- **Selectors that recompute scores.** If you need a score, it should be a
  scorer that runs upstream. Selectors are read-only over the manifest.
- **Selectors that mutate inputs.** Don't shuffle `candidates` in place;
  copy first.
- **Returning more than `k`.** Always slice; no exceptions.
- **Hard-coding metric ids.** Make `metric_id` and `embedding` config
  fields so the same selector works with `clip_emb`, `dinov2_emb`,
  `arcface_emb`.
- **Failing on empty candidate sets.** Return `[]`; the report will
  surface the empty result.
