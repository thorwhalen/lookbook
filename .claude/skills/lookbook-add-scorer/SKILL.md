---
name: lookbook-add-scorer
description: Use when adding a new per-image metric (scorer) to lookbook — wrapping a model from pyiqa/insightface/CLIP, adding technical metrics like noise or compression detection, or any "score one image" annotation producer. Triggers on "add a scorer", "wrap pyiqa model X", "add a metric for Y", "score images by Z".
---

# Adding a scorer to lookbook

A scorer is the unit of per-image annotation. It implements the `Scorer`
protocol from `lookbook/base.py`:

```python
class Scorer(Protocol):
    metric_id: str
    cost_tier: int
    requires: tuple[str, ...]
    config_hash: str
    def score(self, ref: ImageRef, manifest: Manifest) -> Any: ...
```

This skill is the recipe for adding one end-to-end. Read
`lookbook-dev` first if you haven't.

## The 8-step recipe

### 1. Decide the cost tier

| Tier | Wall-clock | Hardware | Examples |
|---|---|---|---|
| T0 | < 1 ms | CPU | Header read, file hash, EXIF |
| T1 | 1–50 ms | CPU | Blur, exposure, perceptual hash |
| T2 | 5–100 ms | GPU helpful | CLIP, DINOv2, ArcFace, NR-IQA |
| T3 | 100 ms – 2 s | GPU required | MLLM critique, FIQA, multi-stage |

The orchestrator runs cheaper tiers first. Set `cost_tier` honestly — the
budget controller (Phase 2) will skip T2/T3 scorers under tight budgets.

### 2. Pick a `metric_id`

Stable identifier, snake_case. This is the manifest key. Don't change it
once shipped — annotations cached against the old id will go stale silently.

Good: `"blur"`, `"face_count"`, `"clip_iqa"`, `"arcface_embedding"`.
Bad: `"BlurDetector_v3_final"`, `"my_metric"`.

### 3. Declare `requires` (other metric_ids you depend on)

Examples:
- `Blur` requires nothing — it works from raw pixels.
- `FaceArea` requires `face_box` — it reads the bounding box from manifest.
- `FIQA` requires `face_box` (and the actual face crop).

The orchestrator topologically sorts scorers by `requires` and runs them in
order. Missing required deps are *not* an error at runtime — the assumption
is they may have been pre-populated by an earlier run.

### 4. Compute `config_hash`

This is the cache key. Re-running with the same config yields a cache hit;
changing a threshold, model name, or any parameter must invalidate.

The standard pattern, implemented in `scorers/technical.py`:

```python
def _hash_config(**kwargs) -> str:
    payload = repr(sorted(kwargs.items())).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]

@dataclass
class MyScorer:
    metric_id: str = "my_metric"
    cost_tier: int = 1
    requires: tuple = ()
    threshold: float = 0.5
    model: str = "v2"

    @property
    def config_hash(self) -> str:
        return _hash_config(threshold=self.threshold, model=self.model)
```

If two configurations should be considered equivalent for caching purposes,
exclude that field from `_hash_config`. Don't include the `metric_id` itself
or fields that don't affect the output.

### 5. Write the scoring function

```python
def score(self, ref: ImageRef, manifest: Manifest) -> Any:
    # Lazy-import heavy deps INSIDE the method:
    import pyiqa  # ← never at module top
    img = ref.open()
    return float(pyiqa.create_metric(self.model)(img))
```

Rules:
- **Lazy-import heavy deps inside the method.** A user importing `lookbook`
  with no extras must not pay the torch import cost.
- **Read upstream annotations from the manifest** if `requires` declares them.
  Use `lookbook.manifest.value_of(manifest, ref.image_id, "face_box")`.
- **Return JSON-able values** when feasible (numbers, dicts, lists, strings).
  This makes manifest persistence work with the default JSON codec.
- **Never mutate the manifest yourself.** The pipeline writes the result
  for you. Returning is the whole interface.

### 6. Handle missing optional deps gracefully

When wrapping an optional dependency, raise a clear `ImportError` pointing
to the right pip extras:

```python
try:
    import pyiqa
except ImportError as e:
    raise ImportError(
        "musiq scorer requires pyiqa. "
        "`pip install lookbook[embed]` or `pip install pyiqa`."
    ) from e
```

### 7. Register

Two equivalent ways:

```python
# A: register an instance with default config (most common)
scorers.register("my_metric", MyScorer())

# B: decorator on a callable (when the scorer is a function with attrs)
@register_scorer("my_metric")
class MyScorer: ...
```

Tunable variants are made on the fly via the facade:
```python
curate(..., scorer_ids=[("my_metric", {"threshold": 0.7})])
```

### 8. Test it

A canonical Phase 1 scorer test (see `tests/test_phase1.py`):

```python
def test_my_metric_basic(sharp_ref):
    v = MyScorer().score(sharp_ref, {})
    assert isinstance(v, float)
    assert 0 <= v <= 1
```

Three things to test:
1. Output shape and type are stable.
2. The `config_hash` changes when config changes.
3. End-to-end: `curate(..., scorer_ids=("my_metric",), ...)` runs without error.

## Common patterns

### Wrapping a `pyiqa` model — a 12-line template

```python
@dataclass
class PyiqaScorer:
    metric_id: str = "musiq"
    cost_tier: int = 2
    requires: tuple = ()
    backend: str = "pyiqa"
    pyiqa_model: str = "musiq"

    @property
    def config_hash(self) -> str:
        return _hash_config(model=self.pyiqa_model)

    def score(self, ref: ImageRef, manifest: Manifest) -> float:
        import pyiqa
        if not hasattr(self, "_iqa"):
            self._iqa = pyiqa.create_metric(self.pyiqa_model)
        with ref.open() as img:
            return float(self._iqa(img.convert("RGB")).item())
```

The `_iqa` attribute caches the model per-instance — pyiqa downloads weights
on first construction, so caching matters. Avoid `lru_cache` on the class
because pyiqa metrics aren't hashable in the obvious way.

### Wrapping an embedder

Embedders implement a separate Protocol (`Embedder` in `base.py`) and
register into `registry.embedders`. Their `embed(ref)` returns a numpy
array; the pipeline writes the vector to `stores.embeddings[space_id]`.

### A scorer that reads the face box

```python
def score(self, ref, manifest) -> dict:
    box = value_of(manifest, ref.image_id, "face_box")
    if box is None:
        return {"face_area": 0.0}  # no face detected upstream
    res = value_of(manifest, ref.image_id, "resolution") or {}
    total = res.get("width", 1) * res.get("height", 1)
    bw = box["x2"] - box["x1"]
    bh = box["y2"] - box["y1"]
    return {"face_area": (bw * bh) / total}
```

Declare `requires=("face_box", "resolution")` on this scorer — the
orchestrator will run those first.

## Cache invalidation rules

Three things invalidate a cached annotation:

1. **`config_hash` mismatch**: scorer was reconfigured.
2. **Manual deletion**: pop the key from the manifest.
3. **Replacement**: write a new `Annotation` to the same key.

Time-based invalidation is **not** automatic. If a model was updated
out-of-band (e.g. pyiqa shipped new weights), bump the `config_hash` by
adding a `version` field to `_hash_config`.

## Anti-patterns to avoid

- **Importing torch / cv2 / pyiqa at module top.** Breaks the laptop tier.
- **Mutating shared state across scorers.** Scorers must be pure functions
  of `(ref, manifest)`. The pipeline calls them in unpredictable order
  within a tier.
- **Returning numpy arrays directly to the manifest.** They don't JSON-serialize.
  Convert to lists, or use a custom codec on the manifest store.
- **Raising on missing `requires`.** The pipeline already orders them; if
  you reach a scorer and its dep isn't there, the user has a non-default
  pipeline and the right move is to return a sensible default (e.g. None,
  zero, or an empty dict).
