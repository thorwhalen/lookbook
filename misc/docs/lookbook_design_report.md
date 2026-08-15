# Lookbook — A Reference-Set Curation System for Personalized Model Training

**Author:** Thor Whalen
**Date:** May 2026
**Status:** Design / research synthesis report (pre-implementation)

---

## 0. TL;DR

`lookbook` is a Python package whose job is to take a candidate pool of N images and return a *reference set* of K well-chosen images for training a personalized model — initially a person LoRA, eventually any "concept LoRA" (object, environment, style, brand).

The core insight is that this is a **two-level scoring problem**:

1. **Per-image scoring** (is this image individually good?)
2. **Set-level scoring** (does this collection together cover the concept?)

…driven by a **funnel orchestration** (cheap filters → mid-cost classifiers → expensive embeddings → submodular set selection), built on a **plugin architecture** so new metrics, new subject types, and new selection algorithms can be added without touching the core.

There **is no dominant open-source package** that owns this niche end-to-end. FiftyOne[1,2] owns generic CV dataset curation, CleanVision[3] owns issue auditing, `imagededup`[4] owns deduplication, `apricot`[5] owns submodular selection, `pyiqa`[6] owns IQA — but **nobody has stitched them into an opinionated, agent-friendly "give me my best 20" tool aimed at the LoRA / personalized-model training audience**. That is the commercialization gap.

The biggest design wins are: (a) treating curation as an **annotation pipeline over image references** (which dovetails with your existing annotation-systems architecture), (b) **separating scoring from selection** so they evolve independently, and (c) modeling the user-facing API as a **declarative recipe** (a YAML/JSON spec or a typed Python dataclass) rather than imperative code.

---

## 1. Problem Framing

### 1.1 The user story spectrum

The headline user story is:

> *"As someone trying to build a personalized model, I have N candidate images and I want help selecting the K I should train on."*

But the package needs to support a wider spectrum of agentic workflows:

- **One-shot funnel**: `lookbook.curate(images, k=20, subject="person")` → 20 images.
- **Stepwise inspection**: each filtering / scoring / selection stage is a callable that produces a manifest the user (or an LLM agent) can inspect, modify, and resume from.
- **Single-image scoring**: `score = lookbook.score(image, metric="aesthetic")` — useful as a leaf-level tool for an agent.
- **Set scoring**: `score = lookbook.score_set(images, metric="diversity_pose")` — measures how a *collection* completes itself.
- **What-if reweighting**: cache all scores once; let the user sweep selection weights without recomputing the heavy metrics. (This is the "scoring manifest" intuition from your initial sketch, made first-class.)
- **Gap-finding**: `lookbook.diagnose(set)` → "you're missing profile views, low light, and full-body shots." This is more valuable than a single score.
- **Active selection**: "show me 10 borderline cases the agent can't decide on." Useful for a human-in-the-loop UI.

### 1.2 Why this isn't just "score and take top-K"

Top-K-by-score collapses to the same image taken from slightly different angles. Several recent guides — Apatero[7], Segmind[8], LlamaGen[9] — converge on the same advice for character LoRAs: **15–30 images, where each image must add new context or perspective**. Variety dominates count. This means **diversity is a constraint, not a tiebreaker**, and the selection algorithm has to reason about the *set* the way a casting director reasons about a roster.

This is the conceptual reason the system has to be two-level: per-image quality is necessary but not sufficient, and the set objective is non-additive. The literature term for this is **submodular maximization under a cardinality constraint**[5,10,11].

### 1.3 Generalizing beyond people

LoRAs aren't only trained on faces. The package needs to extend to:

| Subject type | What "good coverage" means |
|---|---|
| **Person / character** | Pose (yaw/pitch), expression, lighting, clothing, distance (close-up / medium / full-body) |
| **Object / product** | Viewpoint sphere, scale, lighting, background variety, occlusion |
| **Environment / scene** | Time-of-day, weather, viewpoint, focal length, season |
| **Style** | Subject diversity (so style is learned independent of subject), consistent rendering attributes |
| **Brand** | Logo position, palette, layout templates |

These are all the same abstract pattern — *"a per-image quality score plus a per-attribute coverage requirement"* — which strongly motivates a plugin-based architecture rather than person-specific code.

---

## 2. Functional Decomposition

### 2.1 The eight verbs

Every operation the user or agent can want decomposes into these primitives. Designing the public API around these (rather than around workflows) keeps the system composable.

1. **Ingest** — turn a directory / URL list / archive / cloud bucket into a uniform iterable of *image references* with metadata.
2. **Probe** — read cheap header information (resolution, EXIF, file hash) without decoding pixels.
3. **Score** — compute one numerical (or categorical, or vector) annotation per image. Always idempotent; results are cached against `(image_id, metric_id, config_hash)`.
4. **Filter** — apply a predicate over annotations to drop images. Pure set operation; never destroys data.
5. **Embed** — compute a vector representation (CLIP, DINOv2, ArcFace) and store it in a vector index.
6. **Select** — choose a subset of size K (or up-to-K) that optimizes a set objective.
7. **Diagnose** — describe what's missing or over-represented in a set, in human / agent-readable terms.
8. **Export** — write the chosen subset to a training-ready directory layout (filename conventions, repeats prefix, regularization images, captions).

### 2.2 Supporting concerns

These aren't user-facing verbs but every implementation needs them:

- **Manifest** — a `MutableMapping[image_id, dict]` that holds every annotation ever computed for an image. This *is* the central data structure. It's persistent (so re-runs are cheap), serializable, and inspectable.
- **Backend registry** — pluggable models for face detection, embedding, IQA. A user can swap `clip_vit_l14` for `dinov2_vitb14` by config.
- **Subject profile** — a named bundle of (default metrics, default selector, default thresholds) for a subject type. `profiles/person.yaml`, `profiles/product.yaml`, etc.
- **Budget controller** — given a wall-clock or compute budget, decides which metrics to skip or downsample. Critical for the agent UX where 200 images on CPU should not take an hour.
- **Provenance** — every annotation records which model version, which prompt, which threshold produced it. This matters for reproducibility and for letting the user audit "why was this picture rejected?".

### 2.3 Output formats

The package should produce, at minimum, three artifacts at the end of a curation run:

- A **chosen set** (the K images, copied or symlinked into a directory).
- A **manifest** (full per-image annotations, in JSON or Parquet).
- A **report** (human-readable: "kept 18, discarded 182, reasons: blur 71, duplicates 43, no face 29, low aesthetic 18, low diversity 21" + a coverage chart for the kept set).

The report is the most underrated deliverable — it is what makes the system *teachable*, both to the human user and to an LLM agent that may want to ask "why?" before acting.

---

## 3. The Metrics Catalogue

This section is the substantive part. Each metric is tagged with: **cost tier**, **input modality**, **subject relevance**, **library**.

### 3.1 Cost tiers

Borrowing the terminology from your initial sketch and refining it into four tiers:

| Tier | Wall-clock per image | Hardware | Purpose |
|---|---|---|---|
| **T0 — Header** | < 1 ms | CPU | Triage |
| **T1 — Pixel** | 1–50 ms | CPU | Core quality / dedup |
| **T2 — Embed** | 5–100 ms | GPU recommended | Diversity, identity, aesthetics |
| **T3 — Heavy** | 100 ms – 2 s | GPU required | MLLM-based scoring, FIQA, multi-stage detection |

Most of the value comes from the T0+T1+T2 layers. T3 is reserved for finalists or for the user who pays for it.

### 3.2 Per-image metrics

#### 3.2.1 Technical quality (T0–T1)

| Metric | Algorithm | Library | Notes |
|---|---|---|---|
| Resolution & aspect | Header read | Pillow / `PIL` | Reject < 1024 px on long side for SDXL/Flux training[7,8]. |
| File hash | MD5 / SHA1 | stdlib | Exact-duplicate detection. |
| Perceptual hash | dHash, pHash, aHash, wHash | `imagededup`[4] / `ImageHash` | Near-duplicate detection across re-encodes/resizes[12]. |
| Blur | Variance of Laplacian | `opencv-python` | Cheap and reliable. Threshold typically ~100 for natural images. |
| Exposure | Histogram percentiles | `opencv-python` / `numpy` | Detects under/over-exposure. |
| Noise | Wavelet / patch variance | `scikit-image` | Useful for high-ISO photos. |
| Compression artifacts | JPEG quality estimate | EXIF + `cv2` | Strong signal that an image was downloaded from social media. |
| Format / EXIF flags | Header | `Pillow` | Tell you whether HEIC, ICC profile, color depth, etc. |
| AI-generated detection | C2PA + image classifier | `pytorch-image-models`, `Hive AI`-style | Optional gate; you may *want* AI images, or specifically not. |

CleanVision[3] bundles many of these into one auditor and is the obvious dependency for the T0/T1 layer. It will detect "blurry, dark, light, low-information, oddly sized, duplicate, near-duplicate" out of the box.

#### 3.2.2 Aesthetic / perceptual quality (T1–T3)

| Metric | Type | Library | Notes |
|---|---|---|---|
| **LAION-Aesthetic V1/V2** | CLIP-MLP regressor | `aesthetic-predictor`[13], HuggingFace `simple-aesthetics-predictor`[14] | The classic 1–10 score; trained on SAC + AVA + LAION-Logos. Fast (T2), interpretable, but biased toward a specific Western "beautiful photo" aesthetic[15]. |
| **NIMA** | InceptionResNet, AVA-trained | `pyiqa`[6] | Predicts mean-opinion-score histogram. Older but robust. |
| **MUSIQ** | Multi-scale ViT NR-IQA | `pyiqa`[6] | Strong on natural-scene quality benchmarks. T2. |
| **MANIQA** | Multi-dim attention NR-IQA | `pyiqa`[6,16] | Best for GAN/AI-distorted content; NTIRE 2022 winner. |
| **CLIP-IQA / CLIP-IQA+** | Prompt-based on CLIP features | `pyiqa`[6] | "Good photo" vs "bad photo" prompt comparison; zero-shot, easy to extend (e.g., "well-lit", "sharp focus")[17]. |
| **TOPIQ** | Top-down semantic NR-IQA | `pyiqa`[6] | The author of `pyiqa`'s own model; modern SOTA. |
| **Q-Align / DeQA-Score / VisualQuality-R1** | MLLM-based | HuggingFace | T3 — best alignment with human judgment but slow and large[18,19]. |

**Practical recommendation:** run *one* aesthetic head and *one* technical NR-IQA head. The combo of LAION-Aesthetic V2 + a `pyiqa` model (MUSIQ or TOPIQ) is a strong default. Use Q-Align only for the final tie-breaking on the kept set.

#### 3.2.3 Person-specific (T1–T3)

| Metric | Library | What it gives you |
|---|---|---|
| Face detection / count | `mediapipe`, `insightface`[20] (RetinaFace), `ultralytics` YOLO | Number and bounding boxes of faces |
| Face-area fraction | derived | Reject < 10% if you want a portrait-LoRA |
| Head pose (yaw/pitch/roll) | `6DRepNet`[21,22], `WHENet`[23] | Coverage on the pose sphere |
| Facial landmarks | MediaPipe FaceMesh, dlib, `face-alignment` | Eye openness, mouth openness, expression coverage |
| Identity embedding | InsightFace ArcFace[20,24] | Verify all images are the same person; rank-by-similarity to a chosen anchor |
| **Face Image Quality** | SDD-FIQA[25], CR-FIQA[26], CLIB-FIQA[27] | Recognizability score — different from aesthetic score; learned from face-recognition success/failure |
| Occlusion | Segmentation + heuristics | Sunglasses, hand on face, crops cutting off chin |
| Expression | FER models, MediaPipe Tasks | Coverage of neutral / smile / serious |
| Gaze direction | `MPIIFaceGaze`-style | Mostly nice-to-have for character LoRAs |

The killer person-LoRA quality measure is **FIQA**, not generic NR-IQA. SDD-FIQA[25] and CR-FIQA[26] specifically estimate the *recognizability* of a face given a recognition model — which is exactly what a character-LoRA cares about. None of the generic IQA toolkits include FIQA, so this is a wrapper layer the package needs to provide itself.

#### 3.2.4 Object / scene-specific (T1–T3)

| Metric | Library | Purpose |
|---|---|---|
| Object detection | `ultralytics` YOLO, `supervision` | Confirm subject is present, count instances, get bbox |
| Open-vocab detection | OWL-ViT, GroundingDINO | Detect by text prompt — generalizes the package to any concept |
| Segmentation mask | SAM, SAM2 | Subject-vs-background separation; useful for composing future training |
| Depth / scene structure | DepthAnything-v2 | Differentiates close-up vs wide-shot |
| Scene classification | Places365, CLIP zero-shot | Coverage of indoor/outdoor/etc. |

#### 3.2.5 Semantic embeddings (T2)

The two embedding spaces worth supporting from day one:

- **CLIP (ViT-L/14 or ViT-H/14)** — best for *semantic* similarity (same concept, different style). Required for CLIP-IQA, LAION-aesthetic, and zero-shot prompt-based attribute detection.
- **DINOv2 (ViT-B/14 or ViT-L/14)** — best for *visual* similarity (same scene/object, robust to style). Voxel51's benchmarks show DINOv2 outperforming CLIP on classification by 5–28 percentage points across natural-image datasets[28].

For person identity specifically, **ArcFace embeddings via InsightFace are mandatory** — neither CLIP nor DINOv2 can reliably tell you "this is the same person."

The right architectural choice is to allow **multiple embedding spaces simultaneously** for different metrics: ArcFace for identity-clustering, DINOv2 for redundancy-detection, CLIP for aesthetic-prompt-scoring.

### 3.3 Set-level metrics

This is where most of the leverage lives. A set-level metric takes the full collection of (per-image features, embeddings, attributes) and emits either a single quality number or a structured diagnosis.

| Metric | Algorithm | Library | What it tells you |
|---|---|---|---|
| **Pairwise diversity** | Mean / min pairwise embedding distance | numpy | Crude but useful baseline |
| **Cluster coverage** | k-means or HDBSCAN on embeddings, count populated clusters | `scikit-learn` | "You filled 12/15 visual clusters" |
| **Attribute bin coverage** | Histogram over discrete attribute bins (pose buckets, lighting buckets) | numpy | The *interpretable* version of diversity |
| **Facility-location score** | $f(S) = \sum_{y \in V} \max_{x \in S} \phi(x,y)$ | `apricot`[5,10] | How well the set "represents" the candidate pool |
| **Feature-based concave** | $f(S) = \sum_d w_d \phi(\sum_{x \in S} m_d(x))$ | `apricot`[5] | Encourages saturation across feature dimensions; scales to millions |
| **Graph-cut / saturated coverage** | submodular variants | `apricot`[5] | Balances representativeness with redundancy avoidance |
| **k-DPP score** | Determinant of kernel submatrix | `DPPy`[29], `dppy` | Probabilistic diversity, principled but heavier |
| **Identity uniformity** | Variance of ArcFace embeddings | derived | All images same person? (For person LoRA) |
| **Style consistency** | Variance of "style tokens" (Gram matrices, low-level CLIP features) | derived | All images same style? (For style LoRA) |
| **Redundancy count** | Pairs within near-duplicate threshold | `imagededup`[4] / FiftyOne Brain[1,30] | Set-level dedup |
| **Class balance** | Per-class counts vs target distribution | derived | For multi-class subjects (multiple outfits, expressions) |

The **submodular maximization framework** is the right unifier for selection[5,10,11,31]. A submodular function $f$ has the diminishing-returns property — adding a new image gives a smaller marginal gain when the set is already large or already covers similar content. The greedy algorithm gives a $(1 - 1/e) \approx 0.63$ approximation guarantee[10], which is far better than top-K and computationally cheap. SubZeroCore[31] is the most recent (Sept 2025) training-free submodular coreset method and is worth tracking for v2.

### 3.4 What metric to use when

Three rules of thumb that should drive the orchestrator's defaults:

1. **Cheaper before more expensive, and earlier metrics gate later metrics.** Don't run face quality on a blurry image. Don't run aesthetic on a duplicate.
2. **Set-level metrics consume per-image metrics, not the reverse.** Diversity is computed over embeddings; embeddings are an upstream per-image metric.
3. **The same per-image score serves multiple set objectives.** Don't recompute. Cache against `(image_id, metric_id, config_hash)`.

---

## 4. Selection Logic — Picking K from N

### 4.1 The four selection paradigms

| Paradigm | When it wins |
|---|---|
| **Threshold filter** | Cleaning out the obviously bad — "blur > 100, face_area > 0.1" |
| **Top-K by score** | When the set objective is genuinely additive, e.g. picking 5 hero shots |
| **Submodular subset selection** | The default for "best K from N" when diversity matters[5,10,11] |
| **Constrained / quota selection** | "5 close-ups, 5 medium, 5 wide, 3 profile, 2 from-behind" — bins-with-quotas |

Most real LoRA curation jobs are a **filter, then submodular**, with optional quotas layered on top.

### 4.2 The orchestration recipe

The proposed default pipeline (the funnel) for a person LoRA:

```
[200 candidates]
  ↓ T0: read headers, drop < 1024 px, drop bad formats
  ↓ T1: pHash dedup → drop near-duplicates
  ↓ T1: blur, exposure → drop bottom 20%
  ↓ T1: face count == 1 (or person-of-interest detection)
  ↓ T2: ArcFace identity check → drop wrong-person
  ↓ T2: head-pose, face-area, FIQA → drop low recognizability
  ↓ T2: aesthetic + DINOv2 embedding (cache only)
  ↓ select: facility-location over DINOv2 ⊕ pose-bin-quota
  ↓ T3 (optional): Q-Align / MLLM critique on the kept K only
[20 chosen]
```

Notice three things about this design:

- **Every arrow is a separately swappable plugin.** None of the steps know about each other; they communicate via the manifest.
- **The selection step can take the full annotation set as input.** The selector chooses which annotations to weight; the user can overwrite weights without rerunning anything upstream.
- **The optional T3 layer is a finalist re-ranker, not a gate.** This is how you let users pay for "premium" curation without inflating cost for everyone.

### 4.3 The "scoring manifest" idea — formalized

The original sketch proposed a per-image JSON object. That is exactly the right primitive but it should be promoted to the type system:

```python
# Conceptual sketch — see §6 for the architectural shape
@dataclass(frozen=True)
class Annotation:
    image_id: str
    metric_id: str
    value: Any  # number, vector, dict, label
    config_hash: str  # which model / threshold produced this
    cost_tier: int
    timestamp: datetime


# The manifest is a Mapping over annotations
Manifest = MutableMapping[Tuple[image_id, metric_id], Annotation]
```

This is **structurally identical** to the standoff-annotation pattern in your existing annotation-systems work — an image-curation annotation is just a `(reference, metadata)` pair where the reference is an image rather than a time interval. The `dol` Mapping interface drops in directly. Selection algorithms and reporting both consume the same manifest.

This shared-abstraction win is why `lookbook` should not be designed as a one-off; it should sit on top of the same conceptual layer that powers your annotation editor and your scene/animation pipeline.

---

## 5. Architecture

### 5.1 Layering

Five layers, each thinner than the next:

```
┌──────────────────────────────────────────────────────┐
│  Interface layer (CLI, py2mcp, http, lib)            │  ← argh dispatch, MCP server
├──────────────────────────────────────────────────────┤
│  Recipe / facade layer                               │  ← lookbook.curate(...)
│   - Subject profiles (person, product, scene, ...)   │
│   - Pipeline templates                               │
├──────────────────────────────────────────────────────┤
│  Orchestration layer                                 │  ← meshed DAG
│   - Pipeline composition                             │
│   - Caching, budget control, provenance              │
├──────────────────────────────────────────────────────┤
│  Plugin layer (the open-closed boundary)             │
│   - Scorers, Filters, Embedders, Selectors           │
│   - Each implements a tiny Protocol                  │
├──────────────────────────────────────────────────────┤
│  Backend layer (heavy dependencies, wrapped)         │
│   - CLIP, DINOv2, InsightFace, pyiqa, apricot        │
│   - Each behind a facade so it can be swapped        │
└──────────────────────────────────────────────────────┘
```

The Backend layer is where the package depends on heavy ML libraries; nothing above it should `import torch`. This is what keeps the package installable on a laptop for the planning/inspection phase even when the actual scoring will run on a remote GPU.

### 5.2 Core protocols

A tiny set of duck-typed interfaces (Python `Protocol`s) is the entire extension surface:

```python
from typing import Protocol, Iterable, Any, Mapping


class ImageRef(Protocol):
    """Anything that knows its identity and how to be opened lazily."""

    image_id: str

    def open(self) -> "PIL.Image.Image": ...

    metadata: Mapping[str, Any]


class Scorer(Protocol):
    metric_id: str
    cost_tier: int
    requires: tuple[str, ...]  # other metric_ids this depends on

    def score(self, ref: ImageRef, manifest: "Manifest") -> Any: ...


class Filter(Protocol):
    def keep(self, ref: ImageRef, manifest: "Manifest") -> bool: ...


class Embedder(Protocol):
    space_id: str

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

This is the **open-closed boundary**: anyone can add a new metric, new selector, new embedder by implementing one of these and registering it. The orchestrator never special-cases anything.

### 5.3 The manifest as SSOT

A single `MutableMapping`-style store holds every annotation ever computed for every image. Built on `dol`, this gives the package:

- **Pluggable persistence** — JSON file, SQLite, Parquet, S3 — by swapping the underlying store.
- **Lazy reads** — only load annotations actually needed by the current step.
- **Trivial caching** — re-running a pipeline picks up where it left off because the manifest is the cache.
- **Auditability** — every annotation carries provenance.

Selection algorithms consume the manifest read-only; scorers append; filters never delete. This SSOT discipline means the system is **inspectable at every step**, which is the right UX for both human users and LLM agents.

### 5.4 Subject profiles — the user-visible plugin point

A subject profile is a YAML/JSON/dataclass that bundles defaults:

```yaml
# profiles/person.yaml
name: person
default_pool_size: 200
target_size: 20

scorers:
  - blur:        { tier: 1, lib: opencv,        threshold: 100 }
  - phash:       { tier: 1, lib: imagededup }
  - face_count:  { tier: 1, lib: insightface,   require: 1 }
  - face_area:   { tier: 1, min: 0.1 }
  - identity:    { tier: 2, lib: insightface,   anchor: "auto" }
  - head_pose:   { tier: 2, lib: 6drepnet }
  - fiqa:        { tier: 2, lib: cr_fiqa }
  - aesthetic:   { tier: 2, lib: laion_v2 }
  - dinov2_emb:  { tier: 2, lib: dinov2_vitb14 }

selector:
  type: facility_location
  embedding: dinov2_emb
  weights: { aesthetic: 0.3, fiqa: 0.5, diversity: 1.0 }
  quotas:
    head_pose_yaw: { "<-30": 3, "-30..30": 8, ">30": 3 }
    face_area:     { "0.1..0.3": 5, ">0.3": 8 }
```

A new subject type — `product`, `scene`, `style`, `font`, `logo` — is a new YAML file. **No core code changes**. This is how you generalize beyond people.

### 5.5 Composition over inheritance, end to end

There are no deep class hierarchies anywhere. Concretely:

- A `Pipeline` is a list of `(Scorer | Filter | Embedder)` plus a final `Selector`.
- A `Scorer` is a function (or callable object) wearing the `Scorer` protocol.
- The composer (built on `meshed`) wires the dependency DAG: if `fiqa` declares `requires=("face_box",)`, the orchestrator runs `face_box` first.
- A `Profile` is a dataclass that, when applied, produces a `Pipeline`.

This means the package can be used as a library (assemble your own pipeline), via a profile (declarative), or via a high-level facade (`lookbook.curate(...)`).

### 5.6 Interface dispatch (your house style)

Following your `python-package-architecture` conventions, the package should expose four entry points to the same dispatch tree:

- **CLI** — `lookbook curate ./photos --profile person --k 20`
- **Python API** — `lookbook.curate(...)`
- **HTTP** — via `qh` for browser/agent calls
- **MCP** — via `py2mcp` so an LLM can reason about candidates and invoke `score`, `filter`, `select`, `diagnose` as primitive tools

The **MCP surface is the underrated win.** Splitting the verbs (§2.1) into individual MCP tools turns `lookbook` into something an agent can iterate against — score one image, ask the user, score another, propose a filter, run selection, show the result. That's a fundamentally better UX than "press button, wait, read report."

---

## 6. The Existing-Tools Landscape

This is the gap-analysis section. Short version: **lots of partial solutions, no opinionated end-to-end one for the personalization-training audience.**

### 6.1 General-purpose dataset-curation toolkits

| Tool | License | Focus | Why it isn't `lookbook` |
|---|---|---|---|
| **FiftyOne** + FiftyOne Brain[1,2,30,32] | Apache-2.0 | Generic CV dataset curation, exploration, deduplication, embeddings | Library + GUI for any dataset; not opinionated about LoRA training, no submodular set selection out of the box, no FIQA, no LoRA-export. **Should be a backend / view layer of `lookbook`, not a competitor.** |
| **CleanVision**[3] | AGPL-3.0 | Issue auditing (blur, dark, light, near-duplicate, low-info) | Excellent T0/T1 layer. AGPL is a license problem for redistribution; lookbook should integrate optionally. |
| **DeepCore** (research) | research | Coreset selection benchmarks | Benchmarks rather than a product. Useful for picking selectors. |

### 6.2 Single-purpose libraries that are actual dependencies

| Library | License | Purpose | Where it slots in |
|---|---|---|---|
| `imagededup`[4] | Apache-2.0 | Perceptual hash + CNN deduplication | T1 dedup |
| `pyiqa` (IQA-PyTorch)[6,16] | Various; check per-model | NR-IQA: BRISQUE, NIQE, NIMA, MUSIQ, MANIQA, TOPIQ, CLIP-IQA, ARNIQA | T2 quality |
| LAION `aesthetic-predictor`[13] | MIT | LAION Aesthetic V1/V2 | T2 aesthetic |
| `simple-aesthetics-predictor`[14] | MIT | HF wrapper of LAION-Aesthetic | T2 aesthetic |
| `apricot-select`[5,10] | MIT | Submodular subset selection (facility location, feature-based, graph-cut, sum-redundancy, saturated coverage) | Selection layer — **this is the workhorse** |
| `DPPy`[29,33] | MIT | Determinantal point processes | Selection layer (alternative) |
| `insightface`[20,24] | MIT (code) / non-comm. (pretrained models) | Face detection (RetinaFace), face recognition (ArcFace) | T1/T2 person metrics |
| `6DRepNet`[21,22] / `WHENet`[23] | research | Head pose | T2 person metrics |
| MediaPipe Face/Pose | Apache-2.0 | Lightweight face/pose | T1 person metrics |
| `ultralytics` YOLO / `supervision` | AGPL-3.0 / MIT | Object detection | T1/T2 object metrics |
| `dol` / `meshed` / `py2mcp` (yours) | various | Stores, DAGs, MCP serving | Architectural backbone |

License watchpoints worth flagging early: **CleanVision is AGPL** and **Ultralytics YOLO is AGPL**. If you intend to ship a commercial offering, prefer `pyiqa` + InsightFace's RetinaFace + apricot, all of which are MIT/permissive[4,5,6,20].

### 6.3 LoRA-adjacent tools

| Tool | What it does | Gap |
|---|---|---|
| **Kohya SS / sd-scripts**[34] | Full training stack with bucketing, captioning helpers | Curation is manual; expects you to bring a clean dataset |
| **kohya-colab / hollowstrawberry**[35] | Colab notebooks; integrates FiftyOne for manual curation | Manual curation only; no scoring/selection automation |
| **LoRA-Dataset-Automaker** (Maximax67)[36] | Notebook with face detection + CLIP similarity + FiftyOne app | Anime-character-specific; not a library; not productized |
| **klippbok** (alvdansen)[37] | Video LoRA dataset prep with CLIP triage | Video-only; opinionated for one studio's workflow |
| **lora-dataset-pipeline** (steegmueller)[38] | Instagram scrape → dedup → person filter → quality → upscale | Person-specific, brittle, no submodular selection |
| **Civitai community guides**[39] | WD14-tagger-based attribute filtering | Manual deletion workflow; depends on tagger quality |

The pattern is consistent: **practitioners stitch together 4-6 tools by hand for each project**. Nothing exists as an installable, agent-friendly, subject-agnostic curation library. That is `lookbook`'s wedge.

### 6.4 Commercial / hosted services

The hosted-LoRA training services (FAL, WaveSpeedAI, Anifusion, LlamaGen, Apatero, Z-Image, etc.[40,41,42,9,7]) all accept a ZIP of images and train. **None of them help you build the ZIP.** They explicitly tell users that dataset quality is the dominant factor and then leave that work to the user. There is a clear opening for either:

- A **hosted curation API** that produces ZIPs ready for these services (sell to end users), or
- A **white-label curation library** that those services embed as a pre-step (sell to the platforms themselves).

The library-first approach is cleaner architecturally — and it's the natural endgame of the package you'd be writing anyway.

### 6.5 The commercialization read

The gap is real. Specifically:

1. **Open-source landscape is fragmented and DIY.** There is no "scikit-learn of LoRA dataset curation."
2. **Commercial training platforms have outsourced this problem to their users.** They'd rather not, but no off-the-shelf solution exists.
3. **The problem is becoming more important, not less.** As consumer-grade LoRA training spreads, "I have a phone roll of my face — make me a LoRA" is the obvious workflow, and a curation layer is what makes it work.
4. **The agent-first interface (MCP) is unoccupied.** Existing tools assume a human in a GUI. An MCP-shaped curation tool that an LLM agent can call is differentiated.

The defensible product, IMO, is *not* "a curation script" — it's "a curation library with strong defaults per subject type, plus a scoring manifest format that becomes the lingua franca." The format and the recipes are the moat.

---

## 7. Concrete Implementation Plan (for the coding agent)

### 7.1 Recommended dependencies

Permissive-license-only initial slice:

```toml
# pyproject.toml — runtime deps
[project]
dependencies = [
  "Pillow>=10",
  "numpy>=1.26",
  "opencv-python-headless>=4.9",
  "scikit-image>=0.22",
  "scikit-learn>=1.4",
  "imagededup>=0.3.2",     # Apache-2.0; perceptual hash + CNN dedup
  "apricot-select>=0.6.1", # MIT; submodular selection
  "pyiqa>=0.1.13",         # MIT; pyiqa toolbox (NR-IQA + aesthetic)
  "torch>=2.1",
  "open-clip-torch>=2.24", # MIT; CLIP / SigLIP
  "transformers>=4.40",    # for DINOv2, MLLM IQA models
  "dol",                   # your store abstractions
  "meshed",                # your DAG composition
  "argh>=0.30",            # CLI dispatch
  "pydantic>=2.6",         # profile / config validation
]

[project.optional-dependencies]
person = [
  "insightface>=0.7.3",    # MIT code, ArcFace + RetinaFace
  "sixdrepnet>=0.1.6",     # head pose
  "mediapipe>=0.10",       # Apache-2.0
]
heavy = [
  "fiftyone>=0.24",        # optional GUI / vector index
]
```

### 7.2 Suggested package layout

```
lookbook/
  __init__.py            # re-exports the public facade
  base.py                # Protocol definitions + Annotation/Manifest types
  __main__.py            # argh CLI dispatch (your house pattern)
  manifest.py            # MutableMapping store, persistence
  refs.py                # ImageRef implementations (path, url, bytes, in-memory)
  pipeline.py            # Pipeline orchestrator (built on meshed)
  budget.py              # cost accounting, dynamic skip logic
  report.py              # diagnostic / export
  scorers/
    __init__.py          # registry
    technical.py         # blur, exposure, dedup, resolution
    aesthetic.py         # LAION, NIMA, MUSIQ, TOPIQ, CLIP-IQA wrappers
    embeddings.py        # CLIP / DINOv2 / ArcFace embedders
    person.py            # face detection, FIQA, head pose, identity
    object.py            # YOLO / open-vocab detection, viewpoint
    scene.py             # Places, depth, lighting
  selectors/
    __init__.py
    threshold.py
    topk.py
    submodular.py        # apricot wrapper (facility loc, feature-based, mixed)
    constrained.py       # quota-aware
    dpp.py               # k-DPP via DPPy
  profiles/
    __init__.py
    person.yaml
    product.yaml
    scene.yaml
    style.yaml
  io/
    ingest.py            # directory, zip, URL list, cloud
    export.py            # kohya-style folder, repeats, captions
  ui/                    # optional, can be deferred
    fiftyone_view.py     # surface manifest in FiftyOne
  mcp.py                 # py2mcp surface
  http.py                # qh surface
```

### 7.3 Phasing

I'd suggest a four-phase build that always keeps something usable shipping:

**Phase 0 — Skeleton (1 week).** Manifest, ImageRef, Pipeline, registry, CLI dispatch, JSON store. No real metrics yet — just a `random_score` plugin so the orchestration can be tested end-to-end.

**Phase 1 — Cheap funnel (1–2 weeks).** Resolution, dedup (via `imagededup`), blur, exposure. Threshold filter + top-K selector. Already useful as a "clean my photo dump" tool.

**Phase 2 — Embedding + submodular (2 weeks).** CLIP and DINOv2 embedders. `apricot`-based facility-location selector. Diversity diagnosis. This is the version that does the "200 → 20" headline workflow generically (no person-specifics yet).

**Phase 3 — Person profile (2–3 weeks).** InsightFace integration, ArcFace identity, FIQA (CR-FIQA wrapper), 6DRepNet head pose. Full person profile + quota selector. This is the version that pulls ahead of everything currently on GitHub.

**Phase 4 — Surfaces (1 week each).** MCP via `py2mcp`. HTTP via `qh`. Optional FiftyOne view layer for visual inspection. Profile templates for product / scene / style.

### 7.4 Things to **not** build (yet)

- **Don't build a GUI from scratch.** FiftyOne already has the right one; surface the manifest in it as a view.
- **Don't reimplement deduplication.** `imagededup` and FiftyOne Brain already do this well.
- **Don't reimplement IQA.** Wrap `pyiqa`. The wrapping is the work.
- **Don't reimplement clustering.** `scikit-learn` HDBSCAN/KMeans + `apricot` cover 95% of cases; `DPPy` covers the rest.
- **Don't try to be a captioning tool.** That's a separate concern downstream of curation; let kohya / Florence / WD14 handle it.
- **Don't build LoRA training.** This is curation only — explicit non-goal.

---

## 8. Questions Worth a Deeper Research Pass

I have enough material here to design the package and pick the libraries. The following are places where I think a focused **deep research** prompt would meaningfully tighten the design before you commit code:

1. **Empirical comparison of NR-IQA models for the specific use case of LoRA training-set screening.** Generic IQA benchmarks (KonIQ, LIVE) don't directly tell you which model best predicts "this image will hurt LoRA training." A small empirical study on a few open person-LoRA datasets would settle MUSIQ vs MANIQA vs TOPIQ vs CLIP-IQA vs Q-Align for your stack.
2. **Per-subject-type metric ontology.** What attributes should the *coverage* metric stratify on for "product" vs "scene" vs "style"? The literature exists in fragments — viewpoint sphere papers, lighting basis sets, scene-categorization taxonomies — but there's no consolidated "subject-type playbook" anywhere I can find.
3. **The face-similarity / over-similarity tradeoff for character LoRAs.** ArcFace-similarity is not a monotone good — too-similar images hurt training (they're effectively duplicates in identity space) and too-dissimilar images aren't the same person. The practitioner literature is full of intuitions but no clean curve. Worth a structured study.
4. **License/commercial-use deep dive on the model zoo.** InsightFace's *code* is MIT, but its *pretrained weights* are non-commercial[20]. SDD-FIQA, CR-FIQA, MUSIQ, MANIQA each have their own license posture. A cleared "permissive baseline" stack is necessary if you want to ship commercially.
5. **Active-curation / human-in-the-loop loop design.** For an agent UX, what's the ideal "show me 5 borderline images and I'll decide" interaction? There's good active-learning literature (margin sampling, BALD, SIMILAR/SCMI[40]) but none directly applied to LoRA dataset curation.

Any of these would benefit from being written up as a deep-research prompt the same way you've structured your prior reports.

---

## 9. References

[1] Voxel51. *FiftyOne Brain — image deduplication and similarity*. Version 1.14.1 documentation. [https://docs.voxel51.com/brain.html](https://docs.voxel51.com/brain.html).

[2] Voxel51. *Image Deduplication with FiftyOne — recipe*. [https://voxel51.com/fiftyone/workflows/deduplication/](https://voxel51.com/fiftyone/workflows/deduplication/).

[3] Cleanlab. *CleanVision: Audit your image data for better computer vision*. [https://cleanlab.ai/blog/learn/cleanvision/](https://cleanlab.ai/blog/learn/cleanvision/). GitHub: [https://github.com/cleanlab/cleanvision](https://github.com/cleanlab/cleanvision).

[4] Idealo. *imagededup — Finding duplicate images made easy*. [https://idealo.github.io/imagededup/](https://idealo.github.io/imagededup/). GitHub: [https://github.com/idealo/imagededup](https://github.com/idealo/imagededup).

[5] Schreiber J. *apricot: Submodular selection for data summarization in Python*. JMLR 21(161):1-6, 2020. [https://www.jmlr.org/papers/v21/19-467.html](https://www.jmlr.org/papers/v21/19-467.html). GitHub: [https://github.com/jmschrei/apricot](https://github.com/jmschrei/apricot).

[6] Chen C, Mo J, et al. *IQA-PyTorch / pyiqa — PyTorch toolbox for image quality assessment*. [https://github.com/chaofengc/IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch); docs at [https://iqa-pytorch.readthedocs.io/](https://iqa-pytorch.readthedocs.io/).

[7] Apatero. *LoRA Training Best Practices Guide 2025 — Flux and Stable Diffusion*. December 2025. [https://apatero.com/blog/lora-training-best-practices-flux-stable-diffusion-2025](https://apatero.com/blog/lora-training-best-practices-flux-stable-diffusion-2025).

[8] Segmind. *Easy Flux LoRA Training Guide for Beginners in 2026*. December 2025. [https://blog.segmind.com/easy-flux-lora-training-guide/](https://blog.segmind.com/easy-flux-lora-training-guide/).

[9] LlamaGen.ai. *AI LoRA Training — Train Consistent Characters*. [https://llamagen.ai/features/lora-training](https://llamagen.ai/features/lora-training).

[10] Schreiber J, Bilmes J, Noble WS. *apricot: Submodular selection for data summarization in Python*. arXiv:1906.03543, 2019. [https://arxiv.org/abs/1906.03543](https://arxiv.org/abs/1906.03543).

[11] Lee S, Kim Y. *Coreset Selection for Object Detection*. arXiv:2404.09161, 2024. [https://arxiv.org/abs/2404.09161](https://arxiv.org/abs/2404.09161).

[12] Hoyt B. *Duplicate image detection with perceptual hashing in Python*. [https://benhoyt.com/writings/duplicate-image-detection/](https://benhoyt.com/writings/duplicate-image-detection/).

[13] LAION. *aesthetic-predictor — A linear estimator on top of CLIP to predict the aesthetic quality of pictures*. [https://github.com/LAION-AI/aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor).

[14] *simple-aesthetics-predictor on PyPI*. [https://pypi.org/project/simple-aesthetics-predictor/](https://pypi.org/project/simple-aesthetics-predictor/).

[15] Taylor J, Agnew W, Sap M, Fox SE, Zhu H. *The Algorithmic Gaze of Image Quality Assessment: An Audit and Trace Ethnography of the LAION-Aesthetics Predictor*. FAccT '26. arXiv:2601.09896. [https://arxiv.org/pdf/2601.09896](https://arxiv.org/pdf/2601.09896).

[16] Yang S, Wu T, Shi S, et al. *MANIQA: Multi-dimension Attention Network for No-Reference Image Quality Assessment*. CVPR 2022. arXiv:2204.08958. [https://arxiv.org/pdf/2204.08958](https://arxiv.org/pdf/2204.08958).

[17] Wang J, Chan KCK, Loy CC. *Exploring CLIP for Assessing the Look and Feel of Images (CLIP-IQA)*. AAAI 2023.

[18] Wu H, et al. *Q-Align: Teaching LMMs for Visual Scoring via Discrete Text-Defined Levels*. arXiv:2312.17090, 2023.

[19] *Revisiting MLLM Based Image Quality Assessment: Errors and Remedy*. AAAI 2026. arXiv:2511.07812. [https://arxiv.org/pdf/2511.07812](https://arxiv.org/pdf/2511.07812).

[20] Deng J, Guo J, Niannan X, Zafeiriou S. *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*. CVPR 2019. InsightFace: [https://github.com/deepinsight/insightface](https://github.com/deepinsight/insightface).

[21] Hempel T, Abdelrahman AA, Al-Hamadi A. *6D Rotation Representation for Unconstrained Head Pose Estimation*. ICIP 2022. arXiv:2202.12555. [https://github.com/thohemp/6DRepNet](https://github.com/thohemp/6DRepNet).

[22] Yakhyo. *head-pose-estimation — Real-time head pose with ResNet/MobileNet backbones*. [https://github.com/yakhyo/head-pose-estimation](https://github.com/yakhyo/head-pose-estimation).

[23] Zhou Y, Gregson J. *WHENet: Real-time Fine-Grained Estimation for Wide Range Head Pose*. BMVC 2020. arXiv:2005.10353. [https://arxiv.org/pdf/2005.10353](https://arxiv.org/pdf/2005.10353).

[24] InsightFace. *Enterprise Face Recognition, Face Swap & Detection AI*. [https://www.insightface.ai/](https://www.insightface.ai/).

[25] Ou F-Z, Chen X, Zhang R, et al. *SDD-FIQA: Unsupervised Face Image Quality Assessment with Similarity Distribution Distance*. CVPR 2021. arXiv:2103.05977. [https://arxiv.org/pdf/2103.05977](https://arxiv.org/pdf/2103.05977).

[26] Boutros F, Fang M, Klemt M, Fu B, Damer N. *CR-FIQA: Face Image Quality Assessment by Learning Sample Relative Classifiability*. CVPR 2023. arXiv:2112.06592. [https://arxiv.org/pdf/2112.06592](https://arxiv.org/pdf/2112.06592).

[27] Ou F-Z, Li C, Wang S, Kwong S. *CLIB-FIQA: Face Image Quality Assessment with Confidence Calibration*. CVPR 2024. [https://openaccess.thecvf.com/content/CVPR2024/papers/Ou_CLIB-FIQA_Face_Image_Quality_Assessment_with_Confidence_Calibration_CVPR_2024_paper.pdf](https://openaccess.thecvf.com/content/CVPR2024/papers/Ou_CLIB-FIQA_Face_Image_Quality_Assessment_with_Confidence_Calibration_CVPR_2024_paper.pdf).

[28] Voxel51. *Finding the Best Embedding Model for Image Classification — DINOv2 vs CLIP vs ResNet*. November 2025. [https://voxel51.com/blog/finding-the-best-embedding-model-for-image-classification](https://voxel51.com/blog/finding-the-best-embedding-model-for-image-classification).

[29] Gautier G, Polito G, Bardenet R, Valko M. *DPPy: Sampling Determinantal Point Processes with Python*. JMLR MLOSS 2019. arXiv:1809.07258. [https://github.com/guilgautier/DPPy](https://github.com/guilgautier/DPPy).

[30] Voxel51. *FiftyOne Skills — fiftyone-dataset-curation*. [https://github.com/voxel51/fiftyone-skills/blob/main/skills/fiftyone-dataset-curation/SKILL.md](https://github.com/voxel51/fiftyone-skills/blob/main/skills/fiftyone-dataset-curation/SKILL.md).

[31] Moser B, et al. *SubZeroCore: A Submodular Approach with Zero Training for Coreset Selection*. arXiv:2509.21748, 2025. [https://arxiv.org/pdf/2509.21748](https://arxiv.org/pdf/2509.21748).

[32] Voxel51. *Double Trouble: Eliminate Image Duplicates with FiftyOne*. [https://voxel51.com/blog/eliminate-image-duplicates-with-fiftyone](https://voxel51.com/blog/eliminate-image-duplicates-with-fiftyone).

[33] Schreurs J, Fanuel M, Suykens JAK. *Towards Deterministic Diverse Subset Sampling*. arXiv:2105.13942, 2021. [https://arxiv.org/pdf/2105.13942](https://arxiv.org/pdf/2105.13942).

[34] bmaltais. *kohya_ss — GUI for Kohya's Stable Diffusion training scripts*. [https://github.com/bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss).

[35] hollowstrawberry. *kohya-colab — Accessible Google Colab notebooks for Stable Diffusion LoRA training*. [https://github.com/hollowstrawberry/kohya-colab](https://github.com/hollowstrawberry/kohya-colab).

[36] Maximax67. *LoRA-Dataset-Automaker*. [https://github.com/Maximax67/LoRA-Dataset-Automaker](https://github.com/Maximax67/LoRA-Dataset-Automaker).

[37] alvdansen. *klippbok — Video dataset curation for LoRA training*. [https://github.com/alvdansen/klippbok](https://github.com/alvdansen/klippbok).

[38] Steegmüller T. *lora-dataset-pipeline — Automated Instagram scraping and dataset preparation pipeline for LoRA*. [https://github.com/tim-steegmueller/lora-dataset-pipeline](https://github.com/tim-steegmueller/lora-dataset-pipeline).

[39] Civitai. *How to automate your picture selection for LoRA training*. [https://civitai.com/articles/3368/how-to-automate-your-picture-selection-for-lora-training](https://civitai.com/articles/3368/how-to-automate-your-picture-selection-for-lora-training).

[40] fal.ai. *Train FLUX LoRA Fast*. [https://fal.ai/models/fal-ai/flux-lora-fast-training](https://fal.ai/models/fal-ai/flux-lora-fast-training).

[41] WaveSpeedAI. *LoRA Training Tools*. [https://wavespeed.ai/collections/training-tools](https://wavespeed.ai/collections/training-tools).

[42] Anifusion. *AI model training & management — Create custom LoRA models*. [https://anifusion.ai/dashboard/models/](https://anifusion.ai/dashboard/models/).
