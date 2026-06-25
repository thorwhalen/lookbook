"""Cross-image identity / likeness scorer — compare a candidate to a reference.

Every scorer in :mod:`lookbook.scorers` rates a *single* image's technical or
face quality. This module adds the missing primitive: a scorer that **compares**
a candidate image against a locked **reference** (or a pool of references) and
returns an identity-similarity number in ``[0, 1]`` (1 = same identity).

This is the measurement substrate for a "reference supervisor" — answering
*"does this generation still match the locked reference?"* with a number a
caller can gate on. It reuses lookbook's existing embedders unchanged:

- ``arcface`` (InsightFace ``buffalo_l``) for **face identity** — the default;
  neither CLIP nor DINOv2 reliably says "same person".
- ``clip`` / ``dinov2`` for **non-face subjects** (locations, architecture,
  props) — injected via the same ``embedder=`` seam.

Design notes
------------

- **The embedder is injected, never hard-wired.** :class:`IdentitySimilarity`
  accepts an :class:`~lookbook.base.Embedder`, a registry name (resolved
  lazily), or a bare ``embed_fn(ref) -> vector`` callable. This keeps the heavy
  ML deps (torch / insightface) behind lookbook's usual lazy boundary —
  importing this module pulls in nothing heavy — and makes the cosine math
  unit-testable with a fake embedder returning known vectors.

- **The reference is embedded once.** Pass a precomputed ``reference_embedding``
  (a single vector or a stacked array of vectors for a pool), or a reference
  image / list of images that the scorer embeds **once at construction** and
  caches. Candidates are never made to pay for re-embedding the reference.

- **Set aggregation** is explicit: when the reference is a pool, the per-view
  similarities are aggregated by ``aggregation`` (default ``"max"`` — "matches
  *any* locked view", the right default for a supervisor; ``"mean"`` is the
  stricter "matches the *average* view").

- The result carries a ``passed`` flag against an injectable ``threshold``.
  This is **advisory** — the consumer (e.g. reelee's reference supervisor)
  decides whether to hard-block. Nothing here raises on a low score.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from lookbook.base import ImageRef, Manifest
from lookbook.registry import scorers

# --------------------------------------------------------------------------- #
# Defaults (no magic numbers below this block)
# --------------------------------------------------------------------------- #

#: Default normalized-similarity threshold for the advisory pass/fail gate.
#: On the ``[0, 1]`` scale (1 = same identity). 0.85 mirrors the common
#: ArcFace "same person" operating point; owner-tunable per call.
DFLT_THRESHOLD: float = 0.85

#: Default embedder used when none is injected. ``"arcface"`` is the identity
#: space; swap for ``"clip"`` / ``"dinov2"`` for non-face subjects.
DFLT_EMBEDDER: str = "arcface"

#: How to fold a pool of per-reference similarities into one number.
DFLT_AGGREGATION: str = "max"

#: How to map a cosine in ``[-1, 1]`` onto ``[0, 1]``.
#: - ``"rescale"``: ``(cos + 1) / 2`` — order-preserving over the full range;
#:   the safe default for any embedder (CLIP/DINOv2 cosines can go negative).
#: - ``"clamp"``: ``max(0, cos)`` — for L2-normalized identity spaces where a
#:   negative cosine just means "unrelated" and should floor at 0.
DFLT_NORMALIZATION: str = "rescale"

#: The metric_id this scorer writes to the manifest.
METRIC_ID: str = "identity_similarity"


# --------------------------------------------------------------------------- #
# Pure math helpers (the heart of the unit tests)
# --------------------------------------------------------------------------- #

_AGGREGATIONS: dict[str, Callable[[np.ndarray], float]] = {
    "max": lambda a: float(np.max(a)),
    "mean": lambda a: float(np.mean(a)),
    "min": lambda a: float(np.min(a)),
}


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity in ``[-1, 1]`` between two vectors.

    Safe against unnormalized inputs and zero vectors (returns 0.0 when either
    has zero magnitude, which is what a no-face ArcFace embedding yields).

    >>> cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    1.0
    >>> cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0]))
    -1.0
    >>> cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    0.0
    >>> cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    0.0
    """
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def normalize_cosine(
    cosine: float, *, normalization: str = DFLT_NORMALIZATION
) -> float:
    """Map a cosine in ``[-1, 1]`` onto ``[0, 1]`` (1 = identical direction).

    >>> normalize_cosine(1.0)
    1.0
    >>> normalize_cosine(-1.0)
    0.0
    >>> normalize_cosine(0.0)
    0.5
    >>> normalize_cosine(-0.3, normalization="clamp")
    0.0
    >>> normalize_cosine(0.6, normalization="clamp")
    0.6
    """
    if normalization == "rescale":
        return float((cosine + 1.0) / 2.0)
    if normalization == "clamp":
        return float(max(0.0, cosine))
    raise ValueError(
        f"Unknown normalization {normalization!r}; "
        f"expected one of {{'rescale', 'clamp'}}."
    )


def _aggregate(
    values: Sequence[float], *, aggregation: str = DFLT_AGGREGATION
) -> float:
    """Fold per-reference similarities into one. See ``_AGGREGATIONS``."""
    if not len(values):  # type: ignore[arg-type]
        raise ValueError("cannot aggregate an empty similarity sequence")
    try:
        fn = _AGGREGATIONS[aggregation]
    except KeyError:
        raise ValueError(
            f"Unknown aggregation {aggregation!r}; "
            f"expected one of {sorted(_AGGREGATIONS)}."
        ) from None
    return fn(np.asarray(values, dtype=np.float64))


def _as_2d(embedding: Any) -> np.ndarray:
    """Coerce a single vector or a pool of vectors into a 2-D (n, d) array.

    Accepts a 1-D vector, a 2-D array, or a sequence of 1-D vectors.
    """
    arr = np.asarray(embedding, dtype=np.float64)
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"reference embedding must be 1-D or 2-D, got shape {arr.shape}.")


# --------------------------------------------------------------------------- #
# Embedder resolution — the injection seam
# --------------------------------------------------------------------------- #

# An embedder can be supplied three ways, none of which import torch here:
#   - an Embedder instance / object exposing ``.embed(ref)``
#   - a registry name like ``"arcface"`` (resolved lazily, on first use)
#   - a bare callable ``embed_fn(ref) -> vector``
EmbedderSpec = Union[str, "Embedder", Callable[[ImageRef], Any]]  # noqa: F821


def _resolve_embed_fn(embedder: EmbedderSpec) -> Callable[[ImageRef], np.ndarray]:
    """Return a ``embed(ref) -> ndarray`` callable from any accepted spec.

    Registry names are resolved here (lazily, so importing this module needs
    no heavy deps). Objects with an ``.embed`` method use it; bare callables
    are used as-is. This is the dependency-injection point the tests exploit
    by passing a fake embedder returning known vectors.
    """
    if isinstance(embedder, str):
        from lookbook.registry import embedders as _embedders

        embedder = _embedders.get(embedder)
    embed = getattr(embedder, "embed", None)
    if callable(embed):
        return embed
    if callable(embedder):
        return embedder
    raise TypeError(
        "embedder must be a registry name, an object with .embed(ref), "
        f"or a callable embed_fn(ref); got {type(embedder).__name__}."
    )


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SimilarityResult:
    """Outcome of comparing a candidate against a (possibly pooled) reference.

    Attributes
    ----------
    identity_cosine:
        The aggregated **raw cosine** in ``[-1, 1]`` (pre-normalization). The
        ``identity_cosine`` name is kept for the supervisor's vocabulary even
        when the embedder is CLIP/DINOv2 — it is the cosine in whatever space
        the embedder defines.
    score:
        The aggregated similarity normalized to ``[0, 1]`` (1 = same identity).
        This is what the gate compares against ``threshold``.
    passed:
        ``score >= threshold``. **Advisory only** — the consumer decides
        whether to act on it.
    threshold:
        The threshold used for ``passed`` (on the ``[0, 1]`` scale).
    per_reference:
        The per-reference normalized similarities before aggregation. Length
        is the size of the reference pool. Lets callers see *which* locked view
        matched (e.g. ``argmax`` for ``aggregation="max"``).
    aggregation:
        The aggregation applied over ``per_reference``.
    n_references:
        Size of the reference pool.
    """

    identity_cosine: float
    score: float
    passed: bool
    threshold: float
    per_reference: tuple[float, ...] = field(default_factory=tuple)
    aggregation: str = DFLT_AGGREGATION
    n_references: int = 0

    def as_dict(self) -> dict:
        """JSON-able view (what the scorer writes to the manifest)."""
        return {
            "identity_cosine": self.identity_cosine,
            "score": self.score,
            "passed": self.passed,
            "threshold": self.threshold,
            "per_reference": list(self.per_reference),
            "aggregation": self.aggregation,
            "n_references": self.n_references,
        }


def _build_result(
    candidate_vec: np.ndarray,
    reference_matrix: np.ndarray,
    *,
    threshold: float,
    aggregation: str,
    normalization: str,
) -> SimilarityResult:
    """Core comparison: candidate vector vs every reference vector → result."""
    per_ref_norm: list[float] = []
    per_ref_cos: list[float] = []
    for row in reference_matrix:
        c = cosine_similarity(candidate_vec, row)
        per_ref_cos.append(c)
        per_ref_norm.append(normalize_cosine(c, normalization=normalization))

    score = _aggregate(per_ref_norm, aggregation=aggregation)
    # Report the cosine of the same reference the aggregation selected, so
    # identity_cosine and score stay consistent (for max/min); for mean we
    # report the mean cosine.
    if aggregation == "mean":
        agg_cos = float(np.mean(per_ref_cos))
    else:
        idx = int(
            np.asarray(per_ref_norm).argmax()
            if aggregation == "max"
            else np.asarray(per_ref_norm).argmin()
        )
        agg_cos = per_ref_cos[idx]

    return SimilarityResult(
        identity_cosine=agg_cos,
        score=score,
        passed=score >= threshold,
        threshold=threshold,
        per_reference=tuple(per_ref_norm),
        aggregation=aggregation,
        n_references=int(reference_matrix.shape[0]),
    )


# --------------------------------------------------------------------------- #
# The scorer
# --------------------------------------------------------------------------- #


@dataclass
class IdentitySimilarity:
    """Score a candidate image's identity-likeness to a locked reference.

    Unlike the single-image scorers, this one is **stateful by reference**:
    it holds the reference embedding(s) and compares each candidate against
    them. Construct one per locked subject, then ``score`` every generation.

    Construction (provide exactly one of the reference inputs):

    - ``reference_embedding`` — a precomputed vector (1-D) or pool (2-D /
      sequence of vectors). Cheapest; the reference is never re-embedded.
    - ``reference_image`` — a single :class:`~lookbook.base.ImageRef` (or
      anything the injected embedder accepts), embedded **once** at
      construction and cached.
    - ``reference_images`` — a pool of refs, each embedded once.

    Parameters (keyword-only past the embedder seam):

    - ``embedder`` — registry name (``"arcface"`` default), an ``Embedder``
      object, or a bare ``embed_fn(ref) -> vector`` callable. Injectable so
      tests pass a fake embedder and non-face subjects pass ``"clip"`` /
      ``"dinov2"``.
    - ``threshold`` — advisory pass/fail cutoff on the ``[0, 1]`` scale.
    - ``aggregation`` — ``"max"`` (default) | ``"mean"`` | ``"min"`` over the
      reference pool.
    - ``normalization`` — cosine → ``[0, 1]`` map; see ``DFLT_NORMALIZATION``.

    The ``score(ref, manifest)`` method returns a JSON-able dict (see
    :meth:`SimilarityResult.as_dict`) so it persists through the manifest's
    default codec. Use :func:`compare_to_reference` for a one-call facade that
    returns the typed :class:`SimilarityResult`.
    """

    metric_id: str = METRIC_ID
    cost_tier: int = 2  # ArcFace is a T2 embedder; honest default.
    requires: tuple = ()
    backend: str = "lookbook:identity_similarity"

    # Reference inputs (provide one). Kept out of the public positional API.
    reference_embedding: Optional[Any] = None
    reference_image: Optional[Any] = None
    reference_images: Optional[Sequence[Any]] = None

    embedder: EmbedderSpec = DFLT_EMBEDDER
    threshold: float = DFLT_THRESHOLD
    aggregation: str = DFLT_AGGREGATION
    normalization: str = DFLT_NORMALIZATION

    def __post_init__(self):
        # Resolve the reference into a cached (n, d) matrix exactly once.
        self._reference_matrix = self._materialize_reference()
        if self.aggregation not in _AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation {self.aggregation!r}; "
                f"expected one of {sorted(_AGGREGATIONS)}."
            )

    # -- config_hash -------------------------------------------------------- #

    @property
    def config_hash(self) -> str:
        # Cache key must change if the reference, threshold, or any knob
        # changes — otherwise a stale candidate score would be reused.
        ref_digest = hashlib.sha1(
            np.ascontiguousarray(self._reference_matrix, dtype=np.float32).tobytes()
        ).hexdigest()[:12]
        emb_name = self.embedder if isinstance(self.embedder, str) else "injected"
        payload = repr(
            (
                emb_name,
                round(float(self.threshold), 6),
                self.aggregation,
                self.normalization,
                ref_digest,
            )
        ).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:12]

    # -- internals ---------------------------------------------------------- #

    def _materialize_reference(self) -> np.ndarray:
        provided = [
            x
            for x in (
                self.reference_embedding,
                self.reference_image,
                self.reference_images,
            )
            if x is not None
        ]
        if len(provided) != 1:
            raise ValueError(
                "IdentitySimilarity needs exactly one of "
                "`reference_embedding`, `reference_image`, or "
                f"`reference_images` (got {len(provided)})."
            )
        if self.reference_embedding is not None:
            return _as_2d(self.reference_embedding)

        embed = _resolve_embed_fn(self.embedder)
        if self.reference_image is not None:
            return _as_2d(np.asarray(embed(self.reference_image)))
        vecs = [np.asarray(embed(r)) for r in self.reference_images]  # type: ignore[union-attr]
        if not vecs:
            raise ValueError("`reference_images` was empty.")
        return np.vstack([v.ravel() for v in vecs])

    def _embed_candidate(self, ref: ImageRef) -> np.ndarray:
        return np.asarray(_resolve_embed_fn(self.embedder)(ref))

    # -- the Scorer protocol ------------------------------------------------ #

    def compare(self, ref: ImageRef) -> SimilarityResult:
        """Compare one candidate ref against the reference; typed result."""
        candidate_vec = self._embed_candidate(ref)
        return _build_result(
            candidate_vec,
            self._reference_matrix,
            threshold=self.threshold,
            aggregation=self.aggregation,
            normalization=self.normalization,
        )

    def score(self, ref: ImageRef, manifest: Manifest) -> dict:
        """Scorer-protocol entry point — returns the JSON-able result dict."""
        return self.compare(ref).as_dict()


# Registered so it's discoverable like every other scorer. The default
# instance carries a placeholder reference so registration never embeds
# anything; real use always supplies a reference via the facade or a
# ``(name, kwargs)`` override. We register a *factory-friendly* instance by
# giving it a trivial 1-vector reference; callers override it.
#
# NOTE: a scorer needs a concrete reference to be meaningful, so the registry
# entry is mostly a discoverability marker (it shows up in `list-plugins`).
scorers.register(
    METRIC_ID,
    IdentitySimilarity(reference_embedding=np.zeros(512, dtype=np.float32)),
)


# --------------------------------------------------------------------------- #
# One-call facade
# --------------------------------------------------------------------------- #


def compare_to_reference(
    reference_image: Any,
    candidate_image: Any,
    *,
    embedder: EmbedderSpec = DFLT_EMBEDDER,
    threshold: float = DFLT_THRESHOLD,
    aggregation: str = DFLT_AGGREGATION,
    normalization: str = DFLT_NORMALIZATION,
) -> SimilarityResult:
    """Compare a candidate to a reference in one call → :class:`SimilarityResult`.

    The supervisor's headline verb. ``reference_image`` may be a single image
    ref or a **pool** (any sequence of refs) — the pool is embedded once and
    folded by ``aggregation``. ``candidate_image`` is a single ref.

    All inputs are whatever the injected ``embedder`` accepts — typically an
    :class:`~lookbook.base.ImageRef`, but with a fake/callable embedder they
    can be anything (tests pass plain ids).

    Parameters
    ----------
    embedder:
        Registry name (``"arcface"`` default for faces; ``"clip"`` / ``"dinov2"``
        for scenes/architecture), an ``Embedder`` object, or a bare
        ``embed_fn(x) -> vector`` callable.
    threshold:
        Advisory pass/fail cutoff on the ``[0, 1]`` scale (1 = same identity).
    aggregation:
        How to fold a reference pool — ``"max"`` (default), ``"mean"``, ``"min"``.
    normalization:
        Cosine → ``[0, 1]`` map — ``"rescale"`` (default) or ``"clamp"``.

    Returns
    -------
    SimilarityResult
        ``.score`` (∈ ``[0, 1]``), ``.passed`` (advisory), ``.identity_cosine``
        (raw cosine), and the per-reference breakdown.
    """
    is_pool = isinstance(reference_image, (list, tuple))
    scorer = IdentitySimilarity(
        reference_images=list(reference_image) if is_pool else None,
        reference_image=None if is_pool else reference_image,
        embedder=embedder,
        threshold=threshold,
        aggregation=aggregation,
        normalization=normalization,
    )
    return scorer.compare(candidate_image)
