"""Tests for the cross-image identity / likeness scorer.

Everything here runs **offline, with no InsightFace / torch** — the scorer's
embedder is injectable, so we pass a fake embedder returning known vectors and
unit-test the cosine→[0,1] math, the reference-set aggregation, and the
threshold pass/fail. The one test that needs the real ArcFace model is gated
behind ``LOOKBOOK_TEST_MODELS=1`` (mirroring ``test_phase2_models.py`` /
``test_phase3_models.py``).
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

from lookbook import (
    IdentitySimilarity,
    SimilarityResult,
    compare_to_reference,
    registry,
)
from lookbook.scorers.identity import (
    DFLT_THRESHOLD,
    cosine_similarity,
    normalize_cosine,
)


# --------------------------------------------------------------------------- #
# Fake embedder — the dependency-injection seam under test
# --------------------------------------------------------------------------- #


class FakeEmbedder:
    """Maps an opaque key to a known vector. No deps, fully deterministic."""

    space_id = "fake"
    cost_tier = 0

    def __init__(self, mapping):
        self._m = {k: np.asarray(v, dtype=np.float64) for k, v in mapping.items()}

    def embed(self, key):
        return self._m[key]


# Canonical 3-D direction vectors.
V_SAME = np.array([1.0, 0.0, 0.0])
V_NEAR = np.array([0.94, 0.34, 0.0])  # ~20° off V_SAME → cosine ~0.94
V_ORTHO = np.array([0.0, 1.0, 0.0])  # cosine 0 with V_SAME
V_OPPOSITE = np.array([-1.0, 0.0, 0.0])  # cosine -1 with V_SAME


# --------------------------------------------------------------------------- #
# Registration / facade plumbing
# --------------------------------------------------------------------------- #


def test_identity_scorer_registered():
    assert "identity_similarity" in registry.scorers.names()


def test_facade_exports():
    import lookbook

    for name in ("IdentitySimilarity", "SimilarityResult", "compare_to_reference"):
        assert name in lookbook.__all__
        assert hasattr(lookbook, name)


# --------------------------------------------------------------------------- #
# Pure math: cosine
# --------------------------------------------------------------------------- #


def test_cosine_identical():
    assert cosine_similarity(V_SAME, V_SAME) == pytest.approx(1.0)


def test_cosine_orthogonal():
    assert cosine_similarity(V_SAME, V_ORTHO) == pytest.approx(0.0)


def test_cosine_opposite():
    assert cosine_similarity(V_SAME, V_OPPOSITE) == pytest.approx(-1.0)


def test_cosine_handles_unnormalized():
    # Magnitude must not matter.
    assert cosine_similarity(np.array([3.0, 0.0]), np.array([7.0, 0.0])) == pytest.approx(1.0)


def test_cosine_zero_vector_is_zero():
    # A no-face ArcFace embedding is a zero vector — must not blow up (no
    # divide-by-zero) and must read as "unrelated".
    assert cosine_similarity(np.zeros(3), V_SAME) == 0.0


# --------------------------------------------------------------------------- #
# Pure math: cosine → [0, 1]
# --------------------------------------------------------------------------- #


def test_normalize_rescale_maps_full_range():
    assert normalize_cosine(1.0) == pytest.approx(1.0)
    assert normalize_cosine(0.0) == pytest.approx(0.5)
    assert normalize_cosine(-1.0) == pytest.approx(0.0)


def test_normalize_clamp_floors_negatives():
    assert normalize_cosine(-0.4, normalization="clamp") == 0.0
    assert normalize_cosine(0.7, normalization="clamp") == pytest.approx(0.7)


def test_normalize_unknown_raises():
    with pytest.raises(ValueError):
        normalize_cosine(0.5, normalization="nope")


def test_normalized_score_in_unit_interval():
    for c in (-1.0, -0.3, 0.0, 0.42, 1.0):
        assert 0.0 <= normalize_cosine(c) <= 1.0


# --------------------------------------------------------------------------- #
# Single-reference comparison via the facade
# --------------------------------------------------------------------------- #


def test_compare_same_identity_scores_one_and_passes():
    emb = FakeEmbedder({"ref": V_SAME, "cand": V_SAME})
    r = compare_to_reference("ref", "cand", embedder=emb)
    assert isinstance(r, SimilarityResult)
    assert r.score == pytest.approx(1.0)
    assert r.identity_cosine == pytest.approx(1.0)
    assert r.passed is True
    assert r.threshold == DFLT_THRESHOLD
    assert r.n_references == 1


def test_compare_orthogonal_is_half_and_fails():
    emb = FakeEmbedder({"ref": V_SAME, "cand": V_ORTHO})
    r = compare_to_reference("ref", "cand", embedder=emb)
    assert r.score == pytest.approx(0.5)
    assert r.passed is False  # 0.5 < 0.85


def test_compare_opposite_is_zero():
    emb = FakeEmbedder({"ref": V_SAME, "cand": V_OPPOSITE})
    r = compare_to_reference("ref", "cand", embedder=emb)
    assert r.score == pytest.approx(0.0)
    assert r.passed is False


def test_threshold_gates_pass_fail():
    emb = FakeEmbedder({"ref": V_SAME, "cand": V_NEAR})
    # cosine ~0.94 → rescale ~0.97
    strict = compare_to_reference("ref", "cand", embedder=emb, threshold=0.99)
    lenient = compare_to_reference("ref", "cand", embedder=emb, threshold=0.90)
    assert strict.score == lenient.score  # same measurement
    assert strict.passed is False  # 0.97 < 0.99
    assert lenient.passed is True  # 0.97 >= 0.90


def test_clamp_normalization_path():
    emb = FakeEmbedder({"ref": V_SAME, "cand": V_OPPOSITE})
    r = compare_to_reference("ref", "cand", embedder=emb, normalization="clamp")
    # cosine -1 clamps to 0.
    assert r.score == 0.0


# --------------------------------------------------------------------------- #
# Reference-set aggregation
# --------------------------------------------------------------------------- #


def test_pool_max_matches_any_locked_view():
    # cand matches r1 exactly, is orthogonal to r2. max → ~1.0.
    emb = FakeEmbedder({"r1": V_SAME, "r2": V_ORTHO, "cand": V_SAME})
    r = compare_to_reference(["r1", "r2"], "cand", embedder=emb, aggregation="max")
    assert r.n_references == 2
    assert len(r.per_reference) == 2
    assert r.score == pytest.approx(1.0)  # max(1.0, 0.5)
    assert r.passed is True
    # identity_cosine tracks the selected (best) reference.
    assert r.identity_cosine == pytest.approx(1.0)


def test_pool_mean_is_stricter():
    emb = FakeEmbedder({"r1": V_SAME, "r2": V_ORTHO, "cand": V_SAME})
    r = compare_to_reference(["r1", "r2"], "cand", embedder=emb, aggregation="mean")
    # mean(1.0, 0.5) = 0.75
    assert r.score == pytest.approx(0.75)
    assert r.passed is False  # 0.75 < 0.85


def test_pool_min_is_strictest():
    emb = FakeEmbedder({"r1": V_SAME, "r2": V_ORTHO, "cand": V_SAME})
    r = compare_to_reference(["r1", "r2"], "cand", embedder=emb, aggregation="min")
    assert r.score == pytest.approx(0.5)  # min(1.0, 0.5)


def test_single_element_pool_equals_single_reference():
    emb = FakeEmbedder({"r1": V_SAME, "cand": V_NEAR})
    pooled = compare_to_reference(["r1"], "cand", embedder=emb)
    single = compare_to_reference("r1", "cand", embedder=emb)
    assert pooled.score == pytest.approx(single.score)
    assert pooled.n_references == 1


def test_unknown_aggregation_raises():
    emb = FakeEmbedder({"ref": V_SAME, "cand": V_SAME})
    with pytest.raises(ValueError):
        compare_to_reference("ref", "cand", embedder=emb, aggregation="median")


# --------------------------------------------------------------------------- #
# The IdentitySimilarity scorer object (construction + caching + protocol)
# --------------------------------------------------------------------------- #


def test_reference_embedded_once_not_per_candidate():
    calls = {"n": 0}

    class CountingEmbedder:
        space_id = "count"
        cost_tier = 0

        def embed(self, key):
            calls["n"] += 1
            return {"ref": V_SAME, "c1": V_SAME, "c2": V_NEAR}[key]

    emb = CountingEmbedder()
    scorer = IdentitySimilarity(reference_image="ref", embedder=emb)
    assert calls["n"] == 1  # reference embedded at construction
    scorer.compare("c1")
    scorer.compare("c2")
    assert calls["n"] == 3  # +1 per candidate, reference never re-embedded


def test_precomputed_reference_embedding_only_embeds_candidate():
    # When a reference vector is supplied directly, the embedder is only ever
    # called for candidates — never for the reference.
    seen = []

    class RecordingEmbedder:
        space_id = "rec"
        cost_tier = 0

        def embed(self, key):
            seen.append(key)
            return {"cand": V_SAME}[key]

    scorer = IdentitySimilarity(reference_embedding=V_SAME, embedder=RecordingEmbedder())
    assert seen == []  # reference came in precomputed; no embedding yet
    r = scorer.compare("cand")
    assert seen == ["cand"]  # only the candidate was embedded
    assert r.score == pytest.approx(1.0)


def test_precomputed_reference_pool_2d_array():
    # A 2-D array is treated as a reference pool.
    pool = np.vstack([V_SAME, V_ORTHO])
    emb = FakeEmbedder({"cand": V_SAME})
    scorer = IdentitySimilarity(reference_embedding=pool, embedder=emb, aggregation="max")
    r = scorer.compare("cand")
    assert r.n_references == 2
    assert r.score == pytest.approx(1.0)


def test_score_returns_jsonable_dict():
    emb = FakeEmbedder({"ref": V_SAME, "cand": V_SAME})
    scorer = IdentitySimilarity(reference_image="ref", embedder=emb)
    out = scorer.score("cand", {})  # manifest unused
    assert isinstance(out, dict)
    assert set(out) >= {
        "identity_cosine",
        "score",
        "passed",
        "threshold",
        "per_reference",
        "aggregation",
        "n_references",
    }
    assert out["score"] == pytest.approx(1.0)
    assert out["passed"] is True
    # JSON-serializable (no numpy scalars leaking through).
    import json

    json.loads(json.dumps(out))


def test_config_hash_changes_with_threshold_and_reference():
    emb = FakeEmbedder({"ref": V_SAME, "ref2": V_NEAR})
    a = IdentitySimilarity(reference_image="ref", embedder=emb, threshold=0.85)
    b = IdentitySimilarity(reference_image="ref", embedder=emb, threshold=0.90)
    c = IdentitySimilarity(reference_image="ref2", embedder=emb, threshold=0.85)
    assert a.config_hash != b.config_hash  # threshold changed
    assert a.config_hash != c.config_hash  # reference changed
    # Same config → stable hash.
    a2 = IdentitySimilarity(reference_image="ref", embedder=emb, threshold=0.85)
    assert a.config_hash == a2.config_hash


def test_requires_exactly_one_reference_input():
    with pytest.raises(ValueError):
        IdentitySimilarity(embedder=FakeEmbedder({}))  # none
    with pytest.raises(ValueError):
        IdentitySimilarity(  # two
            reference_embedding=V_SAME,
            reference_image="ref",
            embedder=FakeEmbedder({"ref": V_SAME}),
        )


def test_bad_embedder_spec_raises():
    from lookbook.scorers.identity import _resolve_embed_fn

    with pytest.raises(TypeError):
        _resolve_embed_fn(object())  # no .embed, not callable


# --------------------------------------------------------------------------- #
# Real-model smoke test (opt-in)
# --------------------------------------------------------------------------- #

_HAS_INSIGHTFACE = importlib.util.find_spec("insightface") is not None
SKIP_MODELS = os.environ.get("LOOKBOOK_TEST_MODELS") != "1" or not _HAS_INSIGHTFACE


@pytest.mark.skipif(
    SKIP_MODELS,
    reason="Set LOOKBOOK_TEST_MODELS=1 with insightface installed for the ArcFace smoke test.",
)
def test_arcface_self_similarity_smoke():
    """A face image compared to itself with the real ArcFace embedder should
    score ~1.0 and pass. Downloads buffalo_l on first run."""
    import io

    from PIL import Image

    from lookbook import BytesImageRef

    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (180, 140, 120)).save(buf, format="PNG")
    ref = BytesImageRef(payload=buf.getvalue(), image_id="face")
    r = compare_to_reference(ref, ref, embedder="arcface")
    # With a real face the cosine would be ~1.0; a synthetic flat image yields
    # a zero embedding → cosine 0 → score 0.5. Either way the call must work
    # and stay in [0, 1].
    assert 0.0 <= r.score <= 1.0
