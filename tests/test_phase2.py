"""Phase 2 tests — embedders, facility-location selection, diagnosis.

These exercise the orchestration with the deterministic `MockEmbedder` so
no model weights are downloaded. The real `CLIPEmbedder` / `DINOv2Embedder`
implementations are smoke-tested in `test_phase2_models.py` and skipped by
default; opt in with `LOOKBOOK_TEST_MODELS=1`.
"""

from __future__ import annotations

import importlib.util
import io
import os

import numpy as np
import pytest
from PIL import Image

# `cluster_coverage` is a Phase 2 deliverable that depends on sklearn.
# When sklearn isn't installed (CI's bare-extras environment), we skip
# the diagnosis tests instead of failing them.
_HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
needs_sklearn = pytest.mark.skipif(
    not _HAS_SKLEARN, reason="sklearn not installed (lookbook[embed])",
)

from lookbook import (
    BytesImageRef,
    Pipeline,
    curate,
    get_stores,
    registry,
)
from lookbook.diagnose import cluster_coverage
from lookbook.embedders.mock import MockEmbedder
from lookbook.selectors.submodular import FacilityLocation


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _ref(idx: int, color=(0, 0, 0)) -> BytesImageRef:
    """Build a deterministic, distinguishable BytesImageRef."""
    img = Image.new("RGB", (32, 32), color)
    # Force unique payloads (so image_ids differ) by drawing one pixel.
    arr = np.asarray(img).copy()
    arr[0, 0] = (idx, idx, idx)
    img = Image.fromarray(arr)
    return BytesImageRef(payload=_png_bytes(img), image_id=f"r{idx:02d}")


@pytest.fixture
def memory_stores():
    return get_stores(
        images_store={}, manifest_store={}, runs_store={}, embeddings={},
    )


@pytest.fixture
def refs10():
    return [_ref(i, color=(i * 20, 0, 0)) for i in range(10)]


# ---------------------------------------------------------------------------
# Mock embedder
# ---------------------------------------------------------------------------


def test_mock_embedder_registered():
    assert "mock" in registry.embedders.names()


def test_mock_embedder_deterministic(refs10):
    e = MockEmbedder()
    v1 = e.embed(refs10[0])
    v2 = e.embed(refs10[0])
    assert np.array_equal(v1, v2)
    assert v1.shape == (64,)
    # L2-normalized.
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-5


def test_mock_embedder_distinguishes_images(refs10):
    e = MockEmbedder()
    vs = np.stack([e.embed(r) for r in refs10])
    sims = vs @ vs.T
    np.fill_diagonal(sims, 0)
    # Off-diagonal similarities should be substantially less than 1.
    assert sims.max() < 0.99


# ---------------------------------------------------------------------------
# Pipeline integration: embedders write to stores.embeddings
# ---------------------------------------------------------------------------


def test_pipeline_runs_embedders(refs10, memory_stores):
    p = Pipeline(
        embedders=[MockEmbedder()],
        selector=registry.selectors.get("top_k"),
    )
    result = p.run(refs10, k=5, stores=memory_stores)
    # Vectors persisted under the space_id.
    space = memory_stores.embeddings["mock"]
    assert len(space) == 10
    # Manifest carries the presence flag.
    for r in refs10:
        ann = memory_stores.manifest.get((r.image_id, "emb:mock"))
        assert ann is not None and ann.value is True
    # Selector worked (TopK uses random_score by default; values aren't set,
    # but the selector still returns up to K from len(survivors).
    assert len(result.kept) == 5


def test_pipeline_caches_embeddings_across_runs(refs10, memory_stores):
    # First run computes embeddings.
    Pipeline(
        embedders=[MockEmbedder(seed=7)],
        selector=registry.selectors.get("top_k"),
    ).run(refs10, k=5, stores=memory_stores)
    n_before = len(memory_stores.embeddings["mock"])
    assert n_before == 10

    # Mutate the stored vector to a sentinel; if the cache is honored, it
    # should NOT be overwritten on the second run.
    sentinel_id = refs10[0].image_id
    memory_stores.embeddings["mock"][sentinel_id] = [0.0] * 64

    Pipeline(
        embedders=[MockEmbedder(seed=7)],
        selector=registry.selectors.get("top_k"),
    ).run(refs10, k=5, stores=memory_stores)
    after = memory_stores.embeddings["mock"][sentinel_id]
    assert all(x == 0.0 for x in after), "embedder cache was not honored"


def test_changing_embedder_config_invalidates_cache(refs10, memory_stores):
    Pipeline(
        embedders=[MockEmbedder(seed=1)],
        selector=registry.selectors.get("top_k"),
    ).run(refs10, k=5, stores=memory_stores)
    sentinel_id = refs10[0].image_id
    memory_stores.embeddings["mock"][sentinel_id] = [0.0] * 64

    # Different seed => different config_hash => recomputed.
    Pipeline(
        embedders=[MockEmbedder(seed=2)],
        selector=registry.selectors.get("top_k"),
    ).run(refs10, k=5, stores=memory_stores)
    after = memory_stores.embeddings["mock"][sentinel_id]
    assert any(x != 0.0 for x in after), "config_hash change did not invalidate"


# ---------------------------------------------------------------------------
# FacilityLocation selector
# ---------------------------------------------------------------------------


def test_facility_location_registered():
    assert "facility_location" in registry.selectors.names()


def test_facility_location_picks_diverse_subset():
    """With strongly clustered embeddings, FL should pick one per cluster
    before duplicating any cluster."""
    rng = np.random.default_rng(0)
    # Three centers in 8-D; 5 points around each.
    centers = rng.standard_normal((3, 8)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    refs = []
    embs = {}
    idx = 0
    for c_idx, c in enumerate(centers):
        for _ in range(5):
            v = c + 0.05 * rng.standard_normal(8).astype(np.float32)
            v /= np.linalg.norm(v)
            r = BytesImageRef(payload=str(idx).encode(), image_id=f"img{idx:02d}")
            refs.append(r)
            embs[r.image_id] = v
            idx += 1

    sel = FacilityLocation(embedding_space="ignored", weight_quality=0.0)
    chosen = sel.select(
        refs, manifest={}, k=3, constraints={"embeddings": embs}
    )
    chosen_clusters = set()
    for r in chosen:
        # Match each chosen to its true cluster by max similarity to centers.
        sims = centers @ embs[r.image_id]
        chosen_clusters.add(int(np.argmax(sims)))
    assert len(chosen_clusters) == 3, "FL didn't pick one per cluster"


def test_facility_location_respects_quality_weight():
    """When weight_quality dominates, FL should pick the highest-quality
    images even if they are similar."""
    from lookbook.base import Annotation
    from lookbook.manifest import put_annotation

    rng = np.random.default_rng(0)
    refs = [BytesImageRef(payload=str(i).encode(), image_id=f"q{i}") for i in range(6)]
    # All near-identical embeddings => diversity gain is roughly equal.
    base = rng.standard_normal(8).astype(np.float32)
    base /= np.linalg.norm(base)
    embs = {}
    for r in refs:
        v = base + 0.001 * rng.standard_normal(8).astype(np.float32)
        v /= np.linalg.norm(v)
        embs[r.image_id] = v

    manifest = {}
    qualities = {f"q{i}": float(i) for i in range(6)}
    for image_id, q in qualities.items():
        put_annotation(
            manifest,
            Annotation(image_id=image_id, metric_id="quality", value=q),
        )
    sel = FacilityLocation(
        quality_metric_id="quality",
        weight_quality=10.0,
        weight_diversity=0.01,
    )
    chosen = sel.select(refs, manifest, k=3, constraints={"embeddings": embs})
    chosen_qs = sorted([qualities[r.image_id] for r in chosen], reverse=True)
    assert chosen_qs == [5.0, 4.0, 3.0], f"got {chosen_qs}"


def test_facility_location_caps_k_at_candidate_count():
    refs = [BytesImageRef(payload=str(i).encode(), image_id=f"c{i}") for i in range(3)]
    embs = {r.image_id: np.eye(3)[i].astype(np.float32) for i, r in enumerate(refs)}
    sel = FacilityLocation()
    out = sel.select(refs, {}, k=99, constraints={"embeddings": embs})
    assert len(out) == 3


def test_facility_location_empty_inputs():
    sel = FacilityLocation()
    assert sel.select([], {}, k=5, constraints={"embeddings": {}}) == []


def test_facility_location_raises_on_missing_embedding():
    refs = [BytesImageRef(payload=b"x", image_id="missing")]
    sel = FacilityLocation()
    with pytest.raises(ValueError, match="missing embeddings"):
        sel.select(refs, {}, k=1, constraints={"embeddings": {}})


# ---------------------------------------------------------------------------
# Cluster coverage
# ---------------------------------------------------------------------------


@needs_sklearn
def test_cluster_coverage_basic():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((4, 8)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    refs = []
    embs = {}
    for c_idx, c in enumerate(centers):
        for _ in range(5):
            v = c + 0.05 * rng.standard_normal(8).astype(np.float32)
            v /= np.linalg.norm(v)
            r = BytesImageRef(payload=str(len(refs)).encode(),
                              image_id=f"i{len(refs):02d}")
            refs.append(r)
            embs[r.image_id] = v
    # Keep only the first 2 (= one cluster fully missed when n=4).
    coverage = cluster_coverage(refs, refs[:2], embs, n_clusters=4)
    assert coverage["n_clusters"] == 4
    assert coverage["n_clusters_filled"] >= 1
    assert coverage["n_clusters_filled"] <= 2
    assert sum(coverage["cluster_sizes_kept"]) == 2
    assert sum(coverage["cluster_sizes_candidates"]) == 20


def test_cluster_coverage_handles_empty_inputs():
    out = cluster_coverage([], [], {}, n_clusters=4)
    assert out["n_clusters"] == 0
    assert out["n_clusters_filled"] == 0


# ---------------------------------------------------------------------------
# End-to-end via curate()
# ---------------------------------------------------------------------------


def test_curate_with_facility_location_via_mock(refs10, memory_stores):
    """End-to-end: random_score + mock embedder + facility-location selection."""
    result = curate(
        refs10,
        k=4,
        scorer_ids=("random_score",),
        embedder_ids=("mock",),
        selector_id=("facility_location",
                     {"embedding_space": "mock",
                      "quality_metric_id": "random_score",
                      "weight_quality": 0.5}),
        stores=memory_stores,
    )
    assert len(result.kept) == 4
    assert "mock" in result.report.get("notes", {}).get("embedder_ids", [])


@needs_sklearn
def test_curate_with_diagnosis(refs10, memory_stores):
    result = curate(
        refs10,
        k=4,
        scorer_ids=("random_score",),
        embedder_ids=("mock",),
        selector_id=("facility_location", {"embedding_space": "mock"}),
        diagnose_clusters=4,
        stores=memory_stores,
    )
    cov = result.report["notes"]["cluster_coverage"]
    assert cov["n_clusters"] == 4
    assert sum(cov["cluster_sizes_kept"]) == 4
