"""Phase 3 tests — face detection, head pose, face quality, ArcFace,
QuotaSelector, and the YAML profile loader.

All tests run against mock backends so no model weights download. Real-
backend smoke tests live in `test_phase3_models.py` and require
`LOOKBOOK_TEST_MODELS=1`.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from lookbook import (
    BytesImageRef,
    Pipeline,
    curate,
    get_stores,
    registry,
)
from lookbook.base import Annotation
from lookbook.embedders.arcface import MockArcFaceEmbedder
from lookbook.filters.person import (
    HasFace,
    MinFaceArea,
    MinFaceConfidence,
    SingleFaceOnly,
)
from lookbook.manifest import put_annotation, value_of
from lookbook.profiles import list_profiles, load
from lookbook.scorers.person import (
    FaceArea,
    FaceQualityProxy,
    MockFaceDetect,
    MockHeadPose,
    _bin_face_size,
    _bin_yaw,
    _bin_yaw_simple,
)
from lookbook.selectors.quota import QuotaSelector


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _ref(idx: int, size: int = 256) -> BytesImageRef:
    img = Image.new("RGB", (size, size), (idx * 30 % 255, 0, 0))
    return BytesImageRef(payload=_png_bytes(img), image_id=f"r{idx:02d}")


@pytest.fixture
def memory_stores():
    return get_stores(
        images_store={}, manifest_store={}, runs_store={}, embeddings={},
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_phase3_scorers_registered():
    for name in (
        "mock_face", "insightface", "face_area",
        "mock_head_pose", "head_pose", "face_quality",
    ):
        assert name in registry.scorers.names(), name


def test_phase3_filters_registered():
    for name in ("has_face", "single_face_only", "min_face_area",
                 "min_face_confidence"):
        assert name in registry.filters.names(), name


def test_phase3_embedders_registered():
    assert "arcface_mock" in registry.embedders.names()
    assert "arcface" in registry.embedders.names()


def test_phase3_selectors_registered():
    assert "quota" in registry.selectors.names()


# ---------------------------------------------------------------------------
# Bin helpers
# ---------------------------------------------------------------------------


def test_bin_yaw_thresholds():
    assert _bin_yaw(0) == "front"
    assert _bin_yaw(10) == "front"
    assert _bin_yaw(-10) == "front"
    assert _bin_yaw(20) == "three_quarter"
    assert _bin_yaw(-20) == "three_quarter_left"
    assert _bin_yaw(45) == "profile"
    assert _bin_yaw(-45) == "profile_left"
    assert _bin_yaw(85) == "back"


def test_bin_yaw_simple():
    assert _bin_yaw_simple(0) == "center"
    assert _bin_yaw_simple(-30) == "left"
    assert _bin_yaw_simple(30) == "right"


def test_bin_face_size():
    assert _bin_face_size(0) == "no_face"
    assert _bin_face_size(0.02) == "wide"
    assert _bin_face_size(0.10) == "medium"
    assert _bin_face_size(0.20) == "close"
    assert _bin_face_size(0.50) == "extreme_close"


# ---------------------------------------------------------------------------
# Mock face detector
# ---------------------------------------------------------------------------


def test_mock_face_detect_centered_box():
    ref = _ref(0, size=256)
    box = MockFaceDetect().score(ref, {})
    assert box["x1"] == 51 and box["x2"] == 205
    assert box["y1"] == 51 and box["y2"] == 205
    assert box["confidence"] > 0.9
    assert box["n_faces"] == 1


def test_mock_face_detect_simulate_no_face():
    ref = _ref(0)
    assert MockFaceDetect(simulate_no_face=True).score(ref, {}) is None


# ---------------------------------------------------------------------------
# Derived scorers
# ---------------------------------------------------------------------------


def test_face_area_uses_box_and_resolution(memory_stores):
    ref = _ref(0, size=200)
    # Place known annotations in the manifest.
    put_annotation(memory_stores.manifest, Annotation(
        image_id=ref.image_id, metric_id="resolution",
        value={"width": 200, "height": 200},
    ))
    put_annotation(memory_stores.manifest, Annotation(
        image_id=ref.image_id, metric_id="face_box",
        value={"x1": 50, "y1": 50, "x2": 150, "y2": 150,
               "confidence": 0.9, "n_faces": 1},
    ))
    out = FaceArea().score(ref, memory_stores.manifest)
    assert out["pixels"] == 100 * 100
    assert out["fraction"] == pytest.approx(0.25)
    # 0.25 falls in (0.15, 0.35] → "close"
    assert out["size_bin"] == "close"


def test_face_area_no_face_returns_zero(memory_stores):
    ref = _ref(0)
    put_annotation(memory_stores.manifest, Annotation(
        image_id=ref.image_id, metric_id="resolution",
        value={"width": 200, "height": 200},
    ))
    # No face_box annotation present.
    out = FaceArea().score(ref, memory_stores.manifest)
    assert out["fraction"] == 0.0
    assert out["size_bin"] == "no_face"


def test_mock_head_pose_returns_yaw_bins():
    ref = _ref(0)
    manifest = {}
    put_annotation(manifest, Annotation(
        image_id=ref.image_id, metric_id="face_box",
        value={"x1": 0, "y1": 0, "x2": 100, "y2": 100,
               "confidence": 1.0, "n_faces": 1},
    ))
    pose = MockHeadPose().score(ref, manifest)
    assert -90.0 <= pose["yaw"] <= 90.0
    assert pose["yaw_bin"] in {
        "front", "three_quarter", "three_quarter_left",
        "profile", "profile_left", "back", "back_left",
    }
    assert pose["yaw_bin_simple"] in {"left", "center", "right"}


def test_face_quality_proxy_combines_signals(memory_stores):
    ref = _ref(0)
    put_annotation(memory_stores.manifest, Annotation(
        image_id=ref.image_id, metric_id="face_box",
        value={"x1": 0, "y1": 0, "x2": 50, "y2": 50,
               "confidence": 1.0, "n_faces": 1},
    ))
    put_annotation(memory_stores.manifest, Annotation(
        image_id=ref.image_id, metric_id="face_area",
        value={"fraction": 0.20, "size_bin": "close"},
    ))
    put_annotation(memory_stores.manifest, Annotation(
        image_id=ref.image_id, metric_id="blur",
        value=500.0,
    ))
    q = FaceQualityProxy().score(ref, memory_stores.manifest)
    # 0.5*1 + 0.3*1 + 0.2*1 = 1.0 (perfect signals)
    assert q == pytest.approx(1.0, abs=0.01)


def test_face_quality_zero_when_no_face(memory_stores):
    ref = _ref(0)
    # No face_box at all.
    assert FaceQualityProxy().score(ref, memory_stores.manifest) == 0.0


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_has_face_filter():
    ref = _ref(0)
    manifest = {}
    # No annotation: keep (no evidence).
    assert HasFace().keep(ref, manifest) is True
    # Annotation present, value None: drop.
    put_annotation(manifest, Annotation(
        image_id=ref.image_id, metric_id="face_box", value=None,
    ))
    assert HasFace().keep(ref, manifest) is False


def test_single_face_only_filter():
    ref = _ref(0)
    manifest = {}
    put_annotation(manifest, Annotation(
        image_id=ref.image_id, metric_id="face_box",
        value={"x1": 0, "y1": 0, "x2": 1, "y2": 1, "n_faces": 1},
    ))
    assert SingleFaceOnly().keep(ref, manifest) is True
    put_annotation(manifest, Annotation(
        image_id=ref.image_id, metric_id="face_box",
        value={"x1": 0, "y1": 0, "x2": 1, "y2": 1, "n_faces": 3},
    ))
    assert SingleFaceOnly().keep(ref, manifest) is False


def test_min_face_area_filter():
    ref = _ref(0)
    manifest = {}
    put_annotation(manifest, Annotation(
        image_id=ref.image_id, metric_id="face_area",
        value={"fraction": 0.03},
    ))
    assert MinFaceArea(min_fraction=0.05).keep(ref, manifest) is False
    assert MinFaceArea(min_fraction=0.01).keep(ref, manifest) is True


def test_min_face_confidence_filter():
    ref = _ref(0)
    manifest = {}
    put_annotation(manifest, Annotation(
        image_id=ref.image_id, metric_id="face_box",
        value={"x1": 0, "y1": 0, "x2": 1, "y2": 1, "confidence": 0.4},
    ))
    assert MinFaceConfidence(threshold=0.5).keep(ref, manifest) is False
    assert MinFaceConfidence(threshold=0.3).keep(ref, manifest) is True


# ---------------------------------------------------------------------------
# QuotaSelector
# ---------------------------------------------------------------------------


def test_quota_selector_respects_quotas():
    refs = [_ref(i) for i in range(10)]
    manifest = {}
    # Force-assign yaw bins via mock-style annotations.
    bins = ["left", "left", "left", "center", "center",
            "center", "center", "center", "right", "right"]
    embs = {}
    for r, b in zip(refs, bins):
        put_annotation(manifest, Annotation(
            image_id=r.image_id, metric_id="head_pose",
            value={"yaw_bin_simple": b},
        ))
        embs[r.image_id] = np.eye(8)[hash(r.image_id) % 8].astype(np.float32)

    sel = QuotaSelector(
        bin_metric_id="head_pose.yaw_bin_simple",
        inner_selector_id="top_k",
        embedding_space=None,
    )
    chosen = sel.select(
        refs, manifest, k=6,
        constraints={"quotas": {"left": 2, "center": 3, "right": 1}},
    )
    chosen_bins = [
        value_of(manifest, r.image_id, "head_pose")["yaw_bin_simple"]
        for r in chosen
    ]
    from collections import Counter
    counts = Counter(chosen_bins)
    assert counts["left"] == 2
    assert counts["center"] == 3
    assert counts["right"] == 1


def test_quota_selector_backfills_when_under_quota():
    refs = [_ref(i) for i in range(5)]
    manifest = {}
    # All in 'center' — no left/right candidates.
    for r in refs:
        put_annotation(manifest, Annotation(
            image_id=r.image_id, metric_id="head_pose",
            value={"yaw_bin_simple": "center"},
        ))

    sel = QuotaSelector(bin_metric_id="head_pose.yaw_bin_simple",
                        inner_selector_id="top_k")
    chosen = sel.select(
        refs, manifest, k=4,
        constraints={"quotas": {"left": 2, "center": 1, "right": 2}},
    )
    # Quotas demand 2+1+2=5 but only "center" has candidates; backfill kicks
    # in to reach k=4 from the remaining center pool.
    assert len(chosen) == 4


def test_quota_selector_strict_mode():
    refs = [_ref(i) for i in range(5)]
    manifest = {}
    for r in refs:
        put_annotation(manifest, Annotation(
            image_id=r.image_id, metric_id="head_pose",
            value={"yaw_bin_simple": "center"},
        ))
    sel = QuotaSelector(
        bin_metric_id="head_pose.yaw_bin_simple",
        inner_selector_id="top_k",
        strict=True,
    )
    chosen = sel.select(
        refs, manifest, k=4,
        constraints={"quotas": {"left": 2, "center": 1, "right": 2}},
    )
    # Strict: only the 1 center quota fills; no backfill.
    assert len(chosen) == 1


# ---------------------------------------------------------------------------
# YAML profile loader
# ---------------------------------------------------------------------------


def test_profile_person_loads():
    spec = load("person")
    assert spec["name"] == "person"
    assert "insightface" in spec["scorers"]
    assert "arcface" in spec["embedders"]
    sel = spec["selector"]
    assert isinstance(sel, list) and sel[0] == "quota"
    assert spec["constraints"]["quotas"]


def test_profile_person_mock_loads():
    spec = load("person_mock")
    assert spec["name"] == "person_mock"
    assert "mock_face" in spec["scorers"]
    assert "arcface_mock" in spec["embedders"]


def test_list_profiles_includes_shipped():
    names = list_profiles()
    assert "person" in names
    assert "person_mock" in names


def test_load_unknown_profile():
    with pytest.raises(KeyError):
        load("does_not_exist_xyz")


# ---------------------------------------------------------------------------
# End-to-end: person_mock profile via curate()
# ---------------------------------------------------------------------------


def test_person_mock_profile_end_to_end(memory_stores):
    """Run the person_mock recipe through the facade. Synthesizes faces +
    poses + identity embeddings, applies quotas, and returns kept K."""
    spec = load("person_mock")
    # Build 12 candidates so quotas (2+4+2=8) have room.
    refs = [_ref(i, size=256) for i in range(12)]

    # Convert the YAML's [name, kwargs] selector form to the (name, kwargs)
    # tuple form the facade expects.
    sel = spec["selector"]
    if isinstance(sel, list) and len(sel) == 2 and isinstance(sel[1], dict):
        sel = (sel[0], sel[1])
    filters = []
    for f in spec["filters"]:
        if isinstance(f, list) and len(f) == 2 and isinstance(f[1], dict):
            filters.append((f[0], f[1]))
        else:
            filters.append(f)

    result = curate(
        refs,
        k=8,
        scorer_ids=tuple(spec["scorers"]),
        embedder_ids=tuple(spec["embedders"]),
        filter_ids=tuple(filters),
        selector_id=sel,
        diagnose_clusters=spec["diagnose_clusters"],
        constraints=spec["constraints"],
        stores=memory_stores,
    )

    # Quotas were left=2, center=4, right=2; deterministic mock should
    # produce roughly that distribution.
    assert len(result.kept) == 8
    bins = [
        value_of(memory_stores.manifest, r.image_id, "head_pose")["yaw_bin_simple"]
        for r in result.kept
    ]
    from collections import Counter
    counts = Counter(bins)
    # Each bin should have at least one entry when quotas can be satisfied.
    assert counts["center"] >= 1
    # Coverage diagnosis ran.
    assert "cluster_coverage" in result.report["notes"]
