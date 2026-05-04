"""Phase 1 tests — cheap funnel scorers, filters, and report.

These exercise the Phase 1 plugins against synthetic Pillow images so the
test suite stays fast and deterministic.
"""

from __future__ import annotations

import importlib.util
import io
import os

import numpy as np
import pytest
from PIL import Image, ImageFilter

# Tests that require imagehash / cv2 are skipped when those optional
# `[funnel]` deps aren't present. The core scorers (resolution, blur,
# exposure, file_hash) still run.
_HAS_IMAGEHASH = importlib.util.find_spec("imagehash") is not None
needs_imagehash = pytest.mark.skipif(
    not _HAS_IMAGEHASH,
    reason="imagehash not installed (lookbook[funnel])",
)

from lookbook import (
    BytesImageRef,
    Pipeline,
    curate,
    get_stores,
    registry,
    score,
)
from lookbook.filters.technical import (
    ExposureRange,
    MinBlur,
    MinResolution,
    NoExactDuplicate,
    NoNearDuplicate,
    fresh_filter,
)
from lookbook.manifest import value_of
from lookbook.report import Report, attribute_drops
from lookbook.scorers.technical import (
    Blur,
    Exposure,
    FileHash,
    PerceptualHash,
    Resolution,
    phash_distance,
)


# ---------------------------------------------------------------------------
# Image fixtures — built in memory so tests stay hermetic
# ---------------------------------------------------------------------------


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _gradient(w: int, h: int, seed: int = 0) -> Image.Image:
    """A noise + gradient image, deterministic via seed. Sharp by default."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    base = (x + y) / 2
    noise = rng.uniform(-30, 30, (h, w)).astype(np.float32)
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _solid(w: int, h: int, gray: int) -> Image.Image:
    arr = np.full((h, w, 3), gray, dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def sharp_ref():
    return BytesImageRef(payload=_png_bytes(_gradient(256, 256, seed=1)))


@pytest.fixture
def blurry_ref():
    img = _gradient(256, 256, seed=1).filter(ImageFilter.GaussianBlur(radius=8))
    return BytesImageRef(payload=_png_bytes(img))


@pytest.fixture
def black_ref():
    return BytesImageRef(payload=_png_bytes(_solid(64, 64, 0)))


@pytest.fixture
def white_ref():
    return BytesImageRef(payload=_png_bytes(_solid(64, 64, 255)))


@pytest.fixture
def memory_stores():
    return get_stores(
        images_store={}, manifest_store={}, runs_store={}, embeddings={},
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_phase1_scorers_registered():
    for name in ("resolution", "file_hash", "phash", "blur", "exposure"):
        assert name in registry.scorers.names(), name


def test_phase1_filters_registered():
    for name in (
        "min_resolution",
        "min_blur",
        "exposure_range",
        "no_exact_duplicate",
        "no_near_duplicate",
    ):
        assert name in registry.filters.names(), name


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_resolution_returns_dimensions(sharp_ref, memory_stores):
    Resolution().score(sharp_ref, memory_stores.manifest)
    v = score(sharp_ref, metric_id="resolution", stores=memory_stores)  # noqa: F841
    res = value_of(memory_stores.manifest, sharp_ref.image_id, "resolution")
    assert res["width"] == 256
    assert res["height"] == 256
    assert res["long_side"] == 256
    assert res["aspect_ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# File hash
# ---------------------------------------------------------------------------


def test_file_hash_is_sha1(sharp_ref):
    import hashlib

    expected = hashlib.sha1(sharp_ref.payload).hexdigest()
    got = FileHash().score(sharp_ref, {})
    assert got == expected


def test_file_hash_distinguishes_different_bytes(sharp_ref, blurry_ref):
    a = FileHash().score(sharp_ref, {})
    b = FileHash().score(blurry_ref, {})
    assert a != b


# ---------------------------------------------------------------------------
# Perceptual hash
# ---------------------------------------------------------------------------


@needs_imagehash
def test_phash_basic(sharp_ref):
    h = PerceptualHash().score(sharp_ref, {})
    assert isinstance(h, str) and len(h) > 0


@needs_imagehash
def test_phash_close_for_same_image(sharp_ref):
    """Re-encoding should yield the same (or very close) phash."""
    h1 = PerceptualHash().score(sharp_ref, {})
    # Re-encode through PNG to simulate trivial recompression.
    img = sharp_ref.open().convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    ref2 = BytesImageRef(payload=buf.getvalue())
    h2 = PerceptualHash().score(ref2, {})
    assert phash_distance(h1, h2) <= 5


def test_phash_distance_basic():
    assert phash_distance("ff", "ff") == 0
    assert phash_distance("ff", "00") == 8
    assert phash_distance("0f", "00") == 4


# ---------------------------------------------------------------------------
# Blur
# ---------------------------------------------------------------------------


def test_blur_higher_for_sharp_than_blurry(sharp_ref, blurry_ref):
    sharp = Blur().score(sharp_ref, {})
    blur = Blur().score(blurry_ref, {})
    assert sharp > blur


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------


def test_exposure_detects_under_and_over(black_ref, white_ref, sharp_ref):
    e_black = Exposure().score(black_ref, {})
    e_white = Exposure().score(white_ref, {})
    e_normal = Exposure().score(sharp_ref, {})
    assert e_black["frac_underexposed"] > 0.99
    assert e_white["frac_overexposed"] > 0.99
    assert e_normal["frac_underexposed"] < 0.5
    assert e_normal["frac_overexposed"] < 0.5


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_min_resolution_filter(sharp_ref, memory_stores):
    Resolution().score(sharp_ref, memory_stores.manifest)
    from lookbook.base import Annotation
    from lookbook.manifest import put_annotation
    put_annotation(
        memory_stores.manifest,
        Annotation(
            image_id=sharp_ref.image_id,
            metric_id="resolution",
            value={"long_side": 256, "width": 256, "height": 256},
        ),
    )
    assert MinResolution(min_long_side=128).keep(sharp_ref, memory_stores.manifest)
    assert not MinResolution(min_long_side=512).keep(sharp_ref, memory_stores.manifest)


def test_min_resolution_keeps_when_no_annotation(sharp_ref, memory_stores):
    """Filters never drop on missing evidence."""
    assert MinResolution().keep(sharp_ref, memory_stores.manifest)


def test_no_exact_duplicate(sharp_ref):
    """Two refs sharing the same payload share the same file_hash and the
    second one is dropped by NoExactDuplicate."""
    other = BytesImageRef(payload=sharp_ref.payload, image_id="other")
    manifest = {}
    from lookbook.base import Annotation
    from lookbook.manifest import put_annotation

    h = FileHash().score(sharp_ref, {})
    put_annotation(
        manifest,
        Annotation(image_id=sharp_ref.image_id, metric_id="file_hash", value=h),
    )
    put_annotation(
        manifest,
        Annotation(image_id=other.image_id, metric_id="file_hash", value=h),
    )
    f = NoExactDuplicate()
    assert f.keep(sharp_ref, manifest) is True
    assert f.keep(other, manifest) is False


@needs_imagehash
def test_no_near_duplicate(sharp_ref):
    near = sharp_ref.open().filter(ImageFilter.GaussianBlur(radius=0.5))
    near_ref = BytesImageRef(payload=_png_bytes(near.convert("RGB")))
    manifest = {}
    from lookbook.base import Annotation
    from lookbook.manifest import put_annotation

    for r in (sharp_ref, near_ref):
        put_annotation(
            manifest,
            Annotation(
                image_id=r.image_id,
                metric_id="phash",
                value=PerceptualHash().score(r, {}),
            ),
        )
    f = NoNearDuplicate(max_distance=5)
    assert f.keep(sharp_ref, manifest) is True
    assert f.keep(near_ref, manifest) is False


def test_fresh_filter_returns_independent_instances():
    a = fresh_filter("no_exact_duplicate")
    b = fresh_filter("no_exact_duplicate")
    assert a is not b
    assert a._seen is not b._seen


# ---------------------------------------------------------------------------
# Report and end-to-end
# ---------------------------------------------------------------------------


def test_attribute_drops_assigns_to_first_failing_filter():
    class AlwaysDrop:
        def __init__(self, name):
            self.name = name
        def keep(self, ref, manifest):
            return False

    refs = [BytesImageRef(payload=b"x", image_id=f"r{i}") for i in range(3)]
    f1 = AlwaysDrop("a")
    f2 = AlwaysDrop("b")
    survivors, drops = attribute_drops(refs, [f1, f2], {})
    assert survivors == []
    # All drops attributed to first filter (short-circuit).
    assert drops == {"a": 3}


@needs_imagehash
def test_curate_with_phase1_funnel(tmp_path):
    """End-to-end: a directory with one good + one duplicate + one tiny."""
    big_img = _gradient(800, 800, seed=42)
    big_img.save(tmp_path / "good.png")

    # Exact duplicate of "good": same bytes.
    (tmp_path / "good_copy.png").write_bytes((tmp_path / "good.png").read_bytes())

    # Way too small to keep at min_long_side=400.
    _gradient(100, 100, seed=42).save(tmp_path / "tiny.png")

    stores = get_stores(
        images_store={}, manifest_store={}, runs_store={}, embeddings={},
    )
    result = curate(
        str(tmp_path),
        k=10,
        scorer_ids=("resolution", "file_hash", "blur", "phash"),
        filter_ids=(
            ("min_resolution", {"min_long_side": 400}),
            "no_exact_duplicate",
            "no_near_duplicate",
        ),
        selector_id="top_k",
        stores=stores,
    )
    # Expect: tiny dropped by min_resolution; one of the duplicates dropped
    # by no_exact_duplicate (or no_near_duplicate, since identical → distance 0).
    assert result.report["n_kept"] == 1
    drops = result.report["dropped_by_filter"]
    assert drops, "expected at least one drop attributed to a filter"
    assert drops.get("min_resolution", 0) >= 1


def test_curate_with_phase1_funnel_uses_min_resolution_default(tmp_path):
    """Sanity: default MinResolution(1024) drops everything below that."""
    _gradient(500, 500, seed=1).save(tmp_path / "medium.png")
    stores = get_stores(
        images_store={}, manifest_store={}, runs_store={}, embeddings={},
    )
    result = curate(
        str(tmp_path),
        k=5,
        scorer_ids=("resolution",),
        filter_ids=("min_resolution",),
        selector_id="top_k",
        stores=stores,
    )
    assert result.report["n_kept"] == 0
    assert result.report["dropped_by_filter"]["min_resolution"] == 1
