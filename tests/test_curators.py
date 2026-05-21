"""Tests for the `technical_quality` scorer and the opinionated curate facades.

`curate_for_character` / `curate_for_environment` are exercised against
synthetic Pillow images so the suite stays fast, deterministic and free of
the ML face-detection dependency — the character facade is driven through
the `mock_face` detector here.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from lookbook import BytesImageRef, curate_for_character, curate_for_environment
from lookbook.base import Annotation
from lookbook.manifest import put_annotation
from lookbook.scorers.technical import TechnicalQuality


# ---------------------------------------------------------------------------
# Synthetic images
# ---------------------------------------------------------------------------


def _png_ref(arr: np.ndarray) -> BytesImageRef:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return BytesImageRef(payload=buf.getvalue())


def _sharp_ref(size: int = 512, square: int = 16) -> BytesImageRef:
    """A high-contrast checkerboard — many edges, high Laplacian variance,
    no exposure clipping (values stay inside [16, 240])."""
    arr = np.full((size, size, 3), 64, dtype=np.uint8)
    for i in range(0, size, square * 2):
        for j in range(0, size, square * 2):
            arr[i : i + square, j : j + square] = 192
            arr[i + square : i + square * 2, j + square : j + square * 2] = 192
    return _png_ref(arr)


def _blurred_ref(size: int = 512) -> BytesImageRef:
    """A flat mid-grey field — zero Laplacian variance (maximally soft)."""
    return _png_ref(np.full((size, size, 3), 128, dtype=np.uint8))


# ---------------------------------------------------------------------------
# technical_quality scorer
# ---------------------------------------------------------------------------


class _Ref:
    """Minimal stand-in — `value_of` only ever reads `.image_id`."""

    def __init__(self, image_id: str):
        self.image_id = image_id


def test_technical_quality_is_one_for_a_perfect_image():
    manifest: dict = {}
    img = "perfect"
    put_annotation(manifest, Annotation(image_id=img, metric_id="blur", value=1000.0))
    put_annotation(
        manifest,
        Annotation(
            image_id=img,
            metric_id="exposure",
            value={"frac_underexposed": 0.0, "frac_overexposed": 0.0},
        ),
    )
    put_annotation(
        manifest,
        Annotation(image_id=img, metric_id="resolution", value={"long_side": 4096}),
    )
    # sharpness clamps to 1, exposure is 1, resolution clamps to 1 -> 1.0
    assert TechnicalQuality().score(_Ref(img), manifest) == 1.0


def test_technical_quality_penalizes_clipping_and_softness():
    manifest: dict = {}
    img = "bad"
    put_annotation(manifest, Annotation(image_id=img, metric_id="blur", value=0.0))
    put_annotation(
        manifest,
        Annotation(
            image_id=img,
            metric_id="exposure",
            value={"frac_underexposed": 0.25, "frac_overexposed": 0.25},
        ),
    )
    put_annotation(
        manifest,
        Annotation(image_id=img, metric_id="resolution", value={"long_side": 0}),
    )
    # sharpness 0, exposure max(0, 1 - 0.5*2) = 0, resolution 0 -> 0.0
    assert TechnicalQuality().score(_Ref(img), manifest) == 0.0


def test_technical_quality_neutral_sharpness_when_blur_absent():
    """A pool scored without the blur scorer still ranks — neutral 0.5."""
    manifest: dict = {}
    img = "no_blur"
    put_annotation(
        manifest,
        Annotation(
            image_id=img,
            metric_id="exposure",
            value={"frac_underexposed": 0.0, "frac_overexposed": 0.0},
        ),
    )
    put_annotation(
        manifest,
        Annotation(image_id=img, metric_id="resolution", value={"long_side": 1024}),
    )
    # 0.5*0.5 + 0.3*1 + 0.2*1 = 0.75
    assert TechnicalQuality().score(_Ref(img), manifest) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# curate_for_environment
# ---------------------------------------------------------------------------


def test_curate_for_environment_ranks_sharp_above_soft():
    sharp, soft = _sharp_ref(), _blurred_ref()
    result = curate_for_environment([sharp, soft], k=2)
    assert [r.image_id for r in result.kept] == [sharp.image_id, soft.image_id]


def test_curate_for_environment_respects_k():
    result = curate_for_environment([_sharp_ref(), _blurred_ref()], k=1)
    assert len(result.kept) == 1
    assert result.kept[0].image_id == _sharp_ref().image_id


# ---------------------------------------------------------------------------
# curate_for_character (driven through the mock face detector)
# ---------------------------------------------------------------------------


def test_curate_for_character_ranks_sharp_above_soft():
    sharp, soft = _sharp_ref(), _blurred_ref()
    result = curate_for_character([sharp, soft], k=2, face_detector="mock_face")
    assert [r.image_id for r in result.kept] == [sharp.image_id, soft.image_id]


def test_curate_for_character_respects_k():
    result = curate_for_character(
        [_sharp_ref(), _blurred_ref()], k=1, face_detector="mock_face"
    )
    assert len(result.kept) == 1


def test_curate_for_character_no_face_still_returns_a_pick():
    """When no image has a face, the facade still yields a best-effort set —
    it never filters the pool down to empty."""
    refs = [_sharp_ref(), _blurred_ref()]
    result = curate_for_character(
        refs, k=2, face_detector=("mock_face", {"simulate_no_face": True})
    )
    assert len(result.kept) == 2
