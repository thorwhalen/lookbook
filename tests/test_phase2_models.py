"""Optional smoke tests for the real CLIP / DINOv2 embedders.

These download model weights from HuggingFace on first run (~150MB+450MB).
They're skipped by default to keep the test suite fast and offline-safe.
Opt in with: `LOOKBOOK_TEST_MODELS=1 pytest tests/test_phase2_models.py`.
"""

from __future__ import annotations

import io
import os

import numpy as np
import pytest
from PIL import Image

from lookbook import BytesImageRef


SKIP_MODELS = os.environ.get("LOOKBOOK_TEST_MODELS") != "1"
pytestmark = pytest.mark.skipif(
    SKIP_MODELS,
    reason="Set LOOKBOOK_TEST_MODELS=1 to run real-model smoke tests.",
)


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def small_ref():
    img = Image.new("RGB", (224, 224), (128, 64, 200))
    return BytesImageRef(payload=_png_bytes(img), image_id="ref")


def test_clip_embedder_smoke(small_ref):
    from lookbook.embedders.clip import CLIPEmbedder

    e = CLIPEmbedder()
    v = e.embed(small_ref)
    assert isinstance(v, np.ndarray)
    assert v.dtype == np.float32
    assert v.ndim == 1
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-3


def test_dinov2_embedder_smoke(small_ref):
    from lookbook.embedders.dinov2 import DINOv2Embedder

    e = DINOv2Embedder()
    v = e.embed(small_ref)
    assert isinstance(v, np.ndarray)
    assert v.dtype == np.float32
    assert v.ndim == 1
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-3
