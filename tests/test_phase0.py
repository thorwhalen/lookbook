"""End-to-end Phase 0 tests.

These verify the full vertical slice runs against in-memory stores. They do
NOT touch the user's app data folder.
"""

from __future__ import annotations

import io
import os
from datetime import datetime

import pytest

from lookbook import (
    Annotation,
    BytesImageRef,
    PathImageRef,
    Pipeline,
    curate,
    get_stores,
    registry,
    score,
)
from lookbook.manifest import (
    get_annotation,
    has_annotation,
    image_ids,
    iter_annotations_for,
    put_annotation,
    value_of,
)
from lookbook.store import manifest_codec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def png_bytes():
    """Tiny valid PNG: a 1x1 transparent image."""
    from PIL import Image

    img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def image_dir(tmp_path, png_bytes):
    for i in range(5):
        (tmp_path / f"img_{i}.png").write_bytes(png_bytes)
    # noise files that should be ignored
    (tmp_path / "notes.txt").write_text("hello")
    return str(tmp_path)


@pytest.fixture
def memory_stores():
    return get_stores(
        images_store={}, manifest_store={}, runs_store={}, embeddings={},
    )


# ---------------------------------------------------------------------------
# Annotation / Manifest
# ---------------------------------------------------------------------------


def test_annotation_key_is_pair():
    a = Annotation(image_id="x", metric_id="m", value=1.0)
    assert a.key == ("x", "m")


def test_manifest_helpers_roundtrip():
    manifest = {}
    a = Annotation(image_id="x", metric_id="m", value=1.0)
    put_annotation(manifest, a)
    assert has_annotation(manifest, "x", "m")
    assert get_annotation(manifest, "x", "m") == a
    assert value_of(manifest, "x", "m") == 1.0
    assert value_of(manifest, "x", "missing", default=None) is None
    assert list(image_ids(manifest)) == ["x"]
    assert list(iter_annotations_for(manifest, "x")) == [a]


# ---------------------------------------------------------------------------
# Refs
# ---------------------------------------------------------------------------


def test_path_ref_id_is_stable(tmp_path, png_bytes):
    p = tmp_path / "x.png"
    p.write_bytes(png_bytes)
    a = PathImageRef(path=str(p))
    b = PathImageRef(path=str(p))
    assert a.image_id == b.image_id
    assert len(a.image_id) == 16


def test_bytes_ref_id_depends_on_payload(png_bytes):
    a = BytesImageRef(payload=png_bytes)
    b = BytesImageRef(payload=png_bytes)
    c = BytesImageRef(payload=png_bytes + b"\x00")
    assert a.image_id == b.image_id
    assert a.image_id != c.image_id


def test_bytes_ref_opens_pil(png_bytes):
    ref = BytesImageRef(payload=png_bytes)
    img = ref.open()
    assert img.size == (1, 1)


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------


def test_get_stores_in_memory():
    s = get_stores(
        images_store={}, manifest_store={}, runs_store={}, embeddings={},
    )
    assert s.root is None
    s.manifest[("x", "m")] = Annotation(image_id="x", metric_id="m", value=1)
    assert ("x", "m") in s.manifest


def test_get_stores_filesystem(tmp_path):
    s = get_stores(root=str(tmp_path))
    a = Annotation(image_id="x", metric_id="m", value=1.0, config_hash="abc")
    s.manifest[("x", "m")] = a
    # Force a re-read by building a fresh codec-wrapped store on the same dir.
    s2 = get_stores(root=str(tmp_path))
    got = s2.manifest[("x", "m")]
    assert got.image_id == "x"
    assert got.metric_id == "m"
    assert got.value == 1.0
    assert got.config_hash == "abc"


def test_manifest_codec_keys_translate(tmp_path):
    from dol import JsonFiles

    raw = JsonFiles(str(tmp_path))
    wrapped = manifest_codec(raw)
    a = Annotation(image_id="abc", metric_id="blur", value=42)
    wrapped[("abc", "blur")] = a
    # Underlying file name encodes the pair (filesystem-safe separator).
    assert any("abc--blur" in k for k in raw)
    # Keys decoded back to tuples on read.
    assert ("abc", "blur") in list(wrapped)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_random_score_registered():
    assert "random_score" in registry.scorers.names()
    s = registry.scorers.get("random_score")
    assert s.metric_id == "random_score"


def test_top_k_registered():
    assert "top_k" in registry.selectors.names()


# ---------------------------------------------------------------------------
# Pipeline / facade
# ---------------------------------------------------------------------------


def test_curate_end_to_end(image_dir, memory_stores):
    result = curate(image_dir, k=3, stores=memory_stores)
    assert len(result.kept) == 3
    assert len(result.candidates) == 5
    # All kept refs have a random_score in the manifest.
    for ref in result.kept:
        assert has_annotation(memory_stores.manifest, ref.image_id, "random_score")


def test_curate_is_idempotent_via_manifest_cache(image_dir, memory_stores):
    """Re-running shouldn't recompute scores when config_hash matches."""
    curate(image_dir, k=3, stores=memory_stores)
    n_first = len(list(memory_stores.manifest))
    # Mutate annotations to detectable sentinels — if the cache is honored,
    # values should not be overwritten on second run.
    for key in list(memory_stores.manifest):
        ann = memory_stores.manifest[key]
        memory_stores.manifest[key] = Annotation(
            image_id=ann.image_id,
            metric_id=ann.metric_id,
            value="SENTINEL",
            config_hash=ann.config_hash,
            cost_tier=ann.cost_tier,
            timestamp=ann.timestamp,
            backend=ann.backend,
        )
    curate(image_dir, k=3, stores=memory_stores)
    n_second = len(list(memory_stores.manifest))
    assert n_second == n_first
    sentinels = [
        memory_stores.manifest[k].value
        for k in memory_stores.manifest
        if memory_stores.manifest[k].value == "SENTINEL"
    ]
    assert sentinels, "cache was not honored — scorer recomputed"


def test_score_one(image_dir, memory_stores):
    paths = sorted(
        os.path.join(image_dir, fn) for fn in os.listdir(image_dir)
        if fn.endswith(".png")
    )
    v = score(paths[0], metric_id="random_score", stores=memory_stores)
    assert isinstance(v, float)
    assert 0.0 <= v < 1.0


def test_run_record_persisted(image_dir, memory_stores):
    result = curate(image_dir, k=2, stores=memory_stores)
    keys = list(memory_stores.runs)
    assert any(result.run_id in k for k in keys)


def test_run_persists_to_filesystem(tmp_path, image_dir):
    """Running with a real on-disk store leaves artifacts behind."""
    stores = get_stores(root=str(tmp_path))
    result = curate(image_dir, k=2, stores=stores)
    # Manifest dir non-empty
    assert os.listdir(os.path.join(str(tmp_path), "manifest"))
    # Runs dir contains our run
    runs = os.listdir(os.path.join(str(tmp_path), "runs"))
    assert any(result.run_id in r for r in runs)
