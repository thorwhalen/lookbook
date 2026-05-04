"""ImageRef.local_path() and the to_local_path() helper."""

from __future__ import annotations

import os

import pytest

from lookbook import (
    BytesImageRef,
    PathImageRef,
    UrlImageRef,
    to_local_path,
)


PNG_HEADER = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPG_HEADER = b"\xff\xd8\xff\xe0" + b"\x00" * 12


def test_path_ref_returns_existing_path(tmp_path):
    f = tmp_path / "x.png"
    f.write_bytes(PNG_HEADER)
    ref = PathImageRef(path=str(f))
    assert ref.local_path() == str(f)
    assert to_local_path(ref) == str(f)


def test_bytes_ref_materializes_into_cache_dir(tmp_path):
    cache = tmp_path / "cache"
    ref = BytesImageRef(payload=PNG_HEADER)
    p = ref.local_path(cache_dir=str(cache))
    assert os.path.exists(p)
    assert p.endswith(".png"), "should sniff PNG extension"
    # Idempotent: second call returns the same path.
    assert ref.local_path(cache_dir=str(cache)) == p
    # Re-call doesn't re-write the file.
    mtime = os.path.getmtime(p)
    ref.local_path(cache_dir=str(cache))
    assert os.path.getmtime(p) == mtime


def test_bytes_ref_jpeg_extension(tmp_path):
    ref = BytesImageRef(payload=JPG_HEADER)
    p = ref.local_path(cache_dir=str(tmp_path))
    assert p.endswith(".jpg")


def test_bytes_ref_unknown_payload_falls_back_to_bin(tmp_path):
    ref = BytesImageRef(payload=b"random garbage \x01\x02\x03")
    p = ref.local_path(cache_dir=str(tmp_path))
    assert p.endswith(".bin")


def test_url_ref_caches_download(tmp_path, monkeypatch):
    """``UrlImageRef.local_path`` materializes once and caches."""
    cache = tmp_path / "cache"

    # Stub urlopen via the .bytes() pre-cache to avoid the network.
    ref = UrlImageRef(url="http://example.invalid/x.png")
    ref._cached = PNG_HEADER  # type: ignore[attr-defined]

    p1 = ref.local_path(cache_dir=str(cache))
    p2 = ref.local_path(cache_dir=str(cache))
    assert p1 == p2
    assert os.path.exists(p1)


def test_to_local_path_helper_dispatches(tmp_path):
    """The free function works on any ref subtype."""
    f = tmp_path / "img.png"
    f.write_bytes(PNG_HEADER)

    p_ref = PathImageRef(path=str(f))
    b_ref = BytesImageRef(payload=PNG_HEADER)
    assert to_local_path(p_ref) == str(f)
    assert to_local_path(b_ref, cache_dir=str(tmp_path / "c")).endswith(".png")


def test_env_var_default_cache_dir(tmp_path, monkeypatch):
    """``LOOKBOOK_REFS_CACHE_DIR`` is honored when no cache_dir is passed."""
    monkeypatch.setenv("LOOKBOOK_REFS_CACHE_DIR", str(tmp_path / "env_cache"))
    ref = BytesImageRef(payload=PNG_HEADER)
    p = ref.local_path()
    assert p.startswith(str(tmp_path / "env_cache"))
