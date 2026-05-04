"""Technical scorers — Phase 0/1.

Phase 0 ships `RandomScore` (placeholder for end-to-end orchestration).

Phase 1 adds the cheap-funnel scorers — these run on CPU with Pillow + numpy
as the base path, falling back to richer backends (cv2, imagehash) when
they're available. None of them require torch or any GPU dep.

| metric_id     | tier | requires    | needs                |
|---------------|------|-------------|----------------------|
| resolution    | T0   | —           | Pillow               |
| file_hash     | T0   | —           | stdlib               |
| phash         | T1   | —           | imagehash (optional) |
| blur          | T1   | —           | numpy (cv2 optional) |
| exposure      | T1   | —           | numpy + Pillow       |
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from lookbook.base import ImageRef, Manifest
from lookbook.registry import scorers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_config(**kwargs) -> str:
    """Stable short hash of a scorer's config dict."""
    payload = repr(sorted(kwargs.items())).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _to_grayscale_array(ref: ImageRef, max_side: int = 512) -> np.ndarray:
    """Open an image as a grayscale numpy array, downsampled if huge.

    Downsampling keeps Phase 1 cheap on photo-dump-sized inputs without
    materially changing blur / exposure measurements.
    """
    from PIL import Image

    img = ref.open()
    if img.mode != "L":
        img = img.convert("L")
    w, h = img.size
    long_side = max(w, h)
    if long_side > max_side:
        scale = max_side / long_side
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return np.asarray(img, dtype=np.float32)


# ---------------------------------------------------------------------------
# Phase 0: random_score (kept for testing)
# ---------------------------------------------------------------------------


@dataclass
class RandomScore:
    """Deterministic pseudo-random score in [0, 1) keyed by image_id."""

    metric_id: str = "random_score"
    cost_tier: int = 0
    requires: tuple = ()
    seed: int = 0
    backend: str = "lookbook:random"

    @property
    def config_hash(self) -> str:
        return _hash_config(seed=self.seed)

    def score(self, ref: ImageRef, manifest: Manifest) -> float:
        h = hashlib.sha1(f"{self.seed}:{ref.image_id}".encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big") / 2**64


scorers.register("random_score", RandomScore())


# ---------------------------------------------------------------------------
# Phase 1: cheap funnel
# ---------------------------------------------------------------------------


@dataclass
class Resolution:
    """Image dimensions and long-side. T0 — header read, no decode."""

    metric_id: str = "resolution"
    cost_tier: int = 0
    requires: tuple = ()
    backend: str = "Pillow"

    @property
    def config_hash(self) -> str:
        return "v1"

    def score(self, ref: ImageRef, manifest: Manifest) -> dict:
        # `Image.open` is lazy — does not decode pixels.
        with ref.open() as img:
            w, h = img.size
            mode = img.mode
        return {
            "width": int(w),
            "height": int(h),
            "long_side": int(max(w, h)),
            "short_side": int(min(w, h)),
            "aspect_ratio": float(w) / float(h) if h else 0.0,
            "mode": mode,
        }


scorers.register("resolution", Resolution())


@dataclass
class FileHash:
    """SHA1 of the raw image bytes — exact-duplicate detection. T0."""

    metric_id: str = "file_hash"
    cost_tier: int = 0
    requires: tuple = ()
    backend: str = "stdlib:sha1"

    @property
    def config_hash(self) -> str:
        return "sha1"

    def score(self, ref: ImageRef, manifest: Manifest) -> str:
        return hashlib.sha1(ref.bytes()).hexdigest()


scorers.register("file_hash", FileHash())


@dataclass
class PerceptualHash:
    """Perceptual hash for near-duplicate detection. T1.

    Uses `imagehash` (MIT) when available; result is the hex string of the
    hash. The default algorithm is `phash` (DCT-based). Hamming distance
    between hex strings (over hash bits) is the standard near-dup metric;
    `lookbook.scorers.technical.phash_distance` provides it.
    """

    metric_id: str = "phash"
    cost_tier: int = 1
    requires: tuple = ()
    backend: str = "imagehash"
    algorithm: str = "phash"  # phash | ahash | dhash | whash

    @property
    def config_hash(self) -> str:
        return _hash_config(algorithm=self.algorithm)

    def score(self, ref: ImageRef, manifest: Manifest):
        try:
            import imagehash  # type: ignore
        except ImportError as e:
            raise ImportError(
                "phash scorer requires `imagehash`. "
                "`pip install lookbook[funnel]` or `pip install imagehash`."
            ) from e
        fn = getattr(imagehash, self.algorithm)
        with ref.open() as img:
            h = fn(img)
        return str(h)


scorers.register("phash", PerceptualHash())


def phash_distance(a: str, b: str) -> int:
    """Hamming distance between two hex-encoded perceptual hashes."""
    if len(a) != len(b):
        raise ValueError(f"hash length mismatch: {len(a)} vs {len(b)}")
    bits_a = bin(int(a, 16))[2:].zfill(len(a) * 4)
    bits_b = bin(int(b, 16))[2:].zfill(len(b) * 4)
    return sum(x != y for x, y in zip(bits_a, bits_b))


@dataclass
class Blur:
    """Variance of Laplacian — higher means sharper. T1.

    Uses cv2 when available (the canonical implementation); otherwise falls
    back to a small numpy 3x3 kernel convolution. Numbers are comparable to
    the cv2 path within a few percent on natural images.
    """

    metric_id: str = "blur"
    cost_tier: int = 1
    requires: tuple = ()
    backend: str = "cv2-or-numpy"
    max_side: int = 512  # downsample for speed

    @property
    def config_hash(self) -> str:
        return _hash_config(max_side=self.max_side, kernel="laplacian3x3")

    def score(self, ref: ImageRef, manifest: Manifest) -> float:
        gray = _to_grayscale_array(ref, max_side=self.max_side)
        try:
            import cv2  # type: ignore

            lap = cv2.Laplacian(gray, cv2.CV_32F)
            return float(lap.var())
        except ImportError:
            # 3x3 Laplacian convolution via numpy (slow but dependency-free).
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            from numpy.lib.stride_tricks import sliding_window_view

            windows = sliding_window_view(gray, (3, 3))
            lap = (windows * kernel).sum(axis=(-1, -2))
            return float(lap.var())


scorers.register("blur", Blur())


@dataclass
class Exposure:
    """Exposure characterization from the grayscale histogram. T1.

    Returns mean, std, and key percentiles. Together they let a downstream
    filter say "drop images that are mostly black or mostly white."
    """

    metric_id: str = "exposure"
    cost_tier: int = 1
    requires: tuple = ()
    backend: str = "numpy"
    max_side: int = 512

    @property
    def config_hash(self) -> str:
        return _hash_config(max_side=self.max_side)

    def score(self, ref: ImageRef, manifest: Manifest) -> dict:
        gray = _to_grayscale_array(ref, max_side=self.max_side)
        flat = gray.ravel()
        return {
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "p01": float(np.percentile(flat, 1)),
            "p05": float(np.percentile(flat, 5)),
            "p50": float(np.percentile(flat, 50)),
            "p95": float(np.percentile(flat, 95)),
            "p99": float(np.percentile(flat, 99)),
            "frac_underexposed": float((flat < 16).mean()),
            "frac_overexposed": float((flat > 240).mean()),
        }


scorers.register("exposure", Exposure())
