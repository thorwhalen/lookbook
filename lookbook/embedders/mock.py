"""Deterministic mock embedder for tests and end-to-end orchestration.

`MockEmbedder` returns a vector seeded by `image_id`, so the same image
always yields the same vector and similar `image_id`s yield similar
vectors (a tiny bit). It does not require any ML dependency.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from lookbook.base import ImageRef
from lookbook.registry import embedders


@dataclass
class MockEmbedder:
    """Deterministic, dependency-free embedder for tests.

    The vector is derived from a SHA1 of `image_id` (mixed with `seed`),
    expanded to `dim` dimensions, and L2-normalized.
    """

    space_id: str = "mock"
    cost_tier: int = 0
    dim: int = 64
    seed: int = 0
    backend: str = "lookbook:mock"

    @property
    def config_hash(self) -> str:
        return f"mock:{self.dim}:{self.seed}"

    def embed(self, ref: ImageRef) -> np.ndarray:
        h = hashlib.sha1(f"{self.seed}:{ref.image_id}".encode("utf-8")).digest()
        # Expand the 20-byte digest deterministically to `dim` floats.
        rng = np.random.default_rng(int.from_bytes(h, "big") % (2**63))
        v = rng.standard_normal(self.dim).astype(np.float32)
        # L2-normalize so cosine similarity == dot product.
        n = np.linalg.norm(v)
        return (v / n) if n > 0 else v


embedders.register("mock", MockEmbedder())
