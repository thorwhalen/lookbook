"""ArcFace identity embedder.

ArcFace embeddings are the standard for face *identity* verification —
neither CLIP nor DINOv2 reliably tells you "this is the same person."
For character-LoRA curation that's the load-bearing question.

`InsightFaceArcFace` shares the underlying model weights with
`InsightFaceDetect` (both go through the `FaceAnalysis` app). The
embedder reads the `face_box` annotation from the manifest to know where
to crop, then runs ArcFace on that crop.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from lookbook.base import ImageRef
from lookbook.manifest import value_of
from lookbook.registry import embedders


@dataclass
class MockArcFaceEmbedder:
    """Deterministic mock identity embedder, no deps.

    The vector is keyed by `image_id`, so:
    - Same image always yields the same vector.
    - Different images yield different vectors.
    - There is no "same person" signal — different images of one person
      get unrelated vectors. Useful only for orchestration tests.
    """

    space_id: str = "arcface_mock"
    cost_tier: int = 1
    dim: int = 512
    seed: int = 0
    backend: str = "lookbook:mock_arcface"

    @property
    def config_hash(self) -> str:
        return f"mock_arcface:{self.dim}:{self.seed}"

    def embed(self, ref: ImageRef) -> np.ndarray:
        h = hashlib.sha1(f"{self.seed}:{ref.image_id}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h, "big") % (2**63))
        v = rng.standard_normal(self.dim).astype(np.float32)
        n = np.linalg.norm(v)
        return (v / n) if n > 0 else v


embedders.register("arcface_mock", MockArcFaceEmbedder())


@dataclass
class InsightFaceArcFace:
    """Real ArcFace embeddings via insightface.

    The implementation is intentionally a thin wrapper. `insightface`'s
    `FaceAnalysis.get(img)` returns Face objects with `.embedding` already
    L2-normalized; we just pick the largest face's embedding.

    The "right" face is the one whose `bbox` overlaps the manifest's
    `face_box` if available; otherwise the largest by area.
    """

    space_id: str = "arcface"
    cost_tier: int = 2
    model_name: str = "buffalo_l"
    det_size: int = 640
    backend: str = "insightface:arcface"

    @property
    def config_hash(self) -> str:
        return f"arcface:{self.model_name}:{self.det_size}"

    def embed(self, ref: ImageRef) -> np.ndarray:
        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except ImportError as e:
            raise ImportError(
                "InsightFaceArcFace requires `insightface`. "
                "`pip install lookbook[person]`."
            ) from e

        if not hasattr(self, "_app"):
            app = FaceAnalysis(name=self.model_name)
            app.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size))
            self._app = app

        with ref.open() as img:
            arr = np.asarray(img.convert("RGB"))
        arr_bgr = arr[:, :, ::-1].copy()
        faces = self._app.get(arr_bgr)
        if not faces:
            # No face — return a zero vector. Downstream callers that care
            # should filter on `face_box is None` upstream so this branch
            # is never reached for kept images.
            return np.zeros(512, dtype=np.float32)

        biggest = max(
            faces,
            key=lambda f: max(0.0, f.bbox[2] - f.bbox[0])
            * max(0.0, f.bbox[3] - f.bbox[1]),
        )
        v = np.asarray(biggest.embedding, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


embedders.register("arcface", InsightFaceArcFace())
