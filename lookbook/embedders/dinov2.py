"""DINOv2 embedder via `transformers`.

DINOv2 is the right embedding for *visual* similarity (same scene/object,
robust to style). Voxel51's benchmarks show it outperforming CLIP on image
classification by 5-28 points on natural-image datasets. For curation, that
makes it the better default for "find redundant shots."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from lookbook.base import ImageRef
from lookbook.embedders.clip import _autodevice
from lookbook.registry import embedders


@dataclass
class DINOv2Embedder:
    """Facebook DINOv2 image embeddings via HuggingFace transformers.

    Defaults to ViT-B/14 (~350 MB). Use `facebook/dinov2-small` for
    smaller-but-weaker (~85 MB), `facebook/dinov2-large` for stronger
    (~1.2 GB).
    """

    space_id: str = "dinov2_base"
    cost_tier: int = 2
    model_name: str = "facebook/dinov2-base"
    device: Optional[str] = None
    backend: str = "transformers:dinov2"

    @property
    def config_hash(self) -> str:
        return f"dinov2:{self.model_name}"

    def embed(self, ref: ImageRef) -> np.ndarray:
        try:
            import torch  # type: ignore
            from transformers import AutoImageProcessor, AutoModel  # type: ignore
        except ImportError as e:
            raise ImportError(
                "DINOv2Embedder requires `torch` and `transformers`. "
                "`pip install lookbook[embed]`."
            ) from e

        if not hasattr(self, "_model"):
            device = self.device or _autodevice(torch)
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(device)
            self._model.eval()
            self._device = device

        with ref.open() as img:
            img = img.convert("RGB")
            inputs = self._processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            # Use the CLS token (pooler_output) as the image embedding.
            v = outputs.pooler_output.squeeze(0).cpu().numpy().astype(np.float32)

        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


embedders.register("dinov2", DINOv2Embedder())
