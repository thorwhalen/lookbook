"""CLIP embedder via `transformers`.

CLIP is the right embedding for *semantic* similarity (same concept, different
style). It's also what LAION-Aesthetic-V2 and CLIP-IQA both consume, so when
those scorers land they can share the cached features computed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from lookbook.base import ImageRef
from lookbook.registry import embedders


@dataclass
class CLIPEmbedder:
    """OpenAI CLIP image embeddings via HuggingFace transformers.

    Defaults to ViT-B/32 (~150 MB, fast enough for CPU). Use ViT-L/14 by
    setting `model_name="openai/clip-vit-large-patch14"` for higher-quality
    semantic embeddings (slower, ~600 MB).
    """

    space_id: str = "clip_vit_b32"
    cost_tier: int = 2
    model_name: str = "openai/clip-vit-base-patch32"
    device: Optional[str] = None  # auto-detect
    backend: str = "transformers:clip"

    @property
    def config_hash(self) -> str:
        return f"clip:{self.model_name}"

    def embed(self, ref: ImageRef) -> np.ndarray:
        try:
            import torch  # type: ignore
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
        except ImportError as e:
            raise ImportError(
                "CLIPEmbedder requires `torch` and `transformers`. "
                "`pip install lookbook[embed]`."
            ) from e

        if not hasattr(self, "_model"):
            device = self.device or _autodevice(torch)
            self._model = CLIPModel.from_pretrained(self.model_name).to(device)
            self._model.eval()
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._device = device

        with ref.open() as img:
            img = img.convert("RGB")
            inputs = self._processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self._model.get_image_features(**inputs)
            v = feats.squeeze(0).cpu().numpy().astype(np.float32)

        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


def _autodevice(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


embedders.register("clip", CLIPEmbedder())
