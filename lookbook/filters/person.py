"""Person-LoRA filters — Phase 3.

These read `face_box`, `face_area`, and (optionally) `head_pose` from the
manifest. Like all filters in lookbook, they keep an image when the
relevant evidence is missing — the design principle is that filters never
drop on absent annotations.
"""

from __future__ import annotations

from dataclasses import dataclass

from lookbook.base import ImageRef, Manifest
from lookbook.manifest import value_of
from lookbook.registry import filters


@dataclass
class HasFace:
    """Drop images with no detected face."""

    name: str = "has_face"

    def keep(self, ref: ImageRef, manifest: Manifest) -> bool:
        box = value_of(manifest, ref.image_id, "face_box")
        # When the detector hasn't run yet, keep — filters are non-destructive.
        if not manifest.get((ref.image_id, "face_box")):
            # Annotation absent entirely → keep (no evidence).
            return True
        return box is not None


filters.register("has_face", HasFace())


@dataclass
class SingleFaceOnly:
    """Drop images with more than one detected face.

    Reads `face_box.n_faces`. Useful for character LoRAs where you want
    isolated subject images.
    """

    name: str = "single_face_only"

    def keep(self, ref: ImageRef, manifest: Manifest) -> bool:
        box = value_of(manifest, ref.image_id, "face_box")
        if not box:
            # No face at all is a separate concern; HasFace filter handles it.
            return True
        return int(box.get("n_faces", 1)) <= 1


filters.register("single_face_only", SingleFaceOnly())


@dataclass
class MinFaceArea:
    """Drop images whose face occupies less than `min_fraction` of the frame."""

    min_fraction: float = 0.05
    name: str = "min_face_area"

    def keep(self, ref: ImageRef, manifest: Manifest) -> bool:
        area = value_of(manifest, ref.image_id, "face_area")
        if not area:
            return True
        return float(area.get("fraction", 0.0)) >= self.min_fraction


filters.register("min_face_area", MinFaceArea())


@dataclass
class MinFaceConfidence:
    """Drop images whose face-detection confidence is below `threshold`."""

    threshold: float = 0.5
    name: str = "min_face_confidence"

    def keep(self, ref: ImageRef, manifest: Manifest) -> bool:
        box = value_of(manifest, ref.image_id, "face_box")
        if not box:
            return True
        return float(box.get("confidence", 0.0)) >= self.threshold


filters.register("min_face_confidence", MinFaceConfidence())
