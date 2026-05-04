"""Person-LoRA scorers — Phase 3.

Two backends ship for face detection:

- `MockFaceDetect`: synthesizes a centered face box, deterministic, no deps.
  Used by tests and by users who already have face boxes from another tool.
- `InsightFaceDetect`: real RetinaFace + landmarks + (optionally) gender/age,
  lazy-imported. Use for production.

Both write to the same `face_box` metric_id. They are distinguished by
`config_hash` so the cache invalidates if the user swaps backends.

Downstream scorers (`FaceArea`, `HeadPose`, `FaceQualityProxy`) read
`face_box` from the manifest via the `requires` mechanism. They never touch
the detector directly — the orchestrator runs the detector first.

The value shape of `face_box` (when a face is detected):

    {
        "x1": int, "y1": int, "x2": int, "y2": int,
        "confidence": float,         # detector score
        "n_faces": int,              # total faces detected before keeping the largest
        "landmarks": list | None,    # 5 (x, y) pairs if available, else None
    }

When no face is detected, the value is `None`.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from lookbook.base import ImageRef, Manifest
from lookbook.manifest import value_of
from lookbook.registry import scorers


def _hash_config(**kwargs) -> str:
    payload = repr(sorted(kwargs.items())).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Face detectors
# ---------------------------------------------------------------------------


@dataclass
class MockFaceDetect:
    """Deterministic face detector — synthesizes one centered box per image.

    Useful for tests and for end-to-end demos without a real detector. The
    box covers the central 60% of the image (or 0% if `simulate_no_face`
    is set, in which case the face_box value is None).
    """

    metric_id: str = "face_box"
    cost_tier: int = 1
    requires: tuple = ()
    backend: str = "lookbook:mock_face"
    simulate_no_face: bool = False

    @property
    def config_hash(self) -> str:
        return _hash_config(backend="mock", simulate_no_face=self.simulate_no_face)

    def score(self, ref: ImageRef, manifest: Manifest):
        if self.simulate_no_face:
            return None
        with ref.open() as img:
            w, h = img.size
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        return {
            "x1": margin_x,
            "y1": margin_y,
            "x2": w - margin_x,
            "y2": h - margin_y,
            "confidence": 0.99,
            "n_faces": 1,
            "landmarks": None,
        }


scorers.register("mock_face", MockFaceDetect())


@dataclass
class InsightFaceDetect:
    """RetinaFace face detection via insightface.

    Returns the largest face if multiple are present. `n_faces` records the
    pre-filter count so a downstream filter can drop multi-face images.
    """

    metric_id: str = "face_box"
    cost_tier: int = 1
    requires: tuple = ()
    backend: str = "insightface:retinaface"
    model_name: str = "buffalo_l"  # also: "buffalo_s", "buffalo_m"
    det_size: int = 640

    @property
    def config_hash(self) -> str:
        return _hash_config(backend="insightface", model=self.model_name, det=self.det_size)

    def score(self, ref: ImageRef, manifest: Manifest):
        try:
            import insightface  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore
            import cv2  # type: ignore
        except ImportError as e:
            raise ImportError(
                "InsightFaceDetect requires `insightface` and `opencv-python-headless`. "
                "`pip install lookbook[person]`."
            ) from e

        if not hasattr(self, "_app"):
            app = FaceAnalysis(name=self.model_name)
            app.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size))
            self._app = app

        with ref.open() as img:
            arr = np.asarray(img.convert("RGB"))
        # insightface expects BGR.
        arr_bgr = arr[:, :, ::-1].copy()
        faces = self._app.get(arr_bgr)
        if not faces:
            return None

        # Keep the largest face.
        def _area(f):
            x1, y1, x2, y2 = f.bbox
            return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

        biggest = max(faces, key=_area)
        x1, y1, x2, y2 = [int(round(v)) for v in biggest.bbox]
        landmarks = None
        if getattr(biggest, "kps", None) is not None:
            landmarks = [[float(p[0]), float(p[1])] for p in biggest.kps]
        return {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": float(biggest.det_score),
            "n_faces": len(faces),
            "landmarks": landmarks,
        }


scorers.register("insightface", InsightFaceDetect())


# ---------------------------------------------------------------------------
# Derived scorers — read face_box from manifest
# ---------------------------------------------------------------------------


def _bin_face_size(area_fraction: float) -> str:
    """Standard face-size bins for character-LoRA quota constraints."""
    if area_fraction <= 0:
        return "no_face"
    if area_fraction < 0.05:
        return "wide"        # full-body / wide shot
    if area_fraction < 0.15:
        return "medium"      # half-body
    if area_fraction < 0.35:
        return "close"       # head-and-shoulders
    return "extreme_close"   # face fills frame


@dataclass
class FaceArea:
    """Face area as a fraction of total image area, plus a size bin.

    Reads `face_box` and `resolution`. Both must be present upstream — the
    orchestrator's topo-sort handles this via the `requires` field.
    """

    metric_id: str = "face_area"
    cost_tier: int = 1
    requires: tuple = ("face_box", "resolution")
    backend: str = "derived"

    @property
    def config_hash(self) -> str:
        return "v1"

    def score(self, ref: ImageRef, manifest: Manifest) -> dict:
        box = value_of(manifest, ref.image_id, "face_box")
        res = value_of(manifest, ref.image_id, "resolution") or {}
        if not box:
            return {
                "fraction": 0.0,
                "pixels": 0,
                "size_bin": "no_face",
            }
        bw = max(0, box["x2"] - box["x1"])
        bh = max(0, box["y2"] - box["y1"])
        face_pixels = bw * bh
        total = max(1, int(res.get("width", 0)) * int(res.get("height", 0)))
        fraction = face_pixels / total
        return {
            "fraction": float(fraction),
            "pixels": int(face_pixels),
            "size_bin": _bin_face_size(fraction),
        }


scorers.register("face_area", FaceArea())


# ---------------------------------------------------------------------------
# Head pose
# ---------------------------------------------------------------------------


def _bin_yaw(yaw: float) -> str:
    """Five-bucket yaw binning suitable for character-LoRA pose coverage."""
    a = abs(yaw)
    if a <= 10:
        return "front"
    if a <= 30:
        return "three_quarter" if yaw > 0 else "three_quarter_left"
    if a <= 60:
        return "profile" if yaw > 0 else "profile_left"
    return "back" if yaw > 0 else "back_left"


def _bin_yaw_simple(yaw: float) -> str:
    """Three-bucket yaw binning: left / center / right."""
    if yaw < -15:
        return "left"
    if yaw > 15:
        return "right"
    return "center"


@dataclass
class MockHeadPose:
    """Synthetic head pose — deterministic from image_id.

    Yaw distributed roughly uniformly in [-90, 90]; pitch/roll narrower.
    Useful for testing the QuotaSelector without a real head-pose model.
    """

    metric_id: str = "head_pose"
    cost_tier: int = 1
    requires: tuple = ("face_box",)
    backend: str = "lookbook:mock_pose"

    @property
    def config_hash(self) -> str:
        return "mock"

    def score(self, ref: ImageRef, manifest: Manifest):
        box = value_of(manifest, ref.image_id, "face_box")
        if not box:
            return None
        h = hashlib.sha1(ref.image_id.encode("utf-8")).digest()
        # Map digest bytes to angles.
        yaw = ((h[0] / 255.0) * 180.0) - 90.0
        pitch = ((h[1] / 255.0) * 60.0) - 30.0
        roll = ((h[2] / 255.0) * 40.0) - 20.0
        return {
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "yaw_bin": _bin_yaw(yaw),
            "yaw_bin_simple": _bin_yaw_simple(yaw),
        }


scorers.register("mock_head_pose", MockHeadPose())


@dataclass
class SixDRepNetHeadPose:
    """Head-pose estimation via 6DRepNet (lazy-imported).

    See https://github.com/thohemp/6DRepNet for the reference implementation.
    """

    metric_id: str = "head_pose"
    cost_tier: int = 2
    requires: tuple = ("face_box",)
    backend: str = "sixdrepnet"
    config_version: str = "v1"

    @property
    def config_hash(self) -> str:
        return _hash_config(backend="6drepnet", v=self.config_version)

    def score(self, ref: ImageRef, manifest: Manifest):
        try:
            from sixdrepnet import SixDRepNet  # type: ignore
            import cv2  # type: ignore
        except ImportError as e:
            raise ImportError(
                "SixDRepNetHeadPose requires `sixdrepnet`. "
                "`pip install lookbook[person]`."
            ) from e

        box = value_of(manifest, ref.image_id, "face_box")
        if not box:
            return None

        if not hasattr(self, "_model"):
            self._model = SixDRepNet()

        with ref.open() as img:
            arr = np.asarray(img.convert("RGB"))
        # Crop to the face box with a small margin for context.
        h_img, w_img = arr.shape[:2]
        pad_x = max(1, int(0.1 * (box["x2"] - box["x1"])))
        pad_y = max(1, int(0.1 * (box["y2"] - box["y1"])))
        x1 = max(0, box["x1"] - pad_x)
        y1 = max(0, box["y1"] - pad_y)
        x2 = min(w_img, box["x2"] + pad_x)
        y2 = min(h_img, box["y2"] + pad_y)
        crop = arr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop_bgr = crop[:, :, ::-1].copy()
        pitch, yaw, roll = self._model.predict(crop_bgr)
        return {
            "yaw": float(yaw),
            "pitch": float(pitch),
            "roll": float(roll),
            "yaw_bin": _bin_yaw(float(yaw)),
            "yaw_bin_simple": _bin_yaw_simple(float(yaw)),
        }


scorers.register("head_pose", SixDRepNetHeadPose())


# ---------------------------------------------------------------------------
# Face quality proxy
# ---------------------------------------------------------------------------


@dataclass
class FaceQualityProxy:
    """Combine detector confidence, face area, and (optionally) blur into a
    single 0..1 score.

    This is a stand-in for true face image quality assessment (FIQA — see
    SDD-FIQA, CR-FIQA, CLIB-FIQA in the design report). Real FIQA needs a
    purpose-trained model; the proxy here is a reasonable default for
    Phase 3 that uses only signals already in the manifest.
    """

    metric_id: str = "face_quality"
    cost_tier: int = 1
    requires: tuple = ("face_box", "face_area")
    backend: str = "derived"
    blur_normalization: float = 500.0  # divisor for the optional blur term

    @property
    def config_hash(self) -> str:
        return _hash_config(blur_norm=self.blur_normalization)

    def score(self, ref: ImageRef, manifest: Manifest) -> float:
        box = value_of(manifest, ref.image_id, "face_box")
        area = value_of(manifest, ref.image_id, "face_area") or {}
        if not box:
            return 0.0

        confidence = float(box.get("confidence", 0.0))
        # Reward face-area in the "useful" 0.05..0.35 band; penalize outliers
        # using a triangular reward peaking around 0.20.
        f = float(area.get("fraction", 0.0))
        area_score = max(0.0, 1.0 - abs(f - 0.20) / 0.20) if f > 0 else 0.0

        # Optional sharpness term — only used if `blur` annotation present.
        blur = value_of(manifest, ref.image_id, "blur")
        if blur is None:
            sharpness = 0.5  # neutral when unknown
        else:
            sharpness = min(1.0, float(blur) / float(self.blur_normalization))

        # Weighted geometric-ish mean — all factors must be at least minimal.
        return float(0.5 * confidence + 0.3 * area_score + 0.2 * sharpness)


scorers.register("face_quality", FaceQualityProxy())
