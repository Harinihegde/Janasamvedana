"""Person detection with a YOLO primary path and a CSRNet-style fallback.

On extremely dense frames the YOLO detector saturates and under-counts. When
that happens we fall back to a lightweight density estimate. A real deployment
would swap in a trained CSRNet checkpoint; here we provide a dependency-free
image-processing surrogate so the pipeline runs end-to-end without an extra
model download, while keeping the *interface* identical to a CSRNet wrapper.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import (
    CSRNET_FALLBACK_MIN_COUNT,
    YOLO_CONF,
    YOLO_IMGSZ,
    YOLO_PERSON_CLASS,
)


class Detector:
    """Callable that returns person bounding boxes, confidences and a count.

    ``__call__`` returns ``(boxes, confidences, count, used_fallback)`` where
    ``boxes`` is an ``(N, 4)`` xyxy array and ``confidences`` the matching
    ``(N,)`` YOLO scores (both possibly empty when the fallback fired, since the
    density estimate yields a count without boxes).
    """

    def __init__(self, weights: str, imgsz: int = YOLO_IMGSZ):
        if not weights:
            raise ValueError(
                "weights path is required: the pipeline will not silently "
                "download a detector"
            )
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz = imgsz

    def _yolo(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.model(
            frame,
            classes=[YOLO_PERSON_CLASS],
            conf=YOLO_CONF,
            imgsz=self.imgsz,
            verbose=False,
        )[0]
        if r.boxes is None:
            return np.empty((0, 4), dtype=float), np.empty((0,), dtype=float)
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

    @staticmethod
    def _looks_dense(frame: np.ndarray) -> bool:
        """Cheap texture heuristic: dense crowds have high edge energy."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Laplacian(gray, cv2.CV_64F)
        return float(edges.var()) > 500.0

    @staticmethod
    def _density_count(frame: np.ndarray) -> int:
        """CSRNet surrogate: estimate a head count from local edge density.

        Not a trained model - a transparent stand-in that returns a *higher*
        count than YOLO on saturated frames so downstream density features stay
        monotonic. Documented as a fallback, never the primary path.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        # Roughly one head per patch of strong edges; scale is illustrative.
        density = edges.mean() / 255.0
        return int(round(density * frame.shape[0] * frame.shape[1] / 2500.0))

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, bool]:
        boxes, conf = self._yolo(frame)
        count = len(boxes)
        if count < CSRNET_FALLBACK_MIN_COUNT and self._looks_dense(frame):
            est = self._density_count(frame)
            if est > count:
                return boxes, conf, est, True
        return boxes, conf, count, False
