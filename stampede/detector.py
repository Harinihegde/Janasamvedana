"""Person detection with a YOLO primary path and a CSRNet-style fallback.

On extremely dense frames the YOLO detector saturates and under-counts. When
that happens we fall back to a density estimate: a real trained CSRNet
checkpoint (see :mod:`.csrnet`) if ``csrnet_weights`` is supplied, otherwise a
dependency-free image-processing surrogate so the pipeline still runs
end-to-end without requiring a model download.
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

    def __init__(self, weights: str, imgsz: int = YOLO_IMGSZ, csrnet_weights: str | None = None):
        if not weights:
            raise ValueError(
                "weights path is required: the pipeline will not silently "
                "download a detector"
            )
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self._csrnet = None
        if csrnet_weights:
            from .csrnet import load_csrnet

            self._csrnet = load_csrnet(csrnet_weights)

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
    def _heuristic_density_count(frame: np.ndarray) -> int:
        """Edge-density surrogate: estimate a head count without a trained model.

        Not a trained model - a transparent stand-in that returns a *higher*
        count than YOLO on saturated frames so downstream density features stay
        monotonic. Used only when no ``csrnet_weights`` were supplied.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        # Roughly one head per patch of strong edges; scale is illustrative.
        density = edges.mean() / 255.0
        return int(round(density * frame.shape[0] * frame.shape[1] / 2500.0))

    def _density_count(self, frame: np.ndarray) -> int:
        if self._csrnet is not None:
            from .csrnet import estimate_count

            return int(round(estimate_count(self._csrnet, frame)))
        return self._heuristic_density_count(frame)

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, bool]:
        boxes, conf = self._yolo(frame)
        count = len(boxes)
        if count < CSRNET_FALLBACK_MIN_COUNT and self._looks_dense(frame):
            est = self._density_count(frame)
            if est > count:
                return boxes, conf, est, True
        return boxes, conf, count, False

    def csrnet_density(self, frame: np.ndarray) -> tuple[float, float]:
        """Always-on ``(csrnet_count, csrnet_peak_density)`` for one frame.

        Unlike ``__call__``'s fallback (only triggered when YOLO undercounts),
        this runs on every frame when a CSRNet checkpoint is loaded, giving
        Stage 1 a continuous density signal independent of YOLO's box count.
        Returns ``(0.0, 0.0)`` when no ``csrnet_weights`` were supplied, so
        callers don't need to branch on whether CSRNet is available.
        """
        if self._csrnet is None:
            return 0.0, 0.0
        from .csrnet import ALWAYS_ON_MAX_SIDE, estimate_density

        return estimate_density(self._csrnet, frame, max_side=ALWAYS_ON_MAX_SIDE)
