"""Per-frame crowd-risk feature extraction.

Produces one feature row per *sampled* frame of a clip, covering the four
families in the project spec:

* **A. Density**   - person count, crowd density, a saturating density index.
* **B. Motion**    - optical-flow magnitude mean/variance and per-person
  velocity mean/variance from centroid tracking.
* **C. Temporal instability** (sliding window) - motion-instability sigma,
  stop-go oscillation energy, and flow-direction consistency.
* **D. Control loss** - spatial trajectory dispersion and density trend.

All features are resolution-invariant where possible (motion normalised by the
frame diagonal, velocity expressed per second) so clips of different size and
frame rate are comparable. Global [0, 1] scaling is applied later by the
classifier using saved normalisation parameters.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .config import (
    FEATURE_COLUMNS,
    FLOW_WIDTH,
    FRAME_STRIDE,
    MAX_FRAMES_PER_CLIP,
    TEMPORAL_WINDOW,
)
from .detector import Detector


class CentroidTracker:
    """Minimal nearest-neighbour tracker matching box centroids frame to frame.

    Returns, on each update, the per-track displacement of every centroid that
    was successfully matched to the previous frame. That is all Stage 1 needs to
    estimate per-person velocity - we deliberately avoid a heavyweight tracker.
    """

    def __init__(self, max_dist: float):
        self.max_dist = max_dist
        self.prev: np.ndarray | None = None  # (N, 2) centroids

    @staticmethod
    def _centroids(boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.empty((0, 2), dtype=float)
        return np.column_stack(
            [(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0]
        )

    def update(self, boxes: np.ndarray) -> np.ndarray:
        """Return an array of matched displacement distances (px)."""
        cur = self._centroids(boxes)
        displacements = np.empty((0,), dtype=float)
        if self.prev is not None and len(cur) and len(self.prev):
            d = cdist(cur, self.prev)  # (n_cur, n_prev)
            # Greedy mutual nearest-neighbour matching.
            matched = []
            while d.size and d.min() <= self.max_dist:
                i, j = np.unravel_index(np.argmin(d), d.shape)
                matched.append(d[i, j])
                d[i, :] = np.inf
                d[:, j] = np.inf
            displacements = np.asarray(matched, dtype=float)
        self.prev = cur
        return displacements


# ---------------------------------------------------------------------------
# Compression / occlusion features (Panic-crush vs dense-Crowdy)
# ---------------------------------------------------------------------------
def compute_bbox_overlap_ratio(boxes: np.ndarray) -> float:
    """Sum of pairwise box-intersection area / sum of box areas.

    Vectorised equivalent of the O(N^2) double loop: crushed crowds (Panic)
    overlap far more than dense-but-distinct crowds (Crowdy).
    """
    if len(boxes) < 2:
        return 0.0
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xl = np.maximum(x1[:, None], x1[None, :])
    xr = np.minimum(x2[:, None], x2[None, :])
    yt = np.maximum(y1[:, None], y1[None, :])
    yb = np.minimum(y2[:, None], y2[None, :])
    inter = np.clip(xr - xl, 0, None) * np.clip(yb - yt, 0, None)
    # Off-diagonal upper triangle = each unordered pair once.
    total_intersection = (inter.sum() - np.trace(inter)) / 2.0
    total_area = float(((x2 - x1) * (y2 - y1)).sum())
    return float(total_intersection / total_area) if total_area > 0 else 0.0


def compute_bbox_area_stats(boxes: np.ndarray, frame_area: float) -> tuple[float, float]:
    """(mean, variance) of box areas as a *fraction of frame area*.

    Normalising by frame area keeps the feature comparable across the mixed
    resolutions in the dataset (the raw-pixel spec would be resolution-biased).
    """
    if len(boxes) == 0:
        return 0.0, 0.0
    areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) / frame_area
    return float(areas.mean()), float(areas.var())


def compute_detection_confidence(conf: np.ndarray) -> tuple[float, float]:
    """(mean, std) YOLO confidence; low mean signals occlusion/crush."""
    if len(conf) == 0:
        return 0.0, 0.0
    return float(conf.mean()), float(conf.std())


def compute_spatial_density_mismatch(frame: np.ndarray, boxes: np.ndarray) -> float:
    """Pixel-density vs head-count mismatch over a 2x2 grid.

    High dark-pixel fill with few detected heads in a quadrant indicates a
    compressed/occluded mass. Deviations from the reference spec, both
    intentional: (1) the ``/255`` on a boolean mean is dropped (it would
    collapse the feature to ~0), and (2) people are assigned to a quadrant by
    box *centre* so boundary-spanning boxes are still counted.
    """
    n = len(boxes)
    if n == 0:
        return 0.0
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    quadrants = [(0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2),
                 (0, h // 2, w // 2, h), (w // 2, h // 2, w, h)]
    mismatches = []
    for x1, y1, x2, y2 in quadrants:
        region = gray[y1:y2, x1:x2]
        if region.size == 0:
            continue
        pixel_density = float((region < 100).mean())  # fraction of dark pixels
        in_quad = int(((cx >= x1) & (cx < x2) & (cy >= y1) & (cy < y2)).sum())
        person_density = in_quad / n
        mismatches.append(max(0.0, pixel_density - person_density))
    return float(np.mean(mismatches)) if mismatches else 0.0


def compute_compression(centroids: np.ndarray, area: float, diag: float
                        ) -> tuple[float, float]:
    """Convex-hull density + mean nearest-neighbour spacing (the "gaps" signal).

    * ``hull_density`` = people per 100k px of the crowd's convex-hull area
      (analogue of ``crowd_density`` but over the *occupied* region only, so a
      tightly-packed Panic crowd scores much higher than a gap-filled Crowdy one
      of the same headcount). Falls back to frame-area density when the hull is
      undefined (< 3 points or collinear).
    * ``nn_distance_mean`` = mean distance from each person to their nearest
      neighbour, normalised by the frame diagonal (small = compressed).
    """
    n = len(centroids)
    if n == 0:
        return 0.0, 0.0
    # Nearest-neighbour spacing.
    if n >= 2:
        d = cdist(centroids, centroids)
        np.fill_diagonal(d, np.inf)
        nn_mean = float((d.min(axis=1) / diag).mean())
    else:
        nn_mean = 0.0
    # Convex-hull density.
    hull_area = None
    if n >= 3:
        try:
            from scipy.spatial import ConvexHull

            hull_area = float(ConvexHull(centroids).volume)  # 2D area
        except Exception:
            hull_area = None
    if hull_area and hull_area > 0:
        hull_density = n / hull_area * 1e5
    else:
        hull_density = n / area * 1e5  # fallback: whole-frame density
    return float(hull_density), nn_mean


def _instant_rows(path: str, detector: Detector, stride: int | None = None,
                  frame_transform=None) -> pd.DataFrame:
    """Compute the per-frame *instantaneous* (non-windowed) feature series.

    ``stride`` overrides the default frame sampling stride (used for speed-
    perturbation augmentation: a smaller stride simulates slower motion, a
    larger stride faster). ``frame_transform`` is an optional ``frame -> frame``
    callable applied before detection/flow (brightness or crop/zoom
    augmentation).
    """
    stride = FRAME_STRIDE if stride is None else max(1, int(stride))
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    diag = float(np.hypot(width, height)) or 1.0
    area = float(width * height) or 1.0
    dt = stride / fps  # seconds between sampled frames

    tracker = CentroidTracker(max_dist=0.12 * max(width, height))
    prev_gray: np.ndarray | None = None
    rows: list[dict] = []
    raw_idx = 0
    kept = 0
    while kept < MAX_FRAMES_PER_CLIP:
        ok, frame = cap.read()
        if not ok:
            break
        if raw_idx % stride != 0:
            raw_idx += 1
            continue
        raw_idx += 1
        kept += 1

        if frame_transform is not None:
            frame = frame_transform(frame)
        boxes, confidences, count, used_fallback = detector(frame)
        displacements = tracker.update(boxes)
        csrnet_count, csrnet_peak_density = detector.csrnet_density(frame)
        # How much CSRNet's density estimate exceeds YOLO's raw box count -
        # large values flag occlusion/crush that YOLO is undercounting.
        csrnet_vs_yolo_ratio = csrnet_count / max(count, 1)

        # E) compression / occlusion features.
        bbox_overlap_ratio = compute_bbox_overlap_ratio(boxes)
        conf_mean, conf_std = compute_detection_confidence(confidences)
        bbox_area_mean, bbox_area_variance = compute_bbox_area_stats(boxes, area)
        spatial_density_mismatch = compute_spatial_density_mismatch(frame, boxes)
        # F) compression ratio (Crowdy-vs-Panic gaps).
        centroids = (boxes[:, :2] + boxes[:, 2:]) / 2.0 if len(boxes) else \
            np.empty((0, 2))
        hull_density, nn_distance_mean = compute_compression(centroids, area, diag)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Downscale to a fixed width for fast, resolution-consistent flow.
        gray = cv2.resize(
            gray, (FLOW_WIDTH, max(1, int(FLOW_WIDTH * gray.shape[0] / gray.shape[1])))
        )
        flow_diag = float(np.hypot(*gray.shape[:2])) or 1.0
        flow_mag_mean = flow_mag_var = 0.0
        dir_coherence = 0.0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # Normalise by frame diagonal -> fraction of frame moved per step.
            flow_mag_mean = float(mag.mean()) / flow_diag
            flow_mag_var = float(mag.var()) / (flow_diag ** 2)
            # Mean resultant length of flow directions: 1 = coherent, 0 = scatter.
            dir_coherence = float(
                np.hypot(np.cos(ang).mean(), np.sin(ang).mean())
            )
        prev_gray = gray

        # Per-person velocity as fraction of frame diagonal per second.
        if displacements.size:
            speeds = displacements / diag / dt
            velocity_per_person = float(speeds.mean())
            velocity_var = float(speeds.var())
        else:
            velocity_per_person = velocity_var = 0.0

        # Spatial dispersion of the current centroids (control-loss proxy).
        cent = tracker.prev
        if cent is not None and len(cent) > 1:
            trajectory_dispersion = float(
                np.hypot(cent[:, 0].std(), cent[:, 1].std()) / diag
            )
        else:
            trajectory_dispersion = 0.0

        rows.append(
            dict(
                frame_index=raw_idx - 1,
                used_fallback=int(used_fallback),
                person_count=float(count),
                crowd_density=count / area * 1e5,  # people per 100k px
                density_norm=float(1.0 - np.exp(-count / 50.0)),
                flow_mag_mean=flow_mag_mean,
                flow_mag_var=flow_mag_var,
                velocity_per_person=velocity_per_person,
                velocity_var=velocity_var,
                direction_consistency=dir_coherence,
                trajectory_dispersion=trajectory_dispersion,
                bbox_overlap_ratio=bbox_overlap_ratio,
                detection_confidence_mean=conf_mean,
                detection_confidence_std=conf_std,
                bbox_area_variance=bbox_area_variance,
                bbox_area_mean=bbox_area_mean,
                spatial_density_mismatch=spatial_density_mismatch,
                hull_density=hull_density,
                nn_distance_mean=nn_distance_mean,
                csrnet_count=csrnet_count,
                csrnet_peak_density=csrnet_peak_density,
                csrnet_vs_yolo_ratio=csrnet_vs_yolo_ratio,
            )
        )
    cap.release()
    return pd.DataFrame(rows)


def _add_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the sliding-window temporal-instability / control-loss columns."""
    w = TEMPORAL_WINDOW
    if df.empty:
        for col in ("motion_instability", "stop_go", "density_trend",
                    "bbox_overlap_trend"):
            df[col] = []
        return df

    roll = df["flow_mag_mean"].rolling(w, min_periods=2)
    # C) motion instability: std of flow magnitude over the window.
    df["motion_instability"] = roll.std().fillna(0.0)
    # C) stop-go: energy of frame-to-frame flow changes (oscillation) in window.
    df["stop_go"] = (
        df["flow_mag_mean"].diff().abs().rolling(w, min_periods=2).mean().fillna(0.0)
    )
    # direction_consistency is per-frame; smooth it over the window.
    df["direction_consistency"] = (
        df["direction_consistency"].rolling(w, min_periods=1).mean().fillna(0.0)
    )

    # D) density trend: slope of person_count over the trailing window.
    def _slope(x: np.ndarray) -> float:
        if len(x) < 2:
            return 0.0
        return float(np.polyfit(np.arange(len(x)), x, 1)[0])

    df["density_trend"] = (
        df["person_count"].rolling(w, min_periods=2).apply(_slope, raw=True).fillna(0.0)
    )
    # E) compression trend: is overlap (crush) increasing over the window?
    df["bbox_overlap_trend"] = (
        df["bbox_overlap_ratio"].rolling(w, min_periods=2).apply(_slope, raw=True).fillna(0.0)
    )
    return df


def frame_features(path: str, detector: Detector, stride: int | None = None,
                   frame_transform=None) -> pd.DataFrame:
    """Extract the full per-frame feature table for a single clip.

    Returns a DataFrame with ``frame_index``, ``used_fallback`` and every column
    in :data:`config.FEATURE_COLUMNS`. Empty (0-row) if the clip is unreadable.
    ``stride`` and ``frame_transform`` enable augmentation (see ``_instant_rows``).
    """
    df = _instant_rows(path, detector, stride=stride, frame_transform=frame_transform)
    df = _add_window_features(df)
    if df.empty:
        return df
    # Guarantee column presence / ordering.
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df[["frame_index", "used_fallback", *FEATURE_COLUMNS]].fillna(0.0)
