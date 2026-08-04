"""Central configuration: class ordering, risk mapping, thresholds, feature names.

Keeping every "magic number" in one module makes the pipeline auditable and lets
Stage 2 thresholds be tuned without touching the detection code.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
# Ordinal ordering from safest to most dangerous. The escalation logic in
# Stage 2 relies on this being monotonic in risk.
CLASSES = ("No Panic", "Normal", "Crowdy", "Panic")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Continuous risk anchor assigned to each class. A frame's scalar risk score is
# the RandomForest class-probability expectation over these anchors:
#     risk = sum_c  P(class=c) * RISK_ANCHOR[c]
# The anchors are chosen so that a confident single-class prediction lands
# inside the Stage 2 band for that class:
#   No Panic -> 0.10   Normal -> 0.35 (< 0.40)   Crowdy -> 0.60 (>= 0.50)
#   Panic    -> 0.90 (>= 0.80)
RISK_ANCHOR = {"No Panic": 0.10, "Normal": 0.35, "Crowdy": 0.60, "Panic": 0.90}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 20260722

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
# We sample frames at a fixed temporal stride rather than a fixed count so that
# optical flow / velocity are computed between *temporally adjacent* sampled
# frames (a fixed count would change the effective dt per clip and make motion
# features incomparable across clips of different length).
FRAME_STRIDE = 4            # keep every 4th frame (~7.5 fps at 30 fps source)
MAX_FRAMES_PER_CLIP = 80    # safety cap on very long clips
TEMPORAL_WINDOW = 10        # sliding window (in *sampled* frames) for instability
# Optical flow is computed on a frame downscaled to this width (aspect
# preserved) for speed; magnitudes are normalised by the downscaled diagonal so
# the features stay resolution-invariant regardless of source size.
FLOW_WIDTH = 320
YOLO_CONF = 0.25
YOLO_IMGSZ = 512            # 512 halves inference time vs 640 with stable counts
YOLO_PERSON_CLASS = 0       # COCO "person"
# If YOLO returns fewer than this many boxes on a frame whose raw pixels look
# dense (high edge energy), fall back to a density-map estimate.
CSRNET_FALLBACK_MIN_COUNT = 3

# Per-frame feature columns produced by features.frame_features (order matters
# for the normalization params saved to disk).
FEATURE_COLUMNS = [
    # A) density
    "person_count",
    "crowd_density",
    "density_norm",
    # B) motion
    "flow_mag_mean",
    "flow_mag_var",
    "velocity_per_person",
    "velocity_var",
    # C) temporal instability (sliding window)
    "motion_instability",
    "stop_go",
    "direction_consistency",
    # D) control loss
    "trajectory_dispersion",
    "density_trend",
    # E) compression / occlusion (distinguishes Panic-crush from dense-Crowdy):
    #    people crushed together overlap more, detect with lower confidence, and
    #    show a pixel-vs-headcount density mismatch. bbox areas are normalised by
    #    frame area so they stay resolution-invariant.
    "bbox_overlap_ratio",
    "bbox_overlap_trend",
    "detection_confidence_mean",
    "detection_confidence_std",
    "bbox_area_variance",
    "bbox_area_mean",
    "spatial_density_mismatch",
    # F) compression ratio - the Crowdy-vs-Panic "gaps" signal. Panic packs
    #    bodies with near-zero gaps (high hull density, tiny nearest-neighbour
    #    spacing); Crowdy is dense but has visible gaps.
    "hull_density",
    "nn_distance_mean",
]

# ---------------------------------------------------------------------------
# Stage 2 - escalation detection
# ---------------------------------------------------------------------------
SMOOTH_WINDOW = 5           # moving-average window for risk smoothing (frames)
VELOCITY_WINDOW = 10        # window for dRisk/dt estimate (frames)

# Normal -> Crowdy alert.
NC_LOW = 0.40               # must be rising from below this
NC_HIGH = 0.50              # ... to at/above this
NC_VELOCITY = 0.004         # min dRisk/dt (per frame) to count as "rising"

# Crowdy -> Panic alert.
CP_LOW = 0.70
CP_HIGH = 0.80
CP_VELOCITY = 0.006

ALERT_COOLDOWN = 15         # frames to suppress duplicate alerts of same type

MESSAGES = {
    "normal_to_crowdy": "⚠️ Crowd density increasing. Monitor situation. "
    "Redirect flow, open additional exits.",
    "crowdy_to_panic": "\U0001f6a8 CRITICAL: Stampede risk detected. Evacuate now. "
    "Close gates, activate emergency response, alert medics.",
}
