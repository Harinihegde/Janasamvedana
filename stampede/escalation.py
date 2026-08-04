"""Stage 2: escalation detection over continuous risk-score timelines.

Consumes the per-frame risk score from Stage 1 and emits two alert types:

* **Normal -> Crowdy** (low/medium urgency): risk rising through ~0.5.
* **Crowdy -> Panic** (critical): risk rising through ~0.8.

Because the dataset contains no naturally-continuous cross-class recordings,
transition *detection rate* and *latency* are measured on **synthetic
escalation sequences** built by concatenating held-out single-class clip
timelines (with known join points as ground-truth transitions). False-positive
rate and risk smoothness are measured on **real** single-class timelines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ALERT_COOLDOWN,
    CP_HIGH,
    CP_LOW,
    CP_VELOCITY,
    MESSAGES,
    NC_HIGH,
    NC_LOW,
    NC_VELOCITY,
    SMOOTH_WINDOW,
    VELOCITY_WINDOW,
)


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------
def smooth(risk: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Causal moving-average smoothing (uses only past/current frames)."""
    risk = np.asarray(risk, dtype=float)
    if len(risk) == 0:
        return risk
    out = np.empty_like(risk)
    for i in range(len(risk)):
        lo = max(0, i - window + 1)
        out[i] = risk[lo : i + 1].mean()
    return out


def risk_velocity(risk: np.ndarray, window: int = VELOCITY_WINDOW) -> np.ndarray:
    """dRisk/dt (per frame) via least-squares slope over a trailing window."""
    risk = np.asarray(risk, dtype=float)
    vel = np.zeros_like(risk)
    for i in range(len(risk)):
        lo = max(0, i - window + 1)
        seg = risk[lo : i + 1]
        if len(seg) >= 2:
            vel[i] = np.polyfit(np.arange(len(seg)), seg, 1)[0]
    return vel


# ---------------------------------------------------------------------------
# Alert detection
# ---------------------------------------------------------------------------
def detect_alerts(risk_raw: np.ndarray, thresholds: dict | None = None) -> dict:
    """Detect escalation alerts on a raw per-frame risk timeline.

    ``thresholds`` optionally overrides the config band edges/velocities with
    keys ``nc_low, nc_high, nc_velocity, cp_low, cp_high, cp_velocity`` - used
    to report a data-calibrated trade-off alongside the spec defaults.

    Returns a dict with the smoothed risk, velocity, and a list of alert
    records ``{type, frame, risk, velocity, confidence, message}``.
    """
    th = {
        "nc_low": NC_LOW, "nc_high": NC_HIGH, "nc_velocity": NC_VELOCITY,
        "cp_low": CP_LOW, "cp_high": CP_HIGH, "cp_velocity": CP_VELOCITY,
    }
    if thresholds:
        th.update(thresholds)
    risk = smooth(risk_raw)
    vel = risk_velocity(risk)
    alerts: list[dict] = []
    last_fire = {"normal_to_crowdy": -10**9, "crowdy_to_panic": -10**9}

    def _recent_below(i: int, thresh: float, window: int = 30) -> bool:
        lo = max(0, i - window)
        return risk[lo : i + 1].min() < thresh

    for i in range(1, len(risk)):
        # Crowdy -> Panic (check first: higher urgency dominates).
        if (
            risk[i - 1] < th["cp_high"]
            and risk[i] >= th["cp_high"]
            and vel[i] > th["cp_velocity"]
            and _recent_below(i, th["cp_low"])
            and i - last_fire["crowdy_to_panic"] > ALERT_COOLDOWN
        ):
            alerts.append(
                dict(
                    type="crowdy_to_panic",
                    frame=int(i),
                    risk=float(risk[i]),
                    velocity=float(vel[i]),
                    confidence=float(min(1.0, vel[i] / th["cp_velocity"] / 3.0 + 0.5)),
                    message=MESSAGES["crowdy_to_panic"],
                )
            )
            last_fire["crowdy_to_panic"] = i
            continue

        # Normal -> Crowdy.
        if (
            risk[i - 1] < th["nc_high"]
            and risk[i] >= th["nc_high"]
            and vel[i] > th["nc_velocity"]
            and _recent_below(i, th["nc_low"])
            and i - last_fire["normal_to_crowdy"] > ALERT_COOLDOWN
        ):
            alerts.append(
                dict(
                    type="normal_to_crowdy",
                    frame=int(i),
                    risk=float(risk[i]),
                    velocity=float(vel[i]),
                    confidence=float(min(1.0, vel[i] / th["nc_velocity"] / 3.0 + 0.5)),
                    message=MESSAGES["normal_to_crowdy"],
                )
            )
            last_fire["normal_to_crowdy"] = i

    return {"risk_smoothed": risk, "velocity": vel, "alerts": alerts}


# ---------------------------------------------------------------------------
# Timeline construction from scored frames
# ---------------------------------------------------------------------------
def clip_timelines(scored: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group scored frames into per-video timelines, ordered by clip then frame.

    ``scored`` must contain ``path``, ``video_id``, ``label``, ``frame_index``
    and ``risk`` columns.
    """
    out: dict[str, pd.DataFrame] = {}
    for vid, g in scored.groupby("video_id", sort=False):
        g = g.sort_values(["path", "frame_index"]).reset_index(drop=True)
        out[vid] = g
    return out


def build_synthetic_sequences(
    timelines: dict[str, pd.DataFrame],
    seq_type: str,
    max_sequences: int = 20,
) -> list[dict]:
    """Concatenate single-class timelines into escalation sequences.

    ``seq_type`` is ``"normal_to_crowdy"`` or ``"crowdy_to_panic"``. Each result
    holds the concatenated raw risk array and the ground-truth transition frame
    index (the join between the two segments).
    """
    if seq_type == "normal_to_crowdy":
        a_label, b_label = "Normal", "Crowdy"
    elif seq_type == "crowdy_to_panic":
        a_label, b_label = "Crowdy", "Panic"
    else:
        raise ValueError(seq_type)

    a_vids = [v for v, g in timelines.items() if g.label.iloc[0] == a_label]
    b_vids = [v for v, g in timelines.items() if g.label.iloc[0] == b_label]
    sequences: list[dict] = []
    for k in range(min(max_sequences, len(a_vids) * len(b_vids))):
        a = timelines[a_vids[k % len(a_vids)]]
        b = timelines[b_vids[k % len(b_vids)]]
        risk = np.concatenate([a.risk.to_numpy(), b.risk.to_numpy()])
        sequences.append(
            dict(
                type=seq_type,
                risk=risk,
                transition_frame=int(len(a)),
                a_video=a_vids[k % len(a_vids)],
                b_video=b_vids[k % len(b_vids)],
            )
        )
    return sequences


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_transitions(
    sequences: list[dict], tolerance: int = 40, thresholds: dict | None = None
) -> dict:
    """Detection rate + latency for synthetic escalation sequences.

    A detection counts if an alert of the matching type fires within
    ``tolerance`` frames of the ground-truth transition. Latency is
    ``transition_frame - alert_frame`` (positive = early warning).
    """
    detected = 0
    latencies: list[int] = []
    per_seq = []
    for seq in sequences:
        res = detect_alerts(seq["risk"], thresholds)
        hits = [
            a
            for a in res["alerts"]
            if a["type"] == seq["type"]
            and abs(a["frame"] - seq["transition_frame"]) <= tolerance
        ]
        if hits:
            detected += 1
            # earliest matching alert
            first = min(hits, key=lambda a: a["frame"])
            latencies.append(seq["transition_frame"] - first["frame"])
        per_seq.append(
            dict(
                a_video=seq["a_video"],
                b_video=seq["b_video"],
                transition_frame=seq["transition_frame"],
                detected=bool(hits),
                n_alerts=len(res["alerts"]),
            )
        )
    n = len(sequences)
    return {
        "n_sequences": n,
        "detection_rate": detected / n if n else 0.0,
        "mean_latency_frames": float(np.mean(latencies)) if latencies else 0.0,
        "median_latency_frames": float(np.median(latencies)) if latencies else 0.0,
        "per_sequence": per_seq,
    }


def evaluate_false_positives(
    timelines: dict[str, pd.DataFrame], thresholds: dict | None = None
) -> dict:
    """False-positive rate & risk smoothness on real single-class timelines.

    A false positive is any escalation alert fired on a timeline whose clips are
    *stable* (single class) and one class below the alert's target, i.e. an
    unwarranted escalation warning.
    """
    stable_labels = {"No Panic", "Normal"}
    fp_nc = fp_total = 0
    variances = []
    per_video = []
    for vid, g in timelines.items():
        risk = g.risk.to_numpy()
        res = detect_alerts(risk, thresholds)
        variances.append(float(np.var(res["risk_smoothed"])))
        n_alerts = len(res["alerts"])
        if g.label.iloc[0] in stable_labels:
            fp_total += 1
            if n_alerts > 0:
                fp_nc += 1
        per_video.append(
            dict(video_id=vid, label=g.label.iloc[0], n_alerts=n_alerts)
        )
    return {
        "n_stable_videos": fp_total,
        "false_positive_rate": fp_nc / fp_total if fp_total else 0.0,
        "mean_risk_variance": float(np.mean(variances)) if variances else 0.0,
        "per_video": per_video,
    }
