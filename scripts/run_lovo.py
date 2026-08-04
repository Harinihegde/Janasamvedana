#!/usr/bin/env python3
"""Leave-One-Video-Out (LOVO) evaluation for A5 RF top-12 + hysteresis sweep.

Why LOVO: with only 23 Panic and 7 No-Panic *source videos*, 5-fold CV produces
empty/near-empty folds and unstable metrics. LOVO trains on all-but-one video
and tests on the held-out one, for every video, giving each video a
leakage-free out-of-fold (OOF) risk timeline.

Stage 1: per-video classification results (+ pooled averages).
Stage 2: the OOF-scored videos are paired into synthetic Safe->Risk sequences;
alerts use a HYSTERESIS rule (fire only after the score stays >= threshold for
K consecutive frames) and the alert threshold is swept 0.50..0.70.

Model = A5: RandomForest(class_weight={Safe:1, Risk:5}) on the 12 selected
features. NOTE: results here are WITHOUT data augmentation.
Outputs -> outputs_improved/lovo_*.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

from run_binary import build_binary_sequences
from stampede.classifier import Normalizer
from stampede.config import RANDOM_STATE
from stampede.escalation import clip_timelines, smooth

POS, NEG = "Stampede Risk", "Safe"
OUT = Path("outputs_improved")
FEATURES_CSV = Path("outputs_compression/frame_features.csv")
TOP12 = json.loads((OUT / "comparison_improve.json").read_text())["top12_features"]
COST_WEIGHT = {NEG: 1.0, POS: 5.0}
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
HYST_K = 3            # consecutive frames above threshold required to fire
TOLERANCE = 40        # frames within which an alert counts as a detection
COOLDOWN = 15
MAX_SEQUENCES = 46


# ---------------------------------------------------------------------------
# Hysteresis alert detector
# ---------------------------------------------------------------------------
def detect_hysteresis(risk_raw, threshold, k=HYST_K, cooldown=COOLDOWN):
    """Return alert frame indices. An alert fires on the frame that completes a
    run of ``k`` consecutive smoothed-risk samples >= ``threshold`` (so a single
    transient spike no longer triggers). Cooldown suppresses repeats."""
    risk = smooth(risk_raw)
    alerts, last, run = [], -10**9, 0
    for i in range(len(risk)):
        if risk[i] >= threshold:
            run += 1
            if run >= k and i - last > cooldown:
                alerts.append(i)      # confirmation frame
                last = i
        else:
            run = 0
    return alerts


# ---------------------------------------------------------------------------
# LOVO out-of-fold scoring
# ---------------------------------------------------------------------------
def lovo_scores(feats, cols):
    """Leave-one-video-out OOF P(risk).

    Training uses every row from other videos (including augmented rows when
    present); only *original* rows (``is_augmented == 0``) of the held-out video
    are scored/tested, so a held-out video's own augmentations never leak in.
    """
    videos = feats.video_id.unique().tolist()
    oof = np.full(len(feats), np.nan)
    is_orig = (feats.is_augmented.values == 0)
    vid_arr = feats.video_id.values
    t0 = time.time()
    for n, v in enumerate(videos, 1):
        held = vid_arr == v
        tr = feats.iloc[(~held).nonzero()[0]]
        te_idx = (held & is_orig).nonzero()[0]
        if len(te_idx) == 0:
            continue
        te = feats.iloc[te_idx]
        norm = Normalizer.fit(tr, cols)
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                     class_weight=COST_WEIGHT,
                                     random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(norm.transform(tr), tr.label.values)
        pidx = list(clf.classes_).index(POS)
        oof[te_idx] = clf.predict_proba(norm.transform(te))[:, pidx]
        if n % 10 == 0 or n == len(videos):
            print(f"  LOVO {n}/{len(videos)}  ({time.time()-t0:.0f}s)", flush=True)
    return oof


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--augmented-csv", type=Path, default=None,
                    help="augmented_features.csv to add to LOVO *training* only")
    ap.add_argument("--prefix", default="lovo",
                    help="output filename prefix (e.g. lovo_aug)")
    ap.add_argument("--hyst-k", type=int, default=HYST_K,
                    help="consecutive frames above threshold required to fire")
    ap.add_argument("--oof-cache", type=Path, default=None,
                    help="cache per-frame OOF risk here; reused if it exists so "
                         "changing --hyst-k/threshold needs no LOVO recompute")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if a.oof_cache and Path(a.oof_cache).exists():
        # Fast path: reuse cached per-frame OOF risk (hysteresis/threshold only).
        feats = pd.read_csv(a.oof_cache)
        print(f"Loaded cached OOF risk from {a.oof_cache} "
              f"({feats.path.nunique()} clips).", flush=True)
    else:
        feats = pd.read_csv(FEATURES_CSV)
        feats["is_augmented"] = 0
        if a.augmented_csv and Path(a.augmented_csv).exists():
            aug = pd.read_csv(a.augmented_csv)
            common = [c for c in feats.columns if c in aug.columns]
            feats = pd.concat([feats, aug[common]], ignore_index=True)
            feats["is_augmented"] = feats["is_augmented"].fillna(1).astype(int)
            print(f"Loaded {len(aug)} augmented rows "
                  f"({aug.path.nunique()} aug clips) for training.", flush=True)
        feats["class4"] = feats.label
        feats["label"] = np.where(feats.label == "Panic", POS, NEG)
        feats["y"] = (feats.label == POS).astype(int)

        feats["risk"] = lovo_scores(feats, TOP12)
        # Stage 1 & Stage 2 evaluate on ORIGINAL frames only (augmented=train-only).
        feats = feats[feats.is_augmented == 0].reset_index(drop=True)
        if a.oof_cache:
            feats[["path", "video_id", "frame_index", "class4", "label", "y",
                   "risk"]].to_csv(a.oof_cache, index=False)
            print(f"Cached OOF risk -> {a.oof_cache}", flush=True)

    # ------------------------------------------------------------------
    # Stage 1: per-video classification (clip = majority of frames at 0.5)
    # ------------------------------------------------------------------
    per_clip = (feats.assign(pred=(feats.risk >= 0.5).astype(int))
                .groupby("path")
                .agg(video_id=("video_id", "first"), class4=("class4", "first"),
                     y=("y", "first"), pred=("pred", "mean")).reset_index())
    per_clip["clip_pred"] = (per_clip.pred >= 0.5).astype(int)

    per_video = (per_clip.groupby("video_id")
                 .agg(class4=("class4", "first"), y=("y", "first"),
                      n_clips=("path", "size"),
                      mean_risk=("pred", "mean"),
                      clips_flagged=("clip_pred", "sum")).reset_index())
    per_video["true_binary"] = np.where(per_video.y == 1, POS, NEG)
    per_video["video_pred"] = np.where(
        per_video.clips_flagged >= per_video.n_clips / 2, POS, NEG)
    per_video["correct"] = per_video.true_binary == per_video.video_pred
    per_video = per_video.sort_values(["class4", "video_id"])
    per_video.to_csv(OUT / f"{a.prefix}_per_video.csv", index=False)

    # Pooled clip-level Stage-1 metrics (the LOVO "average").
    p, r, f1, _ = precision_recall_fscore_support(
        per_clip.y, per_clip.clip_pred, average="binary", pos_label=1, zero_division=0)
    stage1 = {
        "n_videos": int(per_video.shape[0]),
        "clip_level": {"accuracy": float((per_clip.y == per_clip.clip_pred).mean()),
                       "risk_precision": float(p), "risk_recall": float(r),
                       "risk_f1": float(f1)},
        "video_level_accuracy": float(per_video.correct.mean()),
        "per_class_video_recall": {
            c: float(g[g.y == 1].correct.mean()) if (g.y == 1).any()
            else (float(g.correct.mean()))
            for c, g in per_video.groupby("class4")},
    }
    (OUT / f"{a.prefix}_stage1_metrics.json").write_text(json.dumps(stage1, indent=2))

    # ------------------------------------------------------------------
    # Stage 2: hysteresis threshold sweep over LOVO-OOF Safe->Risk sequences
    # ------------------------------------------------------------------
    timelines = clip_timelines(feats)
    seqs = build_binary_sequences(timelines, max_sequences=MAX_SEQUENCES)
    stable = [g for _, g in timelines.items() if g.label.iloc[0] == NEG]

    sweep = []
    for thr in THRESHOLDS:
        det, lat = 0, []
        for s in seqs:
            alerts = detect_hysteresis(s["risk"], thr, k=a.hyst_k)
            hits = [al for al in alerts if abs(al - s["transition_frame"]) <= TOLERANCE]
            if hits:
                det += 1
                lat.append(s["transition_frame"] - min(hits))
        fp = sum(1 for g in stable
                 if detect_hysteresis(g.risk.to_numpy(), thr, k=a.hyst_k))
        sweep.append({
            "threshold": thr,
            "detection_rate": round(det / len(seqs), 3) if seqs else 0.0,
            "false_positive_rate": round(fp / len(stable), 3) if stable else 0.0,
            "median_latency_frames": round(float(np.median(lat)), 1) if lat else None,
            "mean_latency_frames": round(float(np.mean(lat)), 1) if lat else None,
            "n_detected": det, "n_sequences": len(seqs),
            "n_fp": fp, "n_stable": len(stable),
        })
        print(f"thr={thr:.2f}  det={sweep[-1]['detection_rate']:.2f} "
              f"FP={sweep[-1]['false_positive_rate']:.2f} "
              f"med_lat={sweep[-1]['median_latency_frames']} "
              f"mean_lat={sweep[-1]['mean_latency_frames']}", flush=True)

    sdf = pd.DataFrame(sweep)
    sdf.to_csv(OUT / f"{a.prefix}_threshold_sweep.csv", index=False)
    (OUT / f"{a.prefix}_threshold_sweep.json").write_text(
        json.dumps({"hysteresis_k": a.hyst_k, "tolerance": TOLERANCE,
                    "cooldown": COOLDOWN,
                    "augmentation": bool(a.augmented_csv) or bool(a.oof_cache),
                    "sweep": sweep, "stage1": stage1}, indent=2))

    # Plot: detection vs FP across thresholds (+ latency)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(sdf.threshold, sdf.detection_rate, "o-", color="#1a73e8", label="detection rate")
    ax1.plot(sdf.threshold, sdf.false_positive_rate, "s-", color="#d93025", label="false-positive rate")
    ax1.set_xlabel("Alert threshold"); ax1.set_ylabel("Rate"); ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(sdf.threshold, sdf.median_latency_frames, "^--", color="#188038", label="median latency")
    ax2.axhline(0, color="#188038", lw=0.6, alpha=0.4)
    ax2.set_ylabel("Median latency (frames, + = early)")
    ax1.legend(loc="center left"); ax2.legend(loc="center right")
    ax1.set_title(f"A5 RF top-12 — LOVO Stage 2 sweep (hysteresis k={a.hyst_k})")
    fig.tight_layout(); fig.savefig(OUT / f"{a.prefix}_threshold_sweep.png", dpi=130); plt.close(fig)

    print("\nStage 1 (LOVO):", json.dumps(stage1, indent=2))
    print("\nPanic per-video (LOVO):")
    pv = per_video[per_video.class4 == "Panic"]
    for _, row in pv.iterrows():
        print(f"  {row.video_id:34s} risk={row.mean_risk:.2f} "
              f"pred={row.video_pred:14s} {'OK' if row.correct else 'MISS'}")


if __name__ == "__main__":
    main()
