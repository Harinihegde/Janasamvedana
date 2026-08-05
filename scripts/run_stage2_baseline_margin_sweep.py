#!/usr/bin/env python3
"""Sweep Stage 2's relative-rise gate (baseline_window / min_rise) on top of
the existing confirm_frames=3 default.

Motivation: results/stage2_confirm_sweep.md already found that confirm_frames
plateaus at a 53% false-positive rate (on the 15 leakage-safe stable No
Panic/Normal videos) with no further gain from longer debounce windows.
Inspecting *which* videos still false-alert shows why: several have Stage 1
risk sustainedly elevated for hundreds of frames (e.g. one Normal video spends
431 consecutive smoothed-risk frames above the Normal->Crowdy threshold, and
another's mean risk is already 0.525 - above the 0.50 alert line) - not brief
noise a longer debounce could catch. A fixed absolute threshold can't tell
that video apart from a genuine transition; a relative-rise gate might, since
a genuine transition rises sharply above its own recent past while a
systematically-elevated-but-fluctuating video does not.

This sweeps stampede.escalation.detect_alerts's new nc_min_rise/cp_min_rise/
baseline_window thresholds (see stampede/config.py) on the same pooled
leakage-safe OOF risk scores used by the confirm_frames sweep, to check
whether it cuts the false-positive rate further without costing detection
rate.

Outputs -> results/stage2_baseline_margin_sweep.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stampede.classifier import train
from stampede.config import CONFIRM_FRAMES, FEATURE_COLUMNS
from stampede.dataset import stratified_group_folds
from stampede.escalation import (
    build_synthetic_sequences,
    clip_timelines,
    evaluate_false_positives,
    evaluate_transitions,
)

FEATURES_CSV = Path("outputs/frame_features.csv")
OUT_JSON = Path("results/stage2_baseline_margin_sweep.json")
OUT_MD = Path("results/stage2_baseline_margin_sweep.md")
MAX_SEQUENCES = 40

# (baseline_window, nc_min_rise, cp_min_rise). Row 0 reproduces the current
# default (gate off) as the baseline to compare against.
GRID = [
    (90, 0.00, 0.00),
    (60, 0.10, 0.10),
    (90, 0.10, 0.10),
    (150, 0.10, 0.10),
    (90, 0.15, 0.15),
    (150, 0.15, 0.15),
    (90, 0.20, 0.20),
    (150, 0.20, 0.20),
]


def pooled_oof_risk(feats: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Leakage-safe pooled out-of-fold risk score for every frame (see
    scripts/run_stage2_confirm_sweep.py's identical helper)."""
    clips = feats[["path", "video_id", "label"]].drop_duplicates("path").reset_index(drop=True)
    folds = stratified_group_folds(clips, n_splits=n_splits)
    oof_parts = []
    for fi, (tr_idx, te_idx) in enumerate(folds, 1):
        tr_paths = set(clips.loc[tr_idx, "path"])
        te_paths = set(clips.loc[te_idx, "path"])
        tr = feats[feats.path.isin(tr_paths)]
        te = feats[feats.path.isin(te_paths)].copy()
        model = train(tr, feature_cols=FEATURE_COLUMNS)
        te["risk"] = model.risk_score(te)
        oof_parts.append(te[["path", "video_id", "label", "frame_index", "risk"]])
        print(f"  fold {fi}/{len(folds)} done ({len(te)} frames)", flush=True)
    return pd.concat(oof_parts, ignore_index=True)


def sweep(timelines: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for baseline_window, nc_min_rise, cp_min_rise in GRID:
        th = dict(
            baseline_window=baseline_window,
            nc_min_rise=nc_min_rise,
            cp_min_rise=cp_min_rise,
        )
        trans = {}
        for seq_type in ("normal_to_crowdy", "crowdy_to_panic"):
            seqs = build_synthetic_sequences(timelines, seq_type, MAX_SEQUENCES)
            trans[seq_type] = evaluate_transitions(
                seqs, thresholds=th, confirm_frames=CONFIRM_FRAMES
            )
        fp = evaluate_false_positives(timelines, thresholds=th, confirm_frames=CONFIRM_FRAMES)
        rows.append(
            dict(
                baseline_window=baseline_window,
                nc_min_rise=nc_min_rise,
                cp_min_rise=cp_min_rise,
                nc_detection_rate=trans["normal_to_crowdy"]["detection_rate"],
                nc_mean_latency=trans["normal_to_crowdy"]["mean_latency_frames"],
                cp_detection_rate=trans["crowdy_to_panic"]["detection_rate"],
                cp_mean_latency=trans["crowdy_to_panic"]["mean_latency_frames"],
                false_positive_rate=fp["false_positive_rate"],
                n_stable_videos=fp["n_stable_videos"],
                n_stable_alerting=round(fp["false_positive_rate"] * fp["n_stable_videos"]),
            )
        )
        r = rows[-1]
        print(
            f"window={baseline_window:3d} nc_rise={nc_min_rise:.2f} cp_rise={cp_min_rise:.2f}  "
            f"NC det={r['nc_detection_rate']:.2f} CP det={r['cp_detection_rate']:.2f}  "
            f"FP={r['false_positive_rate']:.2f} ({r['n_stable_alerting']}/{r['n_stable_videos']})",
            flush=True,
        )
    return rows


def write_report(rows: list[dict]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, indent=2))

    lines = [
        "# Stage 2 relative-rise gate sweep (4-class production model)",
        "",
        "Pooled 5-fold leakage-safe out-of-fold risk scores across all 71 "
        "videos, `confirm_frames=3` (current default) fixed, sweeping the new "
        "`nc_min_rise`/`cp_min_rise`/`baseline_window` relative-rise gate on "
        "top of it.",
        "",
        "| baseline_window | min_rise (NC/CP) | Normal->Crowdy detect | "
        "Crowdy->Panic detect | False-positive rate | stable videos alerting |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['baseline_window']} | {r['nc_min_rise']:.2f}/{r['cp_min_rise']:.2f} | "
            f"{r['nc_detection_rate']:.0%} | {r['cp_detection_rate']:.0%} | "
            f"{r['false_positive_rate']:.0%} | {r['n_stable_alerting']}/{r['n_stable_videos']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    feats = pd.read_csv(FEATURES_CSV)
    for col in FEATURE_COLUMNS:
        if col not in feats.columns:
            feats[col] = 0.0
    print(f"Loaded {len(feats)} frames from {FEATURES_CSV}", flush=True)
    print("Computing pooled leakage-safe OOF risk scores (5-fold)...", flush=True)
    oof = pooled_oof_risk(feats)
    timelines = clip_timelines(oof)
    print(f"Built {len(timelines)} video timelines. Sweeping relative-rise gate...", flush=True)
    rows = sweep(timelines)
    write_report(rows)
    print(f"\nWrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
