#!/usr/bin/env python3
"""Sweep Stage 2's new confirm_frames debounce on the 4-class production model.

Motivation: stampede/escalation.py's detect_alerts previously fired the moment
a single smoothed-risk frame crossed a band edge (with a velocity + recent-dip
gate). That is the escalation logic scripts/run_stage2.py actually ships
(README's real usage path) - and its false-positive rate on the (very small,
3-video) leakage-safe test split was 67%. Separately, scripts/run_lovo.py
already validated a K-consecutive-frame confirmation debounce, but only for
the *binary* Safe/Risk model. This script checks whether the same idea, now
wired into the production detect_alerts() as an additional confirm_frames
parameter, helps the 4-class model - evaluated on leakage-safe pooled
out-of-fold (OOF) risk scores across *all* 71 videos (not just the 3-video
test split) so the false-positive rate isn't measured on too few videos.

Outputs -> results/stage2_confirm_sweep.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stampede.classifier import train
from stampede.config import FEATURE_COLUMNS
from stampede.dataset import stratified_group_folds
from stampede.escalation import (
    build_synthetic_sequences,
    clip_timelines,
    evaluate_false_positives,
    evaluate_transitions,
)

FEATURES_CSV = Path("outputs/frame_features.csv")
OUT_JSON = Path("results/stage2_confirm_sweep.json")
OUT_MD = Path("results/stage2_confirm_sweep.md")
CONFIRM_VALUES = [1, 3, 4, 5, 6]
MAX_SEQUENCES = 40


def pooled_oof_risk(feats: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Leakage-safe pooled out-of-fold risk score for every frame.

    Mirrors stampede.evaluate.cross_validate's fold construction, but keeps the
    continuous risk_score() per frame (rather than just the discrete class
    vote) since that is what Stage 2 actually consumes.
    """
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
    for cf in CONFIRM_VALUES:
        trans = {}
        for seq_type in ("normal_to_crowdy", "crowdy_to_panic"):
            seqs = build_synthetic_sequences(timelines, seq_type, MAX_SEQUENCES)
            trans[seq_type] = evaluate_transitions(seqs, confirm_frames=cf)
        fp = evaluate_false_positives(timelines, confirm_frames=cf)
        rows.append(
            dict(
                confirm_frames=cf,
                nc_detection_rate=trans["normal_to_crowdy"]["detection_rate"],
                nc_mean_latency=trans["normal_to_crowdy"]["mean_latency_frames"],
                nc_n_sequences=trans["normal_to_crowdy"]["n_sequences"],
                cp_detection_rate=trans["crowdy_to_panic"]["detection_rate"],
                cp_mean_latency=trans["crowdy_to_panic"]["mean_latency_frames"],
                cp_n_sequences=trans["crowdy_to_panic"]["n_sequences"],
                false_positive_rate=fp["false_positive_rate"],
                n_stable_videos=fp["n_stable_videos"],
                mean_risk_variance=fp["mean_risk_variance"],
            )
        )
        print(
            f"confirm_frames={cf:2d}  NC det={rows[-1]['nc_detection_rate']:.2f} "
            f"lat={rows[-1]['nc_mean_latency']:.1f}  CP det={rows[-1]['cp_detection_rate']:.2f} "
            f"lat={rows[-1]['cp_mean_latency']:.1f}  FP={rows[-1]['false_positive_rate']:.2f} "
            f"(n_stable={rows[-1]['n_stable_videos']})",
            flush=True,
        )
    return rows


def write_report(rows: list[dict]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, indent=2))

    lines = [
        "# Stage 2 confirmation-frame sweep (4-class production model)",
        "",
        "Pooled 5-fold leakage-safe out-of-fold risk scores across all 71 videos "
        "(not just the 3-video test split), evaluated with detect_alerts()'s new "
        "`confirm_frames` debounce at the spec thresholds (0.5 / 0.8).",
        "",
        "| confirm_frames | Normal->Crowdy detect | latency | Crowdy->Panic detect | "
        "latency | False-positive rate | n_stable_videos |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['confirm_frames']} | {r['nc_detection_rate']:.0%} | "
            f"{r['nc_mean_latency']:.1f}f | {r['cp_detection_rate']:.0%} | "
            f"{r['cp_mean_latency']:.1f}f | {r['false_positive_rate']:.0%} | "
            f"{r['n_stable_videos']} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    feats = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(feats)} frames from {FEATURES_CSV}", flush=True)
    print("Computing pooled leakage-safe OOF risk scores (5-fold)...", flush=True)
    oof = pooled_oof_risk(feats)
    timelines = clip_timelines(oof)
    print(f"Built {len(timelines)} video timelines. Sweeping confirm_frames...", flush=True)
    rows = sweep(timelines)
    write_report(rows)
    print(f"\nWrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
