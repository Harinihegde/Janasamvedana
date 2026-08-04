#!/usr/bin/env python3
"""4-class before/after test for the compression-ratio features.

Trains the 4-class RandomForest with and without the two new compression
features (``hull_density``, ``nn_distance_mean``) on the SAME frames and the
SAME leakage-safe StratifiedGroupKFold folds, then reports:
  * the Crowdy<->Panic confusion block (the boundary these features target),
  * per-class precision/recall/F1 for Crowdy and Panic,
  * overall accuracy / macro-F1 / weighted-F1.

Metrics are clip-level from cross-validation out-of-fold predictions.
Output -> outputs_improved/compression2_report.{json,md} + confusion PNGs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from stampede.classifier import train
from stampede.config import CLASSES
from stampede.dataset import stratified_group_folds
from stampede.evaluate import clip_vote
from stampede.visualize import plot_confusion

OUT = Path("outputs_improved")
FEATURES_CSV = Path("outputs_compression2/frame_features.csv")

# The 19 features that existed BEFORE this change (base 12 + 7 occlusion).
BEFORE_19 = [
    "person_count", "crowd_density", "density_norm", "flow_mag_mean",
    "flow_mag_var", "velocity_per_person", "velocity_var", "motion_instability",
    "stop_go", "direction_consistency", "trajectory_dispersion", "density_trend",
    "bbox_overlap_ratio", "bbox_overlap_trend", "detection_confidence_mean",
    "detection_confidence_std", "bbox_area_variance", "bbox_area_mean",
    "spatial_density_mismatch",
]
NEW = ["hull_density", "nn_distance_mean"]
AFTER_21 = BEFORE_19 + NEW


def cv_oof(feats, cols, folds, clips):
    """Return clip-level OOF (actual, predicted) using the given feature cols."""
    parts = []
    for tr_idx, te_idx in folds:
        tr_p = set(clips.loc[tr_idx, "path"]); te_p = set(clips.loc[te_idx, "path"])
        tr = feats[feats.path.isin(tr_p)]; te = feats[feats.path.isin(te_p)]
        model = train(tr, feature_cols=cols)
        pred = model.predict(te)
        parts.append(clip_vote(te, pred))
    oof = pd.concat(parts, ignore_index=True)
    return oof


def metrics(oof, imp=None):
    y, p = oof.actual.values, oof.predicted.values
    rep = classification_report(y, p, labels=list(CLASSES), output_dict=True,
                                zero_division=0)
    cm = confusion_matrix(y, p, labels=list(CLASSES))
    ci = {c: i for i, c in enumerate(CLASSES)}
    cp = ci["Crowdy"], ci["Panic"]
    block = [[int(cm[cp[0]][cp[0]]), int(cm[cp[0]][cp[1]])],
             [int(cm[cp[1]][cp[0]]), int(cm[cp[1]][cp[1]])]]
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
        "crowdy": {k: rep["Crowdy"][k] for k in ("precision", "recall", "f1-score")},
        "panic": {k: rep["Panic"][k] for k in ("precision", "recall", "f1-score")},
        "confusion_full": cm.tolist(),
        "crowdy_panic_block": block,  # rows/cols = [Crowdy, Panic]
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(FEATURES_CSV)
    clips = (feats[["path", "video_id", "label"]].drop_duplicates("path")
             .reset_index(drop=True))
    folds = stratified_group_folds(clips, n_splits=5)

    before = metrics(cv_oof(feats, BEFORE_19, folds, clips))
    after_oof = cv_oof(feats, AFTER_21, folds, clips)
    after = metrics(after_oof)

    # Feature importance of the new features (fit once on all data).
    model_all = train(feats, feature_cols=AFTER_21)
    imp = dict(sorted(zip(AFTER_21, model_all.rf.feature_importances_),
                      key=lambda kv: kv[1], reverse=True))
    new_ranks = {f: (list(imp).index(f) + 1, round(imp[f], 4)) for f in NEW}

    report = {"before_19_features": before, "after_21_features": after,
              "new_features": NEW, "new_feature_rank_importance": new_ranks,
              "n_features_total": len(AFTER_21)}
    (OUT / "compression2_report.json").write_text(json.dumps(report, indent=2))

    # Confusion PNGs (full 4-class) before/after.
    plot_confusion(before["confusion_full"], list(CLASSES),
                   OUT / "compression2_confusion_before.png",
                   f"Before (19 feat) — macroF1={before['macro_f1']:.3f}")
    plot_confusion(after["confusion_full"], list(CLASSES),
                   OUT / "compression2_confusion_after.png",
                   f"After (21 feat) — macroF1={after['macro_f1']:.3f}")

    def blk(m):
        b = m["crowdy_panic_block"]
        return (f"      pred:Crowdy pred:Panic\n"
                f"  Crowdy   {b[0][0]:5d}     {b[0][1]:5d}\n"
                f"  Panic    {b[1][0]:5d}     {b[1][1]:5d}")

    L = ["# Compression-ratio features — 4-class before/after (CV OOF, clip-level)\n",
         f"New features: {NEW}  (ranks/importance: {new_ranks})\n",
         "## Crowdy<->Panic confusion block (rows=actual, cols=pred)\n",
         "### Before (19 features)\n```", blk(before), "```",
         "### After (21 features)\n```", blk(after), "```\n",
         "## Per-class & overall (Crowdy / Panic are the target)\n",
         "| Metric | Before (19) | After (21) | Δ |", "|---|---|---|---|"]
    for lab, key in [("Crowdy precision", ("crowdy", "precision")),
                     ("Crowdy recall", ("crowdy", "recall")),
                     ("Crowdy F1", ("crowdy", "f1-score")),
                     ("Panic precision", ("panic", "precision")),
                     ("Panic recall", ("panic", "recall")),
                     ("Panic F1", ("panic", "f1-score"))]:
        bv = before[key[0]][key[1]]; av = after[key[0]][key[1]]
        L.append(f"| {lab} | {bv:.3f} | {av:.3f} | {av-bv:+.3f} |")
    for lab, key in [("Overall accuracy", "accuracy"),
                     ("Overall macro-F1", "macro_f1"),
                     ("Overall weighted-F1", "weighted_f1")]:
        bv, av = before[key], after[key]
        L.append(f"| {lab} | {bv:.3f} | {av:.3f} | {av-bv:+.3f} |")
    (OUT / "compression2_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
