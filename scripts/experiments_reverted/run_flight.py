#!/usr/bin/env python3
"""4-class before/after test for the flight/scatter motion features.

Merges flow_extra.csv (flow_dir_divergence, flight_score) into the 21-feature
set and trains the 4-class RF before (21) vs after (23) on identical frames /
leakage-safe StratifiedGroupKFold folds. Reports:
  * recall on the 25 MOTION_RUNNING clips (all actual Panic - flight-panic that
    the density/compression features miss), before vs after,
  * overall Panic & Crowdy P/R/F1 + accuracy/macro-F1 (CV out-of-fold, clip).

Output -> outputs_improved/flight_report.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, f1_score)

from run_compression2 import BEFORE_19, cv_oof  # reuse OOF harness
from stampede.config import CLASSES
from stampede.dataset import stratified_group_folds

OUT = Path("outputs_improved")
FEATURES_CSV = Path("outputs_compression2/frame_features.csv")
FLOW_EXTRA = OUT / "flow_extra.csv"
TRIAGE = OUT / "panic_lowdensity_all/triage.csv"

BEFORE_21 = BEFORE_19 + ["hull_density", "nn_distance_mean"]
NEW = ["flow_dir_divergence", "flight_score"]
AFTER_23 = BEFORE_21 + NEW


def overall(oof):
    y, p = oof.actual.values, oof.predicted.values
    rep = classification_report(y, p, labels=list(CLASSES), output_dict=True,
                                zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "panic": {k: rep["Panic"][k] for k in ("precision", "recall", "f1-score")},
        "crowdy": {k: rep["Crowdy"][k] for k in ("precision", "recall", "f1-score")},
    }


def subset_breakdown(oof, clips):
    sub = oof[oof.path.apply(lambda p: Path(p).name).isin(clips)]
    return {"n": int(len(sub)),
            "recall_panic": float((sub.predicted == "Panic").mean()),
            "pred_counts": sub.predicted.value_counts().to_dict()}


def main():
    feats = pd.read_csv(FEATURES_CSV)
    extra = pd.read_csv(FLOW_EXTRA)
    feats = feats.merge(extra, on=["path", "frame_index"], how="left")
    miss = feats[NEW].isna().any(axis=1).sum()
    print(f"merged; rows missing flow-extra: {miss} (filled 0)")
    feats[NEW] = feats[NEW].fillna(0.0)

    clips = (feats[["path", "video_id", "label"]].drop_duplicates("path")
             .reset_index(drop=True))
    folds = stratified_group_folds(clips, n_splits=5)

    before = cv_oof(feats, BEFORE_21, folds, clips)
    after = cv_oof(feats, AFTER_23, folds, clips)

    tri = pd.read_csv(TRIAGE)
    motion = set(tri[tri.category == "MOTION_RUNNING"]["clip"])

    report = {
        "new_features": NEW,
        "motion_running_subset": {
            "n_clips": len(motion),
            "before_21": subset_breakdown(before, motion),
            "after_23": subset_breakdown(after, motion)},
        "overall_before_21": overall(before),
        "overall_after_23": overall(after),
    }
    # feature importance of the new features
    from stampede.classifier import train
    m = train(feats, feature_cols=AFTER_23)
    imp = dict(sorted(zip(AFTER_23, m.rf.feature_importances_),
                      key=lambda kv: kv[1], reverse=True))
    report["new_feature_rank"] = {f: (list(imp).index(f) + 1, round(imp[f], 4))
                                  for f in NEW}
    (OUT / "flight_report.json").write_text(json.dumps(report, indent=2))

    mb, ma = report["motion_running_subset"]["before_21"], \
        report["motion_running_subset"]["after_23"]
    ob, oa = report["overall_before_21"], report["overall_after_23"]
    L = ["# Flight/scatter motion features — 4-class before/after (CV OOF)\n",
         f"New: {NEW}  ranks/importance: {report['new_feature_rank']}\n",
         f"## MOTION_RUNNING subset ({len(motion)} clips, all actual Panic)\n",
         "| | Before (21) | After (23) |", "|---|---|---|",
         f"| Recall (predicted Panic) | {mb['recall_panic']:.3f} | {ma['recall_panic']:.3f} |",
         f"| Predicted-label breakdown | {mb['pred_counts']} | {ma['pred_counts']} |",
         "", "## Overall (CV OOF, clip-level)\n",
         "| Metric | Before (21) | After (23) | Δ |", "|---|---|---|---|"]
    for lab, a, key in [("Panic precision", "panic", "precision"),
                        ("Panic recall", "panic", "recall"),
                        ("Panic F1", "panic", "f1-score"),
                        ("Crowdy F1", "crowdy", "f1-score")]:
        L.append(f"| {lab} | {ob[a][key]:.3f} | {oa[a][key]:.3f} | {oa[a][key]-ob[a][key]:+.3f} |")
    for lab, key in [("Accuracy", "accuracy"), ("Macro-F1", "macro_f1")]:
        L.append(f"| {lab} | {ob[key]:.3f} | {oa[key]:.3f} | {oa[key]-ob[key]:+.3f} |")
    (OUT / "flight_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
