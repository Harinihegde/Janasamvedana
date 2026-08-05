#!/usr/bin/env python3
"""Does adding real UMN fleeing-crowd footage as extra Panic training data help?

Context: our own dataset has only 23 Panic source videos, almost entirely
crush-type; the README names "more Panic footage, especially fleeing/running"
as the #1 documented lever, and we have zero real footage of that behavior.
The UMN "Unusual Crowd Activity" dataset (mha.cs.umn.edu, free academic-use
license) contains 11 real staged escape events, ground-truthed by an on-screen
"Abnormal Crowd Activity" banner burned into the original video - detected
programmatically (see conversation) to cut 11 Panic clips + 3 domain-matched
calm "Normal" clips (guarding against the model learning "this camera = Panic"
as a shortcut instead of real motion).

Method: leakage-safe 5-fold CV on our *own* 1,330-clip dataset (same folds as
the baseline), but every fold's training set gets all 14 UMN clips added
(train-only, never held out - we don't have enough UMN videos to carve out a
fair leakage-safe test split there too). Evaluation stays entirely on our own
held-out clips, so any change in accuracy is attributable to the extra
training signal, not to easier/different test data.

Outputs -> results/umn_augment_comparison.md
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stampede.classifier import train
from stampede.config import FEATURE_COLUMNS
from stampede.dataset import stratified_group_folds
from stampede.evaluate import _metrics, clip_vote

BASELINE_CSV = Path("outputs/frame_features.csv")
UMN_CSV = Path("outputs_umn_only/frame_features.csv")
OUT_MD = Path("results/umn_augment_comparison.md")


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill any FEATURE_COLUMNS missing from a CSV extracted before they
    existed (e.g. the CSRNet always-on columns added this session) with 0.0,
    matching stampede.features.frame_features's own convention."""
    df = df.copy()
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def run(extra_df: pd.DataFrame | None, label: str) -> dict:
    base = _ensure_feature_columns(pd.read_csv(BASELINE_CSV))
    clips = base[["path", "video_id", "label"]].drop_duplicates("path").reset_index(drop=True)
    folds = stratified_group_folds(clips, n_splits=5)

    oof_parts = []
    for tr_idx, te_idx in folds:
        tr_paths = set(clips.loc[tr_idx, "path"])
        te_paths = set(clips.loc[te_idx, "path"])
        tr = base[base.path.isin(tr_paths)]
        te = base[base.path.isin(te_paths)]
        if extra_df is not None:
            tr = pd.concat([tr, extra_df], ignore_index=True)
        model = train(tr, feature_cols=FEATURE_COLUMNS)
        pred = model.predict(te)
        oof_parts.append(clip_vote(te, pred))

    oof = pd.concat(oof_parts, ignore_index=True)
    metrics = _metrics(oof.actual.values, oof.predicted.values)
    print(f"--- {label} ---")
    print("accuracy:", round(metrics["accuracy"], 4), "macro_f1:", round(metrics["macro_f1"], 4))
    for c in ["No Panic", "Normal", "Crowdy", "Panic"]:
        r = metrics["per_class"][c]
        print(f"  {c:10s} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1-score']:.3f} n={int(r['support'])}")
    return metrics


def main() -> None:
    umn = _ensure_feature_columns(pd.read_csv(UMN_CSV))
    print(f"UMN extra training data: {len(umn)} frames from "
          f"{umn.video_id.nunique()} clips ({umn[umn.label=='Panic'].video_id.nunique()} Panic, "
          f"{umn[umn.label=='Normal'].video_id.nunique()} Normal)\n")

    baseline_metrics = run(None, "BASELINE (no UMN data)")
    print()
    umn_metrics = run(umn, "WITH UMN augmentation")

    lines = [
        "# Does adding real UMN fleeing-crowd data help? (train-only augmentation)",
        "",
        "5-fold leakage-safe CV on our own 1,330-clip dataset (evaluation set "
        "identical in both runs); the only difference is whether all 14 UMN "
        "clips (11 real escape events + 3 domain-matched calm clips) are added "
        "to every fold's *training* set.",
        "",
        "| Metric | Baseline | +UMN | Delta |",
        "|---|---|---|---|",
    ]

    def row(name, key_path):
        b = baseline_metrics
        u = umn_metrics
        for k in key_path:
            b = b[k]
            u = u[k]
        lines.append(f"| {name} | {b:.3f} | {u:.3f} | {u-b:+.3f} |")

    row("Accuracy", ["accuracy"])
    row("Macro-F1", ["macro_f1"])
    for c in ["No Panic", "Normal", "Crowdy", "Panic"]:
        row(f"{c} recall", ["per_class", c, "recall"])
        row(f"{c} precision", ["per_class", c, "precision"])
        row(f"{c} F1", ["per_class", c, "f1-score"])

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_MD}")


if __name__ == "__main__":
    main()
