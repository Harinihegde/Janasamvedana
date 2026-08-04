#!/usr/bin/env python3
"""Stage 1: train the crowd-risk classifier and report leakage-safe metrics.

Reads ``frame_features.csv`` (from ``extract_features.py``), trains a
RandomForest on the grouped train split, evaluates on the held-out test videos
(frame and clip level), runs StratifiedGroupKFold cross-validation, and saves
the model, normalisation params, metrics report, confusion matrix and
per-frame test predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stampede.classifier import train
from stampede.config import FEATURE_COLUMNS
from stampede.evaluate import cross_validate, evaluate_holdout
from stampede.visualize import plot_confusion, plot_cv_summary

BASELINE_LEAKY_ACC = 0.8083  # previous clip-level accuracy WITH data leakage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("outputs"))
    ap.add_argument("--cv-splits", type=int, default=5)
    a = ap.parse_args()

    feats = pd.read_csv(a.output / "frame_features.csv")
    train_df = feats[feats.split == "train"].reset_index(drop=True)
    test_df = feats[feats.split == "test"].reset_index(drop=True)

    # --- Held-out evaluation (leakage-safe grouped split) ---
    holdout = evaluate_holdout(train_df, test_df)

    # --- Cross-validation (leakage-safe folds) ---
    cv = cross_validate(feats, n_splits=a.cv_splits)

    # --- Fit final model on the full train split & persist ---
    model = train(train_df)
    model.save(a.output / "stage1_model.pkl")
    (a.output / "normalization_params.json").write_text(
        json.dumps(model.normalizer.to_dict(), indent=2)
    )

    # --- Per-frame test predictions with risk score ---
    y_pred = model.predict(test_df)
    risk = model.risk_score(test_df)
    preds = pd.DataFrame(
        dict(
            video_id=test_df.video_id,
            path=test_df.path,
            frame_number=test_df.frame_index,
            actual_label=test_df.label,
            predicted_label=y_pred,
            risk_score=risk,
        )
    )
    preds.to_csv(a.output / "test_predictions.csv", index=False)

    # --- Report ---
    clip_acc = holdout["clip_level"]["accuracy"]
    report = {
        "protocol": "leakage-safe: train/test split by video_id; CV via "
        "StratifiedGroupKFold grouped on video_id",
        "baseline_leaky_accuracy": BASELINE_LEAKY_ACC,
        "holdout": holdout,
        "cross_validation": cv,
        "comparison_to_baseline": {
            "baseline_clip_accuracy_with_leakage": BASELINE_LEAKY_ACC,
            "leakage_safe_holdout_clip_accuracy": clip_acc,
            "leakage_safe_cv_clip_accuracy_mean": cv["clip_accuracy_mean"],
            "delta_vs_baseline": clip_acc - BASELINE_LEAKY_ACC,
            "note": "A drop is expected and correct - it removes the "
            "same-video leakage that inflated the baseline.",
        },
        "feature_columns": FEATURE_COLUMNS,
        "feature_importances": dict(
            sorted(
                zip(FEATURE_COLUMNS, model.rf.feature_importances_.tolist()),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ),
    }
    (a.output / "stage1_report.json").write_text(json.dumps(report, indent=2))

    # --- Visuals ---
    plot_confusion(
        holdout["clip_level"]["confusion_matrix"],
        holdout["clip_level"]["confusion_labels"],
        a.output / "confusion_matrix_clip.png",
        f"Stage 1 clip-level confusion (acc={clip_acc:.2%})",
    )
    plot_confusion(
        holdout["frame_level"]["confusion_matrix"],
        holdout["frame_level"]["confusion_labels"],
        a.output / "confusion_matrix_frame.png",
        f"Stage 1 frame-level confusion (acc={holdout['frame_level']['accuracy']:.2%})",
    )
    plot_cv_summary(cv["fold_accuracies"], a.output / "cv_accuracy.png")

    # --- Console summary ---
    print(json.dumps(report["comparison_to_baseline"], indent=2))
    print("\nHold-out clip-level per-class F1:")
    for cls, m in holdout["clip_level"]["per_class"].items():
        if isinstance(m, dict) and "f1-score" in m:
            print(f"  {cls:12s} P={m['precision']:.2f} R={m['recall']:.2f} "
                  f"F1={m['f1-score']:.2f} (n={int(m['support'])})")


if __name__ == "__main__":
    main()
