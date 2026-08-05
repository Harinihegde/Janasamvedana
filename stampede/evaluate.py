"""Stage 1 evaluation: per-class metrics, confusion matrix, clip aggregation, CV.

Two evaluation *units* are reported:

* **frame level** - every sampled frame is scored independently (this is the
  unit Stage 2 operates on).
* **clip level** - frame predictions are aggregated by majority vote per clip;
  this is the more honest headline number because frames within a clip are
  highly correlated.

All splits are leakage-safe (grouped by ``video_id``) - see :mod:`dataset`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .classifier import Stage1Model, train
from .config import CLASSES
from .dataset import stratified_group_folds
from .sublabels import panic_flight_sample_weight


def _metrics(y_true, y_pred, labels: list[str] | None = None) -> dict:
    labels = list(CLASSES) if labels is None else list(labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
        "confusion_labels": labels,
    }


def clip_vote(frame_df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """Aggregate frame predictions to one row per clip via majority vote."""
    tmp = frame_df[["path", "video_id", "label"]].copy()
    tmp["pred"] = y_pred
    out = []
    for path, g in tmp.groupby("path", sort=False):
        pred = g.pred.value_counts().idxmax()
        out.append(
            dict(
                path=path,
                video_id=g.video_id.iloc[0],
                actual=g.label.iloc[0],
                predicted=pred,
            )
        )
    return pd.DataFrame(out)


def evaluate_holdout(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels: list[str] | None = None,
    risk_anchors: dict | None = None,
    panic_flight_upweight: bool = False,
) -> dict:
    """Train on ``train_df`` and evaluate at frame and clip level on ``test_df``.

    ``panic_flight_upweight`` upweights confirmed Panic-flight frames within
    ``train_df`` so training isn't crush-dominated (see
    :mod:`stampede.sublabels`).
    """
    sample_weight = panic_flight_sample_weight(train_df) if panic_flight_upweight else None
    model = train(train_df, risk_anchors=risk_anchors, sample_weight=sample_weight)
    y_pred = model.predict(test_df)
    frame_metrics = _metrics(test_df.label.values, y_pred, labels)

    clip_df = clip_vote(test_df, y_pred)
    clip_metrics = _metrics(clip_df.actual.values, clip_df.predicted.values, labels)
    return {
        "frame_level": frame_metrics,
        "clip_level": clip_metrics,
        "n_train_frames": int(len(train_df)),
        "n_test_frames": int(len(test_df)),
        "n_test_clips": int(clip_df.shape[0]),
    }


def cross_validate(
    frame_df: pd.DataFrame,
    n_splits: int = 5,
    labels: list[str] | None = None,
    risk_anchors: dict | None = None,
    panic_flight_upweight: bool = False,
) -> dict:
    """Leakage-safe StratifiedGroupKFold CV, reported at clip level.

    Because folds are built at the clip granularity but grouped by ``video_id``,
    no source video ever spans a fold boundary. In addition to per-fold
    accuracy/F1, the out-of-fold (OOF) clip predictions are pooled across all
    folds to give a single stable confusion matrix and per-class F1 (each clip
    is a test clip in exactly one fold).
    """
    # Build a clip-level manifest for fold assignment (one row per clip).
    clips = (
        frame_df[["path", "video_id", "label"]]
        .drop_duplicates("path")
        .reset_index(drop=True)
    )
    folds = stratified_group_folds(clips, n_splits=n_splits)
    accs, macros, weighted = [], [], []
    oof_frames = []
    for tr_idx, te_idx in folds:
        tr_paths = set(clips.loc[tr_idx, "path"])
        te_paths = set(clips.loc[te_idx, "path"])
        tr = frame_df[frame_df.path.isin(tr_paths)]
        te = frame_df[frame_df.path.isin(te_paths)]
        sample_weight = panic_flight_sample_weight(tr) if panic_flight_upweight else None
        model = train(tr, risk_anchors=risk_anchors, sample_weight=sample_weight)
        pred = model.predict(te)
        clip_df = clip_vote(te, pred)
        oof_frames.append(clip_df)
        accs.append(accuracy_score(clip_df.actual, clip_df.predicted))
        macros.append(
            f1_score(clip_df.actual, clip_df.predicted, average="macro", zero_division=0)
        )
        weighted.append(
            f1_score(
                clip_df.actual, clip_df.predicted, average="weighted", zero_division=0
            )
        )
    oof = pd.concat(oof_frames, ignore_index=True)
    oof_metrics = _metrics(oof.actual.values, oof.predicted.values, labels)
    return {
        "n_splits": len(folds),
        "clip_accuracy_mean": float(np.mean(accs)),
        "clip_accuracy_std": float(np.std(accs)),
        "clip_macro_f1_mean": float(np.mean(macros)),
        "clip_macro_f1_std": float(np.std(macros)),
        "clip_weighted_f1_mean": float(np.mean(weighted)),
        "fold_accuracies": [float(a) for a in accs],
        "oof": oof_metrics,
    }
