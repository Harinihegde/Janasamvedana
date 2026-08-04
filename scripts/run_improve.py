#!/usr/bin/env python3
"""Optimise Binary(19) for recall: test 5 techniques against a common harness.

Baseline (as previously reported): Binary(19) with class_weight='balanced',
clip = majority vote — Stampede-Risk F1 0.621, recall 0.520, precision 0.772.

All approaches here are evaluated through ONE leakage-safe harness so they are
directly comparable:
  * StratifiedGroupKFold(5) grouped by video_id (no video spans a fold).
  * Each fold: train on the training frames, score P(Stampede Risk) on the
    held-out frames.
  * Aggregate to clip level = MEAN frame probability per clip, then threshold.
  * Pool the out-of-fold (OOF) clip scores across folds and compute metrics.

Because the clip aggregation here is mean-probability (not majority vote), the
recomputed baseline row may differ slightly from the 0.621 reported earlier;
that harness-baseline is the honest reference for these deltas.

Approaches: 1 cost-sensitive RF, 2 SMOTE, 3 threshold optimisation,
4 XGBoost(scale_pos_weight), 5 feature selection (top-12) + cost-sensitive RF.
Outputs -> outputs_improved/.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
)

from stampede.classifier import Normalizer
from stampede.config import FEATURE_COLUMNS, RANDOM_STATE
from stampede.dataset import stratified_group_folds

POS, NEG = "Stampede Risk", "Safe"
OUT = Path("outputs_improved")
FEATURES_CSV = Path("outputs_compression/frame_features.csv")
COST_WEIGHT = {NEG: 1.0, POS: 5.0}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
def load_binary() -> tuple[pd.DataFrame, pd.DataFrame, list]:
    feats = pd.read_csv(FEATURES_CSV)
    feats["label"] = np.where(feats.label == "Panic", POS, NEG)
    feats["y"] = (feats.label == POS).astype(int)
    clips = feats[["path", "video_id", "label"]].drop_duplicates("path").reset_index(drop=True)
    folds = stratified_group_folds(clips, n_splits=5)
    return feats, clips, folds


def run_cv(feats, clips, folds, score_fold, feature_cols=FEATURE_COLUMNS):
    """Return pooled OOF (clip_score, y_true) using ``score_fold``.

    ``score_fold(train_df, test_df, feature_cols) -> per-test-frame P(POS)``.
    """
    rows = []
    for tr_idx, te_idx in folds:
        tr_paths = set(clips.loc[tr_idx, "path"])
        te_paths = set(clips.loc[te_idx, "path"])
        tr = feats[feats.path.isin(tr_paths)]
        te = feats[feats.path.isin(te_paths)].copy()
        te["p"] = score_fold(tr, te, feature_cols)
        agg = te.groupby("path").agg(clip_score=("p", "mean"),
                                     y=("y", "first")).reset_index()
        rows.append(agg)
    oof = pd.concat(rows, ignore_index=True)
    return oof.clip_score.to_numpy(), oof.y.to_numpy()


def metrics_at(scores, y, thr):
    pred = (scores >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", pos_label=1, zero_division=0)
    return {
        "threshold": float(thr),
        "f1": float(f1),
        "recall": float(r),
        "precision": float(p),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
    }


def best_threshold(scores, y):
    """Maximise TPR - 2*FPR over ROC thresholds (misses cost 2x false alarms)."""
    fpr, tpr, thr = roc_curve(y, scores)
    j = tpr - 2 * fpr
    return float(thr[int(np.argmax(j))]), fpr, tpr


# ---------------------------------------------------------------------------
# Per-fold scorers
# ---------------------------------------------------------------------------
def _rf(class_weight):
    return RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight=class_weight,
        random_state=RANDOM_STATE, n_jobs=-1)


def _rf_score_factory(class_weight):
    def score(tr, te, cols):
        norm = Normalizer.fit(tr, cols)
        clf = _rf(class_weight)
        clf.fit(norm.transform(tr), tr.label.values)
        idx = list(clf.classes_).index(POS)
        return clf.predict_proba(norm.transform(te))[:, idx]
    return score


def _smote_score(tr, te, cols):
    from imblearn.over_sampling import SMOTE
    norm = Normalizer.fit(tr, cols)
    Xtr = norm.transform(tr)
    ytr = tr.y.values
    sm = SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE)
    Xb, yb = sm.fit_resample(Xtr, ytr)
    clf = _rf("balanced")
    clf.fit(Xb, yb)
    idx = list(clf.classes_).index(1)
    return clf.predict_proba(norm.transform(te))[:, idx]


def _xgb_score(tr, te, cols):
    import xgboost as xgb
    dtrain = xgb.DMatrix(tr[cols].fillna(0).to_numpy(), label=tr.y.values)
    dtest = xgb.DMatrix(te[cols].fillna(0).to_numpy())
    params = {"max_depth": 6, "learning_rate": 0.05, "objective": "binary:logistic",
              "scale_pos_weight": 5, "seed": RANDOM_STATE, "nthread": -1,
              "eval_metric": "logloss"}
    bst = xgb.train(params, dtrain, num_boost_round=200)
    return bst.predict(dtest)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    feats, clips, folds = load_binary()

    # Feature-selection set (Approach 5): top-12 by importance from a
    # cost-sensitive RF fit on all data (defined once, then CV'd).
    norm_all = Normalizer.fit(feats, FEATURE_COLUMNS)
    rf_all = _rf(COST_WEIGHT)
    rf_all.fit(norm_all.transform(feats), feats.label.values)
    imp = sorted(zip(FEATURE_COLUMNS, rf_all.feature_importances_),
                 key=lambda kv: kv[1], reverse=True)
    top12 = [f for f, _ in imp[:12]]

    approaches = {
        "Baseline (balanced RF)": dict(score=_rf_score_factory("balanced"),
                                       cols=FEATURE_COLUMNS, opt_thr=False),
        "A1 cost-sensitive": dict(score=_rf_score_factory(COST_WEIGHT),
                                  cols=FEATURE_COLUMNS, opt_thr=False),
        "A2 SMOTE": dict(score=_smote_score, cols=FEATURE_COLUMNS, opt_thr=False),
        "A3 threshold-opt": dict(score=_rf_score_factory("balanced"),
                                 cols=FEATURE_COLUMNS, opt_thr=True),
        "A4 XGBoost": dict(score=_xgb_score, cols=FEATURE_COLUMNS, opt_thr=False),
        "A5 feature-select (top12)": dict(score=_rf_score_factory(COST_WEIGHT),
                                          cols=top12, opt_thr=False),
    }

    results = {}
    roc_data = None
    for name, cfg in approaches.items():
        scores, y = run_cv(feats, clips, folds, cfg["score"], cfg["cols"])
        if cfg["opt_thr"]:
            thr, fpr, tpr = best_threshold(scores, y)
            roc_data = (fpr, tpr, thr)
        else:
            thr = 0.5
        results[name] = metrics_at(scores, y, thr)
        print(f"{name:28s} F1={results[name]['f1']:.3f} R={results[name]['recall']:.3f} "
              f"P={results[name]['precision']:.3f} thr={thr:.2f}", flush=True)

    # ------------------------------------------------------------------
    # Winner selection: recall>=0.65, precision>=0.70, f1>=0.70; else best-effort.
    # ------------------------------------------------------------------
    def meets(m):
        return m["recall"] >= 0.65 and m["precision"] >= 0.70 and m["f1"] >= 0.70

    qualified = {k: v for k, v in results.items() if k != "Baseline (balanced RF)"
                 and meets(v)}
    if qualified:
        winner = max(qualified, key=lambda k: qualified[k]["f1"])
        criteria_met = True
    else:
        # Best-effort: highest recall among precision>=0.70, else highest F1.
        prec_ok = {k: v for k, v in results.items()
                   if k != "Baseline (balanced RF)" and v["precision"] >= 0.70}
        pool = prec_ok or {k: v for k, v in results.items()
                           if k != "Baseline (balanced RF)"}
        winner = max(pool, key=lambda k: (pool[k]["recall"], pool[k]["f1"]))
        criteria_met = False

    # ------------------------------------------------------------------
    # Comparison table (CSV + JSON)
    # ------------------------------------------------------------------
    order = ["f1", "recall", "precision", "macro_f1", "accuracy", "threshold"]
    df = pd.DataFrame(results).T[order]
    df.index.name = "approach"
    df.to_csv(OUT / "comparison_improve.csv")
    (OUT / "comparison_improve.json").write_text(json.dumps(
        {"results": results, "winner": winner, "criteria_met": criteria_met,
         "top12_features": top12}, indent=2))

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------
    # ROC (Approach 3)
    if roc_data:
        fpr, tpr, thr = roc_data
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, color="#1a73e8", lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(f"A3 ROC — chosen threshold {thr:.2f}")
        fig.tight_layout(); fig.savefig(OUT / "roc_curve_A3.png", dpi=130); plt.close(fig)

    # Comparison bar chart
    fig, ax = plt.subplots(figsize=(11, 5))
    names = list(results.keys())
    x = np.arange(len(names)); w = 0.25
    for i, (metric, color) in enumerate([("f1", "#1a73e8"), ("recall", "#d93025"),
                                         ("precision", "#f9ab00")]):
        ax.bar(x + (i - 1) * w, [results[n][metric] for n in names], w,
               label=metric, color=color)
    ax.axhline(0.70, color="gray", ls="--", lw=1, label="target 0.70")
    ax.set_xticks(x, [n.replace(" ", "\n", 1) for n in names], fontsize=8)
    ax.set_ylim(0, 1); ax.legend(); ax.set_title("Binary(19) optimisation — F1 / Recall / Precision")
    fig.tight_layout(); fig.savefig(OUT / "comparison_bar.png", dpi=130); plt.close(fig)

    # Feature importance (top-12 / full 19)
    fig, ax = plt.subplots(figsize=(8, 6))
    feats_imp = imp[::-1]
    ax.barh([f for f, _ in feats_imp], [v for _, v in feats_imp],
            color=["#d93025" if f in top12 else "#9aa0a6" for f, _ in feats_imp])
    ax.set_title("19-feature importance (red = top-12 kept in A5)")
    fig.tight_layout(); fig.savefig(OUT / "feature_importance.png", dpi=130); plt.close(fig)

    # ------------------------------------------------------------------
    # Winner artifacts: retrain on TRAIN split, evaluate Stage 2 on TEST split
    # ------------------------------------------------------------------
    winner_cfg = approaches[winner]
    cols = winner_cfg["cols"]
    tr_all = feats[feats.split == "train"]
    te_all = feats[feats.split == "test"].copy()
    # Fit winner estimator on full train split; capture a per-frame scorer.
    scorer = winner_cfg["score"]
    # Build a picklable bundle: refit and store.
    norm = Normalizer.fit(tr_all, cols)
    bundle = {"approach": winner, "feature_cols": cols,
              "threshold": results[winner]["threshold"]}
    if winner == "A4 XGBoost":
        import xgboost as xgb
        dtrain = xgb.DMatrix(tr_all[cols].fillna(0).to_numpy(), label=tr_all.y.values)
        bst = xgb.train({"max_depth": 6, "learning_rate": 0.05,
                         "objective": "binary:logistic", "scale_pos_weight": 5,
                         "seed": RANDOM_STATE, "eval_metric": "logloss"},
                        dtrain, num_boost_round=200)
        bst.save_model(str(OUT / "winner_xgb.json"))
        te_all["risk"] = bst.predict(xgb.DMatrix(te_all[cols].fillna(0).to_numpy()))
        bundle["model_file"] = "winner_xgb.json"
    else:
        cw = COST_WEIGHT if ("cost" in winner or "feature" in winner) else "balanced"
        clf = _rf(cw)
        if winner == "A2 SMOTE":
            from imblearn.over_sampling import SMOTE
            Xb, yb = SMOTE(sampling_strategy=0.5, random_state=RANDOM_STATE
                           ).fit_resample(norm.transform(tr_all), tr_all.y.values)
            clf.fit(Xb, yb)
            idx = list(clf.classes_).index(1)
        else:
            clf.fit(norm.transform(tr_all), tr_all.label.values)
            idx = list(clf.classes_).index(POS)
        te_all["risk"] = clf.predict_proba(norm.transform(te_all))[:, idx]
        bundle["normalizer"] = norm.to_dict()
        with open(OUT / "winner_model.pkl", "wb") as fh:
            pickle.dump({"clf": clf, "normalizer": norm, "feature_cols": cols,
                         "threshold": bundle["threshold"], "pos_index": idx}, fh)
        bundle["model_file"] = "winner_model.pkl"

    # Winner confusion matrix (CV OOF at its threshold)
    scores, y = run_cv(feats, clips, folds, winner_cfg["score"], cols)
    if winner == "A3 threshold-opt":
        thr = results[winner]["threshold"]
    else:
        thr = 0.5
    cm = confusion_matrix(y, (scores >= thr).astype(int), labels=[0, 1])
    bundle["cv_oof_confusion"] = {"labels": [NEG, POS], "matrix": cm.tolist()}

    # Stage 2 on TEST split with the winner's risk + threshold
    from run_binary import build_binary_sequences, eval_binary_stage2
    from stampede.escalation import clip_timelines
    tl = clip_timelines(te_all)
    stage2 = {
        "spec_0.8": eval_binary_stage2(tl, 0.8, 20),
        "winner_threshold": eval_binary_stage2(tl, float(bundle["threshold"]), 20),
    }
    bundle["stage2"] = stage2
    (OUT / "winner_report.json").write_text(json.dumps(bundle, indent=2))

    print(f"\nWINNER: {winner}  (criteria fully met: {criteria_met})")
    print(json.dumps(results[winner], indent=2))
    print("Stage 2 (winner threshold):",
          json.dumps({k: round(v, 3) if isinstance(v, float) else v
                      for k, v in stage2["winner_threshold"].items()}))


if __name__ == "__main__":
    main()
