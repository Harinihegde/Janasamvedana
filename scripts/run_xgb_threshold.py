#!/usr/bin/env python3
"""A4+A3 combo: sweep the XGBoost(scale_pos_weight=5) decision threshold.

XGBoost at threshold 0.5 gives high recall (0.705) but low precision (0.517).
Raising the threshold trades recall for precision. This finds the operating
point that satisfies both hard constraints (recall >= 0.65 AND precision >=
0.70) with the highest F1, using the same leakage-safe 5-fold OOF harness.
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
from sklearn.metrics import confusion_matrix

from run_improve import OUT, POS, NEG, load_binary, metrics_at, run_cv, _xgb_score
from stampede.config import FEATURE_COLUMNS, RANDOM_STATE

REC_MIN, PREC_MIN = 0.65, 0.70


def main():
    feats, clips, folds = load_binary()
    scores, y = run_cv(feats, clips, folds, _xgb_score, FEATURE_COLUMNS)

    grid = np.round(np.linspace(0.30, 0.90, 61), 3)
    sweep = [metrics_at(scores, y, t) for t in grid]
    sdf = pd.DataFrame(sweep)
    sdf.to_csv(OUT / "xgb_threshold_sweep.csv", index=False)

    # Operating points that meet BOTH constraints, best F1 among them.
    ok = sdf[(sdf.recall >= REC_MIN) & (sdf.precision >= PREC_MIN)]
    if len(ok):
        chosen = ok.loc[ok.f1.idxmax()].to_dict()
        met = True
    else:
        # Closest: among recall>=REC_MIN pick max precision; else max F1.
        rec_ok = sdf[sdf.recall >= REC_MIN]
        pool = rec_ok if len(rec_ok) else sdf
        chosen = pool.loc[pool.precision.idxmax() if len(rec_ok) else pool.f1.idxmax()].to_dict()
        met = False
    thr = float(chosen["threshold"])

    # Reference points for the report.
    def at(t):
        i = int(np.argmin(np.abs(grid - t)))
        return {k: round(v, 3) for k, v in sweep[i].items()}

    report = {
        "constraints": {"recall_min": REC_MIN, "precision_min": PREC_MIN},
        "both_constraints_met": met,
        "chosen_threshold": thr,
        "chosen_metrics": {k: round(v, 4) for k, v in chosen.items()},
        "reference_points": {"thr_0.50": at(0.50), "thr_0.60": at(0.60),
                             "thr_0.70": at(0.70), "thr_0.80": at(0.80)},
    }

    # Precision/recall/F1 vs threshold plot.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sdf.threshold, sdf.recall, label="recall", color="#d93025", lw=2)
    ax.plot(sdf.threshold, sdf.precision, label="precision", color="#f9ab00", lw=2)
    ax.plot(sdf.threshold, sdf.f1, label="F1", color="#1a73e8", lw=2)
    ax.axhline(REC_MIN, color="#d93025", ls=":", lw=1)
    ax.axhline(PREC_MIN, color="#f9ab00", ls=":", lw=1)
    ax.axvline(thr, color="k", ls="--", lw=1.5, label=f"chosen thr={thr:.3f}")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score"); ax.set_ylim(0, 1)
    ax.legend(); ax.set_title("XGBoost(scale_pos_weight=5) — metrics vs threshold")
    fig.tight_layout(); fig.savefig(OUT / "xgb_threshold_sweep.png", dpi=130); plt.close(fig)

    # CV OOF confusion at chosen threshold.
    cm = confusion_matrix(y, (scores >= thr).astype(int), labels=[0, 1])
    report["cv_oof_confusion"] = {"labels": [NEG, POS], "matrix": cm.tolist()}

    # Retrain on TRAIN split; save model; Stage 2 on TEST split at chosen threshold.
    import xgboost as xgb
    tr = feats[feats.split == "train"]; te = feats[feats.split == "test"].copy()
    dtrain = xgb.DMatrix(tr[FEATURE_COLUMNS].fillna(0).to_numpy(), label=tr.y.values)
    bst = xgb.train({"max_depth": 6, "learning_rate": 0.05, "objective": "binary:logistic",
                     "scale_pos_weight": 5, "seed": RANDOM_STATE, "eval_metric": "logloss"},
                    dtrain, num_boost_round=200)
    bst.save_model(str(OUT / "winner_xgb_tuned.json"))
    te["risk"] = bst.predict(xgb.DMatrix(te[FEATURE_COLUMNS].fillna(0).to_numpy()))

    from run_binary import eval_binary_stage2
    from stampede.escalation import clip_timelines
    tl = clip_timelines(te)
    report["stage2"] = {
        "spec_0.8": {k: (round(v, 3) if isinstance(v, float) else v)
                     for k, v in eval_binary_stage2(tl, 0.8, 20).items()},
        "chosen_threshold": {k: (round(v, 3) if isinstance(v, float) else v)
                             for k, v in eval_binary_stage2(tl, thr, 20).items()},
    }
    report["model_files"] = {"xgboost": "winner_xgb_tuned.json",
                             "feature_cols": FEATURE_COLUMNS}
    (OUT / "xgb_tuned_report.json").write_text(json.dumps(report, indent=2))

    print(json.dumps({"both_met": met, "chosen_threshold": thr,
                      "chosen": report["chosen_metrics"],
                      "refs": report["reference_points"]}, indent=2))


if __name__ == "__main__":
    main()
