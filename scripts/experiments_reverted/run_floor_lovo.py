#!/usr/bin/env python3
"""Properly evaluate the density floor under the leakage-safe LOVO protocol.

Binary Safe-vs-Stampede-Risk, 21 features, leave-one-video-out. One LOVO pass
gives each clip a base (model) OOF risk; the floor is then applied post-hoc, so
we compare identical predictions WITH vs WITHOUT the density floor on held-out
videos. Threshold-free (ROC-AUC) plus P/R/F1 at a fixed 0.5 operating point.
Output -> outputs_improved/floor_lovo.{json,md}
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from stampede.classifier import Normalizer
from stampede.config import (DENSITY_FLOOR_HI, DENSITY_FLOOR_LO, FEATURE_COLUMNS,
                             RANDOM_STATE)

POS, NEG = "Stampede Risk", "Safe"
OUT = Path("outputs_improved")


def main():
    d = pd.read_csv("outputs_compression2/frame_features.csv")
    d["label"] = np.where(d.label == "Panic", POS, NEG)
    d["y"] = (d.label == POS).astype(int)
    vids = d.video_id.unique().tolist()

    # LOVO base (model) OOF risk per frame
    d["base_risk"] = np.nan
    t0 = time.time()
    for n, v in enumerate(vids, 1):
        tr = d[d.video_id != v]; te_idx = d.index[d.video_id == v]
        te = d.loc[te_idx]
        norm = Normalizer.fit(tr, FEATURE_COLUMNS)
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                     class_weight="balanced", random_state=RANDOM_STATE,
                                     n_jobs=-1)
        clf.fit(norm.transform(tr), tr.label.values)
        pidx = list(clf.classes_).index(POS)
        d.loc[te_idx, "base_risk"] = clf.predict_proba(norm.transform(te))[:, pidx]
        if n % 15 == 0 or n == len(vids):
            print(f"  LOVO {n}/{len(vids)} ({time.time()-t0:.0f}s)", flush=True)

    # floor applied post-hoc
    floor = np.clip((d.density_norm - DENSITY_FLOOR_LO) /
                    (DENSITY_FLOOR_HI - DENSITY_FLOOR_LO), 0, 1)
    d["floored_risk"] = np.maximum(d.base_risk, floor)

    # clip-level
    clip = d.groupby("path").agg(y=("y", "first"), label=("label", "first"),
                                 base=("base_risk", "mean"),
                                 floored=("floored_risk", "mean")).reset_index()

    def metrics(score):
        auc = float(roc_auc_score(clip.y, score))
        pred = (score >= 0.5).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(clip.y, pred, average="binary",
                                                      pos_label=1, zero_division=0)
        # FP among Crowdy specifically (dense-safe)
        return {"AUC": round(auc, 3), "risk_precision": round(float(p), 3),
                "risk_recall": round(float(r), 3), "risk_f1": round(float(f1), 3)}

    base_m = metrics(clip.base)
    floor_m = metrics(clip.floored)
    crowdy = clip[clip.label == NEG]  # note: Crowdy is inside Safe
    # how many extra Safe clips does the floor flag (base<0.5 -> floored>=0.5)?
    flipped_safe = int(((clip.label == NEG) & (clip.base < 0.5) & (clip.floored >= 0.5)).sum())
    flipped_risk = int(((clip.label == POS) & (clip.base < 0.5) & (clip.floored >= 0.5)).sum())

    rep = {"protocol": "binary LOVO (leave-one-video-out), clip-level, 21 features",
           "floor_LO_HI": [DENSITY_FLOOR_LO, DENSITY_FLOOR_HI],
           "WITHOUT_floor": base_m, "WITH_floor": floor_m,
           "floor_effect": {
               "safe_clips_newly_flagged_risk": flipped_safe,
               "risk_clips_newly_flagged_risk": flipped_risk,
               "note": "floor flips clips base<0.5 -> >=0.5 purely from density; "
                       "if it flags more SAFE than RISK it hurts."}}
    (OUT / "floor_lovo.json").write_text(json.dumps(rep, indent=2))

    L = ["# Density floor — leakage-safe LOVO evaluation (in-domain)\n",
         "Binary Safe-vs-Risk, leave-one-video-out, clip-level.\n",
         "| Metric | WITHOUT floor | WITH floor | Δ |", "|---|---|---|---|"]
    for k in ["AUC", "risk_precision", "risk_recall", "risk_f1"]:
        L.append(f"| {k} | {base_m[k]} | {floor_m[k]} | {round(floor_m[k]-base_m[k],3):+} |")
    L += ["", f"Clips the floor newly flags as RISK (base<0.5 → ≥0.5): "
          f"**{flipped_safe} SAFE** vs **{flipped_risk} RISK** "
          f"→ the floor flags {'more SAFE than RISK (hurts)' if flipped_safe>flipped_risk else 'more RISK (helps)'}.",
          "", "Threshold 0.80/0.95 was originally chosen with knowledge of the "
          "external test; this evaluation uses only in-domain held-out videos."]
    (OUT / "floor_lovo.md").write_text("\n".join(L))
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
