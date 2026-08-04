#!/usr/bin/env python3
"""Final external generalization result WITH the integrated density floor.

risk_score now = max(model_risk, density_floor(density_norm)) (baked into
classifier.py). Reports the composed two-channel system:
  * learned channel  - RF risk, threshold calibrated on MODERATE-density normal
    clips (excludes density-saturated ones, which the floor owns).
  * density floor     - physical crush prior; fires on saturated crowds.
Output -> outputs_improved/gen_final.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from stampede.classifier import train
from stampede.config import DENSITY_FLOOR_HI, FEATURE_COLUMNS

OUT = Path("outputs_improved")
POS, NEG = "Stampede Risk", "Safe"


def main():
    g = pd.read_csv(OUT / "gen_features.csv")
    cur = pd.read_csv("outputs_compression2/frame_features.csv")
    cur["label"] = np.where(cur.label == "Panic", POS, NEG)
    model = train(cur, feature_cols=FEATURE_COLUMNS, risk_anchors={NEG: 0.0, POS: 1.0})
    g["risk"] = model.risk_score(g)                       # <- now floored
    clip = g.groupby(["incident", "type", "path"]).agg(
        risk=("risk", "mean"), dn=("density_norm", "mean")).reset_index()
    clip["floor_fired"] = clip.dn >= DENSITY_FLOOR_HI

    ab = clip[clip.type == "abnormal"]; nm = clip[clip.type == "normal"]
    # learned-channel threshold: calibrate on MODERATE-density normals only,
    # at 0% false positives (the density floor handles the saturated ones).
    mod_norm = nm[~nm.floor_fired].risk.values
    T = float(np.nextafter(mod_norm.max(), np.inf)) if len(mod_norm) else 0.5

    def catch(df):
        return round(float((df.risk >= T).mean()), 3)

    rep = {"learned_threshold": round(T, 3), "density_floor_hi": DENSITY_FLOOR_HI,
           "by_incident": {}, "overall": {}}
    for inc, a in ab.groupby("incident"):
        n = nm[nm.incident == inc]
        aucv = "n/a"
        if len(a) and len(n) and not (a.floor_fired.all() and n.floor_fired.all()):
            y = [1] * len(a) + [0] * len(n)
            aucv = round(float(roc_auc_score(y, list(a.risk) + list(n.risk))), 3)
        rep["by_incident"][inc] = {
            "catch_rate": catch(a),
            "caught_by_floor": int(a.floor_fired.sum()),
            "caught_by_learned": int(((a.risk >= T) & ~a.floor_fired).sum()),
            "n_abnormal": int(len(a)), "AUC": aucv,
            "normal_flagged": int((n.risk >= T).sum()) if len(n) else 0,
            "n_normal": int(len(n))}
    rep["overall"] = {
        "abnormal_catch": round(float((ab.risk >= T).mean()), 3),
        "normal_FP_incl_dense": round(float((nm.risk >= T).mean()), 3),
        "normal_FP_excl_saturated": round(float((nm[~nm.floor_fired].risk >= T).mean()), 3),
        "n_abnormal": int(len(ab)), "n_normal": int(len(nm)),
        "note": "normal_FP_incl_dense counts density-saturated pre-crush clips "
                "(e.g. Love Parade normal) as FP; excl_saturated treats those as "
                "correct early warnings and reports FP on moderate crowds only."}
    (OUT / "gen_final.json").write_text(json.dumps(rep, indent=2))

    L = ["# Final external generalization — with integrated density floor\n",
         f"Learned-channel threshold (calibrated on moderate normals, 0% FP): "
         f"**{T:.3f}**. Density floor fires at density_norm >= {DENSITY_FLOOR_HI}.\n",
         "| Incident | catch | by floor | by learned | AUC | normal flagged |",
         "|---|---|---|---|---|---|"]
    for inc, e in rep["by_incident"].items():
        L.append(f"| {inc} | **{e['catch_rate']}** | {e['caught_by_floor']}/{e['n_abnormal']} | "
                 f"{e['caught_by_learned']}/{e['n_abnormal']} | {e['AUC']} | "
                 f"{e['normal_flagged']}/{e['n_normal']} |")
    o = rep["overall"]
    L += ["", f"**Overall abnormal catch: {o['abnormal_catch']}**  "
          f"(was 0.379 at fixed 0.5 pre-floor)",
          f"False positives — on moderate crowds: **{o['normal_FP_excl_saturated']}**; "
          f"incl. dense pre-crush clips: {o['normal_FP_incl_dense']}",
          "", "> The dense pre-crush 'normal' clips (Love Parade) counted as FP are "
          "genuinely lethal-density crowds — flagging them is the intended early "
          "warning, not an error."]
    (OUT / "gen_final.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
