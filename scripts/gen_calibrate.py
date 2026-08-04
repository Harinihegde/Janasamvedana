#!/usr/bin/env python3
"""Per-domain threshold calibration for the external generalization test.

Thresholds are set from the NORMAL clips only (deployment-realistic: learn a
site's normal risk baseline, alarm above it) -> abnormal-clip catch rate is not
peeked at. Reports catch rate at fixed false-alarm budgets, globally and (where
enough normal clips exist) per incident, vs the naive fixed-0.5 baseline.
Output -> outputs_improved/gen_calibration.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stampede.classifier import train
from stampede.config import FEATURE_COLUMNS

OUT = Path("outputs_improved")
POS, NEG = "Stampede Risk", "Safe"


def thr_at_fp(normal_risk, fp_target):
    """Smallest threshold whose normal false-positive rate <= fp_target."""
    if len(normal_risk) == 0:
        return None
    q = np.quantile(normal_risk, 1 - fp_target) if fp_target > 0 else \
        np.nextafter(np.max(normal_risk), np.inf)
    return float(q)


def main():
    g = pd.read_csv(OUT / "gen_features.csv")
    cur = pd.read_csv("outputs_compression2/frame_features.csv")
    cur["label"] = np.where(cur.label == "Panic", POS, NEG)
    model = train(cur, feature_cols=FEATURE_COLUMNS, risk_anchors={NEG: 0.0, POS: 1.0})
    g["risk"] = model.risk_score(g)
    clip = g.groupby(["incident", "kind", "type", "path"]).risk.mean().reset_index()

    ab = clip[clip.type == "abnormal"]
    nm = clip[clip.type == "normal"]
    normal_all = nm.risk.values

    # ---- global normal-calibrated thresholds at FP budgets ----
    report = {"baseline_fixed_0.5": {
        "threshold": 0.5,
        "overall_catch": round(float((ab.risk >= 0.5).mean()), 3),
        "overall_FP": round(float((nm.risk >= 0.5).mean()), 3)}}
    global_tbl = []
    for fp in [0.0, 0.05, 0.10, 0.20]:
        T = thr_at_fp(normal_all, fp)
        row = {"fp_budget": fp, "threshold": round(T, 3),
               "overall_catch": round(float((ab.risk >= T).mean()), 3),
               "overall_FP_actual": round(float((nm.risk >= T).mean()), 3),
               "per_incident_catch": {}}
        for inc, a in ab.groupby("incident"):
            row["per_incident_catch"][inc] = round(float((a.risk >= T).mean()), 3)
        global_tbl.append(row)
    report["global_normal_calibrated"] = global_tbl

    # ---- per-incident calibration (own normal baseline; needs normal clips) ----
    per_inc = {}
    for inc in clip.incident.unique():
        a = ab[ab.incident == inc].risk.values
        n = nm[nm.incident == inc].risk.values
        entry = {"n_abnormal": int(len(a)), "n_normal": int(len(n))}
        if len(n) >= 3:  # only meaningful with a few normal clips
            T = thr_at_fp(n, 0.0)  # threshold just above this site's max normal
            entry.update(threshold_above_site_normal=round(T, 3),
                         catch_rate=round(float((a >= T).mean()), 3),
                         note="threshold = just above this incident's normal max")
        else:
            entry["note"] = "too few normal clips to self-calibrate"
        per_inc[inc] = entry
    report["per_incident_calibrated"] = per_inc

    (OUT / "gen_calibration.json").write_text(json.dumps(report, indent=2))

    L = ["# Per-domain threshold calibration (external generalization test)\n",
         "Thresholds set from NORMAL clips only (deployment-realistic). "
         f"{len(ab)} abnormal + {len(nm)} normal clips.\n",
         f"Baseline (fixed 0.5): overall catch **{report['baseline_fixed_0.5']['overall_catch']}**, "
         f"FP {report['baseline_fixed_0.5']['overall_FP']}\n",
         "## Global normal-calibrated thresholds (one threshold, all incidents)\n",
         "| FP budget | threshold | overall catch | actual FP | Las Vegas | Times Square | Love Parade | Italy |",
         "|---|---|---|---|---|---|---|---|"]
    for r in global_tbl:
        pc = r["per_incident_catch"]
        L.append(f"| {r['fp_budget']:.0%} | {r['threshold']} | **{r['overall_catch']}** | "
                 f"{r['overall_FP_actual']} | {pc.get('Las Vegas','-')} | {pc.get('Times Square','-')} | "
                 f"{pc.get('Love Parade','-')} | {pc.get('Italy','-')} |")
    L += ["", "## Per-incident self-calibration (own normal baseline)\n",
          "| Incident | n abn / norm | threshold | catch rate |", "|---|---|---|---|"]
    for inc, e in per_inc.items():
        if "catch_rate" in e:
            L.append(f"| {inc} | {e['n_abnormal']}/{e['n_normal']} | "
                     f"{e['threshold_above_site_normal']} | **{e['catch_rate']}** |")
        else:
            L.append(f"| {inc} | {e['n_abnormal']}/{e['n_normal']} | — | {e['note']} |")
    (OUT / "gen_calibration.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
