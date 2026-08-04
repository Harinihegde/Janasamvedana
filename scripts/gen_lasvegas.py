#!/usr/bin/env python3
"""Strengthen the generalization result: threshold-free AUC across all 4
incidents + a Las Vegas case study (continuous normal->panic risk timeline with
Stage-2 alert). Reuses gen_features.csv (no re-streaming).
Output -> outputs_improved/gen_auc.json + gen_lasvegas_timeline.png
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from stampede.classifier import train
from stampede.config import FEATURE_COLUMNS
from stampede.escalation import smooth, risk_velocity

OUT = Path("outputs_improved")
POS, NEG = "Stampede Risk", "Safe"


def chunk_num(p):
    m = re.search(r"chunk(\d+)", p)
    return int(m.group(1)) if m else 0


def main():
    g = pd.read_csv(OUT / "gen_features.csv")
    cur = pd.read_csv("outputs_compression2/frame_features.csv")
    cur["label"] = np.where(cur.label == "Panic", POS, NEG)
    model = train(cur, feature_cols=FEATURE_COLUMNS, risk_anchors={NEG: 0.0, POS: 1.0})
    g["risk"] = model.risk_score(g)
    g["sub"] = g.path.str.split("/").str[1]
    g["chunk"] = g.path.apply(chunk_num)

    # clip-level risk
    clip = (g.groupby(["incident", "kind", "type", "path"]).risk.mean().reset_index())

    # ---- AUC (threshold-free): abnormal(1) vs normal(0) ----
    auc = {}
    ab_all, nm_all = [], []
    for inc, c in clip.groupby("incident"):
        ab = c[c.type == "abnormal"].risk.values
        nm = c[c.type == "normal"].risk.values
        ab_all += list(ab); nm_all += list(nm)
        if len(ab) and len(nm):
            y = [1]*len(ab) + [0]*len(nm)
            auc[inc] = round(float(roc_auc_score(y, list(ab)+list(nm))), 3)
        else:
            auc[inc] = "n/a (one class only)"
    y_all = [1]*len(ab_all) + [0]*len(nm_all)
    overall_auc = round(float(roc_auc_score(y_all, ab_all + nm_all)), 3)
    sep = {"abnormal_mean_risk": round(float(np.mean(ab_all)), 3),
           "normal_mean_risk": round(float(np.mean(nm_all)), 3)}

    # ---- Las Vegas case study: Train(normal) -> Test_1(abnormal), same angle ----
    # rows are already in stream order (Train chunks, then Test_1 chunks); keep it
    lv = g[(g.incident == "Las Vegas") & (g["sub"].isin(["Train", "Test_1"]))].copy()
    lv["order"] = lv["sub"].map({"Train": 0, "Test_1": 1})
    lv = lv.sort_values(["order", "chunk"], kind="stable").reset_index(drop=True)
    risk_raw = lv.risk.to_numpy()
    transition = int((lv["sub"] == "Train").sum())   # onset of abnormal
    rs = smooth(risk_raw, window=9)
    vel = risk_velocity(rs, window=10)
    # alert: smoothed risk crosses 0.5 upward with positive velocity, after transition-search
    alert = None
    for i in range(1, len(rs)):
        if rs[i-1] < 0.5 <= rs[i] and vel[i] > 0:
            alert = i; break
    latency = (alert - transition) if alert is not None else None

    lasvegas = {
        "transition_frame_idx": transition, "n_frames": int(len(rs)),
        "normal_mean_risk": round(float(rs[:transition].mean()), 3),
        "abnormal_mean_risk": round(float(rs[transition:].mean()), 3),
        "alert_frame_idx": alert,
        "alert_latency_frames_after_onset": latency,
    }

    report = {"per_incident_AUC": auc, "overall_AUC": overall_auc,
              "risk_separation": sep, "las_vegas_case_study": lasvegas,
              "note": "AUC over clip-level mean risk. Italy has no normal clips "
                      "-> no AUC. Overall AUC pools all incidents."}
    (OUT / "gen_auc.json").write_text(json.dumps(report, indent=2))

    # plot
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(rs))
    ax.plot(x, risk_raw, color="#9aa0a6", lw=.7, alpha=.5, label="risk (raw)")
    ax.plot(x, rs, color="#1a73e8", lw=2, label="risk (smoothed)")
    ax.axvline(transition, color="#d93025", ls="--", lw=1.5, label="panic onset (Train→Test_1)")
    ax.axhline(0.5, color="#f9ab00", ls=":", lw=1, label="alert threshold 0.5")
    if alert is not None:
        ax.axvline(alert, color="#188038", lw=1.5, label=f"ALERT (+{latency}f after onset)")
    ax.set_ylim(0, 1); ax.set_xlabel("sampled frame (Las Vegas: normal → gunfire panic)")
    ax.set_ylabel("Stampede-risk score")
    ax.set_title("Las Vegas Mandalay Bay CCTV — risk rises at panic onset (external, unseen in training)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "gen_lasvegas_timeline.png", dpi=130); plt.close(fig)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
