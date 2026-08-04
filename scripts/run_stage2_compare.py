#!/usr/bin/env python3
"""Cross-validated Stage 2 escalation comparison across three binary models.

Models:
  1. XGBoost @ 0.50        (scale_pos_weight=5, 19 features)
  2. A5 RF top-12          (cost-sensitive {Safe:1, Risk:5}, 12 selected features)
  3. Binary (12 features)  (balanced RF, original 12 features)

For every leakage-safe fold we train the model on the fold's training clips,
score P(Stampede Risk) per held-out frame, build synthetic Safe->Risk sequences
from the held-out videos (known join = ground-truth transition), and run the
escalation detector. Latency = transition_frame - first_alert_frame
(positive = early warning, negative = late). All models use the same 0.50
alert-crossing threshold so the comparison is apples-to-apples.

Outputs -> outputs_improved/stage2_compare.{json,csv,png}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from run_binary import build_binary_sequences, detect_risk_alerts
from stampede.classifier import Normalizer
from stampede.config import FEATURE_COLUMNS, RANDOM_STATE
from stampede.dataset import stratified_group_folds
from stampede.escalation import clip_timelines

POS, NEG = "Stampede Risk", "Safe"
OUT = Path("outputs_improved")
ALERT_THRESHOLD = 0.50
TOLERANCE = 40
TOP12 = ["detection_confidence_mean", "bbox_area_mean", "detection_confidence_std",
         "spatial_density_mismatch", "stop_go", "flow_mag_mean",
         "direction_consistency", "bbox_area_variance", "flow_mag_var",
         "velocity_var", "bbox_overlap_ratio", "crowd_density"]
BASE12 = [  # original pre-compression feature set
    "person_count", "crowd_density", "density_norm", "flow_mag_mean",
    "flow_mag_var", "velocity_per_person", "velocity_var", "motion_instability",
    "stop_go", "direction_consistency", "trajectory_dispersion", "density_trend"]


def load(csv):
    d = pd.read_csv(csv)
    d["label"] = np.where(d.label == "Panic", POS, NEG)
    d["y"] = (d.label == POS).astype(int)
    return d


def rf_scorer(class_weight, cols):
    def score(tr, te):
        norm = Normalizer.fit(tr, cols)
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                     class_weight=class_weight,
                                     random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(norm.transform(tr), tr.label.values)
        idx = list(clf.classes_).index(POS)
        return clf.predict_proba(norm.transform(te))[:, idx]
    return score


def xgb_scorer(cols):
    def score(tr, te):
        import xgboost as xgb
        dtr = xgb.DMatrix(tr[cols].fillna(0).to_numpy(), label=tr.y.values)
        bst = xgb.train({"max_depth": 6, "learning_rate": 0.05,
                         "objective": "binary:logistic", "scale_pos_weight": 5,
                         "seed": RANDOM_STATE, "eval_metric": "logloss"},
                        dtr, num_boost_round=200)
        return bst.predict(xgb.DMatrix(te[cols].fillna(0).to_numpy()))
    return score


def stage2_on_fold(te_frames, cols, scores):
    """Build sequences from held-out videos, return per-seq (detected, latency)
    and (n_stable, n_fp) for false positives on stable Safe timelines."""
    te = te_frames.copy()
    te["risk"] = scores
    tl = clip_timelines(te)
    seqs = build_binary_sequences(tl, max_sequences=40)
    per_seq = []
    for s in seqs:
        res = detect_risk_alerts(s["risk"], ALERT_THRESHOLD)
        hits = [a for a in res["alerts"]
                if abs(a["frame"] - s["transition_frame"]) <= TOLERANCE]
        if hits:
            lat = s["transition_frame"] - min(h["frame"] for h in hits)
            per_seq.append((True, lat))
        else:
            per_seq.append((False, None))
    # False positives on stable Safe videos.
    n_stable = n_fp = 0
    for _, g in tl.items():
        if g.label.iloc[0] != NEG:
            continue
        n_stable += 1
        if detect_risk_alerts(g.risk.to_numpy(), ALERT_THRESHOLD)["alerts"]:
            n_fp += 1
    return per_seq, n_stable, n_fp


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d19 = load("outputs_compression/frame_features.csv")
    d12 = load("outputs/frame_features.csv")
    clips = d19[["path", "video_id", "label"]].drop_duplicates("path").reset_index(drop=True)
    folds = stratified_group_folds(clips, n_splits=5)

    models = {
        "XGBoost @ 0.50": dict(data=d19, cols=FEATURE_COLUMNS, scorer=xgb_scorer(FEATURE_COLUMNS)),
        "A5 RF top-12": dict(data=d19, cols=TOP12, scorer=rf_scorer({NEG: 1.0, POS: 5.0}, TOP12)),
        "Binary (12 feat)": dict(data=d12, cols=BASE12, scorer=rf_scorer("balanced", BASE12)),
    }

    summary = {}
    fold_latencies = {m: [[] for _ in folds] for m in models}  # per-model per-fold
    for name, cfg in models.items():
        data, cols, scorer = cfg["data"], cfg["cols"], cfg["scorer"]
        det = tot = n_stable = n_fp = 0
        all_lat = []
        for fi, (tr_idx, te_idx) in enumerate(folds):
            tr_paths = set(clips.loc[tr_idx, "path"]); te_paths = set(clips.loc[te_idx, "path"])
            tr = data[data.path.isin(tr_paths)]; te = data[data.path.isin(te_paths)]
            scores = scorer(tr, te)
            per_seq, ns, nf = stage2_on_fold(te, cols, scores)
            n_stable += ns; n_fp += nf
            for detected, lat in per_seq:
                tot += 1
                if detected:
                    det += 1
                    all_lat.append(lat)
                    fold_latencies[name][fi].append(lat)
        summary[name] = {
            "detection_rate": round(det / tot, 3) if tot else 0.0,
            "median_latency_frames": round(float(np.median(all_lat)), 1) if all_lat else None,
            "mean_latency_frames": round(float(np.mean(all_lat)), 1) if all_lat else None,
            "false_positive_rate": round(n_fp / n_stable, 3) if n_stable else 0.0,
            "n_sequences": tot, "n_detected": det, "n_stable_videos": n_stable,
        }
        s = summary[name]
        print(f"{name:20s} det={s['detection_rate']:.2f} med_lat={s['median_latency_frames']} "
              f"mean_lat={s['mean_latency_frames']} FP={s['false_positive_rate']:.2f} "
              f"(seq={tot}, stable={n_stable})", flush=True)

    (OUT / "stage2_compare.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(summary).T.to_csv(OUT / "stage2_compare.csv")

    # Plot: per-fold latency boxplots, one panel per model.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    colors = {"XGBoost @ 0.50": "#d93025", "A5 RF top-12": "#1a73e8",
              "Binary (12 feat)": "#f9ab00"}
    for ax, name in zip(axes, models):
        data = [fl if fl else [np.nan] for fl in fold_latencies[name]]
        bp = ax.boxplot(data, positions=range(1, len(folds) + 1), patch_artist=True,
                        widths=0.6, showmeans=True)
        for box in bp["boxes"]:
            box.set(facecolor=colors[name], alpha=0.5)
        # jittered points
        for fi, fl in enumerate(fold_latencies[name], 1):
            if fl:
                ax.scatter(np.full(len(fl), fi) + np.linspace(-0.15, 0.15, len(fl)),
                           fl, s=12, color=colors[name], alpha=0.7, zorder=3)
        ax.axhline(0, color="k", ls="--", lw=1)
        med = summary[name]["median_latency_frames"]
        ax.set_title(f"{name}\ndet={summary[name]['detection_rate']:.0%}, "
                     f"median lat={med}f, FP={summary[name]['false_positive_rate']:.0%}",
                     fontsize=10)
        ax.set_xlabel("CV fold")
    axes[0].set_ylabel("Latency (frames)  ↑ early / ↓ late")
    fig.suptitle("Stage 2 escalation latency across folds "
                 "(positive = early warning, negative = late)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "stage2_latency_by_fold.png", dpi=130)
    plt.close(fig)
    print(f"\nWrote {OUT}/stage2_compare.json/.csv and stage2_latency_by_fold.png")


if __name__ == "__main__":
    main()
