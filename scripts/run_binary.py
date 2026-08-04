#!/usr/bin/env python3
"""Binary variant: Safe vs Stampede Risk.

Regroups the 4-class labels into a binary problem and re-runs Stage 1 and
Stage 2 on the **already-extracted** per-frame features (no re-extraction):

    Safe          = No Panic + Normal + Crowdy
    Stampede Risk = Panic

Reuses the exact same 12 features, leakage-safe grouped split and
StratifiedGroupKFold CV as the 4-class pipeline, so the two are directly
comparable. Binary risk score = P(Stampede Risk). Stage 2 detects the single
Safe -> Stampede Risk escalation.

Outputs go to ``--output`` (default ``outputs_binary``); the 4-class artifacts
in ``outputs/`` are left untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stampede.classifier import train
from stampede.config import FEATURE_COLUMNS
from stampede.escalation import clip_timelines, risk_velocity, smooth
from stampede.evaluate import cross_validate, evaluate_holdout
from stampede.visualize import plot_confusion, plot_cv_summary, plot_risk_curve

BINARY_LABELS = ["Safe", "Stampede Risk"]
BINARY_ANCHORS = {"Safe": 0.0, "Stampede Risk": 1.0}
SPEC_THRESHOLD = 0.8  # same "Panic" band edge the 4-class Stage 2 uses


def to_binary(label: pd.Series) -> pd.Series:
    return np.where(label == "Panic", "Stampede Risk", "Safe")


# ---------------------------------------------------------------------------
# Binary Stage 2: single Safe -> Stampede Risk transition
# ---------------------------------------------------------------------------
def detect_risk_alerts(risk_raw, high, velocity=0.004, cooldown=15):
    """Alert when smoothed risk crosses ``high`` upward with rising velocity."""
    risk = smooth(risk_raw)
    vel = risk_velocity(risk)
    alerts, last = [], -10**9
    for i in range(1, len(risk)):
        recent_below = risk[max(0, i - 30) : i + 1].min() < high - 0.1
        if (
            risk[i - 1] < high
            and risk[i] >= high
            and vel[i] > velocity
            and recent_below
            and i - last > cooldown
        ):
            alerts.append(
                dict(type="safe_to_risk", frame=int(i), risk=float(risk[i]),
                     velocity=float(vel[i]),
                     confidence=float(min(1.0, vel[i] / velocity / 3.0 + 0.5)))
            )
            last = i
    return {"risk_smoothed": risk, "velocity": vel, "alerts": alerts}


def build_binary_sequences(timelines, max_sequences=20):
    safe = [v for v, g in timelines.items() if g.label.iloc[0] == "Safe"]
    risk = [v for v, g in timelines.items() if g.label.iloc[0] == "Stampede Risk"]
    seqs = []
    for k in range(min(max_sequences, len(safe) * len(risk))):
        a = timelines[safe[k % len(safe)]]
        b = timelines[risk[k % len(risk)]]
        seqs.append(dict(
            risk=np.concatenate([a.risk.to_numpy(), b.risk.to_numpy()]),
            transition_frame=int(len(a)),
            a_video=safe[k % len(safe)], b_video=risk[k % len(risk)]))
    return seqs


def eval_binary_stage2(timelines, high, max_sequences=20, tolerance=40):
    seqs = build_binary_sequences(timelines, max_sequences)
    detected, latencies = 0, []
    for s in seqs:
        res = detect_risk_alerts(s["risk"], high)
        hits = [a for a in res["alerts"]
                if abs(a["frame"] - s["transition_frame"]) <= tolerance]
        if hits:
            detected += 1
            latencies.append(s["transition_frame"] - min(h["frame"] for h in hits))
    # False positives on stable Safe timelines.
    fp = fp_total = 0
    for _, g in timelines.items():
        if g.label.iloc[0] != "Safe":
            continue
        fp_total += 1
        if detect_risk_alerts(g.risk.to_numpy(), high)["alerts"]:
            fp += 1
    return {
        "threshold": high,
        "n_sequences": len(seqs),
        "detection_rate": detected / len(seqs) if seqs else 0.0,
        "mean_latency_frames": float(np.mean(latencies)) if latencies else 0.0,
        "median_latency_frames": float(np.median(latencies)) if latencies else 0.0,
        "false_positive_rate": fp / fp_total if fp_total else 0.0,
        "n_stable_videos": fp_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=Path("outputs/frame_features.csv"))
    ap.add_argument("--output", type=Path, default=Path("outputs_binary"))
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--max-sequences", type=int, default=20)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    plot_dir = a.output / "stage2_plots"
    plot_dir.mkdir(exist_ok=True)

    feats = pd.read_csv(a.features)
    feats["label"] = to_binary(feats["label"])
    train_df = feats[feats.split == "train"].reset_index(drop=True)
    test_df = feats[feats.split == "test"].reset_index(drop=True)

    # --- Stage 1: holdout + CV (leakage-safe, identical protocol) ---
    holdout = evaluate_holdout(train_df, test_df, BINARY_LABELS, BINARY_ANCHORS)
    cv = cross_validate(feats, a.cv_splits, BINARY_LABELS, BINARY_ANCHORS)

    model = train(train_df, risk_anchors=BINARY_ANCHORS)
    model.save(a.output / "stage1_model.pkl")
    (a.output / "normalization_params.json").write_text(
        json.dumps(model.normalizer.to_dict(), indent=2))

    y_pred = model.predict(test_df)
    risk = model.risk_score(test_df)
    pd.DataFrame(dict(
        video_id=test_df.video_id, path=test_df.path, frame_number=test_df.frame_index,
        actual_label=test_df.label, predicted_label=y_pred, risk_score=risk,
    )).to_csv(a.output / "test_predictions.csv", index=False)

    # --- Stage 2: Safe -> Stampede Risk ---
    feats["risk"] = model.risk_score(feats)
    test_tl = clip_timelines(feats[feats.split == "test"].copy())
    # Data-calibrated threshold: Youden's J on P(risk) for Panic vs non-Panic.
    r = feats.risk.to_numpy(); is_panic = feats.label.to_numpy() == "Stampede Risk"
    grid = np.linspace(0.3, 0.95, 66)
    cal = max(grid, key=lambda t: (r[is_panic] >= t).mean() - (r[~is_panic] >= t).mean())
    stage2 = {
        "spec_threshold_0.8": eval_binary_stage2(test_tl, SPEC_THRESHOLD, a.max_sequences),
        "calibrated": eval_binary_stage2(test_tl, round(float(cal), 3), a.max_sequences),
    }
    # Sample plots.
    for k, s in enumerate(build_binary_sequences(test_tl, a.max_sequences)[:3]):
        res = detect_risk_alerts(s["risk"], round(float(cal), 3))
        plot_risk_curve(s["risk"], res["risk_smoothed"], res["alerts"],
                        plot_dir / f"synthetic_safe_to_risk_{k}.png",
                        f"Binary Safe->Risk: {s['a_video']} -> {s['b_video']}",
                        transition_frame=s["transition_frame"])

    # --- Report + visuals ---
    report = {
        "label_scheme": {"Safe": "No Panic + Normal + Crowdy", "Stampede Risk": "Panic"},
        "features_used": FEATURE_COLUMNS,
        "holdout": holdout,
        "cross_validation": cv,
        "stage2": stage2,
        "feature_importances": dict(sorted(
            zip(FEATURE_COLUMNS, model.rf.feature_importances_.tolist()),
            key=lambda kv: kv[1], reverse=True)),
    }
    (a.output / "stage1_report.json").write_text(json.dumps(report, indent=2))
    plot_confusion(cv["oof"]["confusion_matrix"], cv["oof"]["confusion_labels"],
                   a.output / "confusion_matrix_cv_oof.png",
                   f"Binary CV OOF confusion (acc={cv['oof']['accuracy']:.2%})")
    plot_confusion(holdout["clip_level"]["confusion_matrix"],
                   holdout["clip_level"]["confusion_labels"],
                   a.output / "confusion_matrix_clip.png",
                   f"Binary clip-level confusion (acc={holdout['clip_level']['accuracy']:.2%})")
    plot_cv_summary(cv["fold_accuracies"], a.output / "cv_accuracy.png")

    print(json.dumps({
        "cv_clip_accuracy": f"{cv['clip_accuracy_mean']:.3f} ± {cv['clip_accuracy_std']:.3f}",
        "cv_clip_macro_f1": f"{cv['clip_macro_f1_mean']:.3f} ± {cv['clip_macro_f1_std']:.3f}",
        "oof_per_class_f1": {k: round(v["f1-score"], 3)
                             for k, v in cv["oof"]["per_class"].items()
                             if isinstance(v, dict) and "f1-score" in v},
        "stage2": stage2,
    }, indent=2))


if __name__ == "__main__":
    main()
