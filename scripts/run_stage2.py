#!/usr/bin/env python3
"""Stage 2: escalation detection and early-alert evaluation.

Scores every frame with the trained Stage 1 model, builds per-video risk
timelines, evaluates:
  * transition detection rate + latency on synthetic escalation sequences,
  * false-positive rate + risk smoothness on real single-class timelines,
and writes an escalation-events CSV, a metrics report, and sample risk-curve
plots (both synthetic transitions and real clips).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stampede.classifier import Stage1Model
from stampede.escalation import (
    build_synthetic_sequences,
    clip_timelines,
    detect_alerts,
    evaluate_false_positives,
    evaluate_transitions,
)
from stampede.visualize import plot_risk_curve


def _youden_threshold(pos: np.ndarray, neg: np.ndarray) -> float:
    """Threshold maximising TPR(pos) - FPR(neg) over the observed risk range."""
    grid = np.linspace(0.3, 0.95, 66)
    best_t, best_j = 0.8, -1.0
    for t in grid:
        tpr = (pos >= t).mean() if len(pos) else 0.0
        fpr = (neg >= t).mean() if len(neg) else 0.0
        if tpr - fpr > best_j:
            best_j, best_t = tpr - fpr, float(t)
    return round(best_t, 3)


def _calibrate_thresholds(feats: pd.DataFrame) -> dict:
    """Derive data-driven band edges from the per-frame risk distribution.

    Crowdy->Panic edge separates Panic (positive) from Crowdy (negative);
    Normal->Crowdy edge separates {Crowdy,Panic} from {No Panic, Normal}.
    """
    risk = feats.risk.to_numpy()
    lab = feats.label.to_numpy()
    cp_high = _youden_threshold(risk[lab == "Panic"], risk[lab == "Crowdy"])
    nc_high = _youden_threshold(
        risk[np.isin(lab, ["Crowdy", "Panic"])],
        risk[np.isin(lab, ["No Panic", "Normal"])],
    )
    return {
        "nc_low": round(nc_high - 0.1, 3), "nc_high": nc_high, "nc_velocity": 0.003,
        "cp_low": round(cp_high - 0.1, 3), "cp_high": cp_high, "cp_velocity": 0.003,
    }


def _run_escalation_eval(timelines, max_sequences, thresholds=None) -> dict:
    """Transition detection + false-positive/smoothness for one threshold set."""
    transition = {}
    for seq_type in ("normal_to_crowdy", "crowdy_to_panic"):
        seqs = build_synthetic_sequences(timelines, seq_type, max_sequences)
        transition[seq_type] = {
            k: v
            for k, v in evaluate_transitions(seqs, thresholds=thresholds).items()
            if k != "per_sequence"
        }
    fp = {
        k: v
        for k, v in evaluate_false_positives(timelines, thresholds).items()
        if k != "per_video"
    }
    return {"transition_detection": transition, "false_positives_and_smoothness": fp}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("outputs"))
    ap.add_argument("--eval-split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--max-sequences", type=int, default=20)
    a = ap.parse_args()

    plot_dir = a.output / "stage2_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    model = Stage1Model.load(a.output / "stage1_model.pkl")
    feats = pd.read_csv(a.output / "frame_features.csv")
    if a.eval_split != "all":
        feats = feats[feats.split == a.eval_split].reset_index(drop=True)
    feats = feats.copy()
    feats["risk"] = model.risk_score(feats)

    timelines = clip_timelines(feats)

    # --- Synthetic escalation sequences (known transition points) ---
    events = []
    transition_metrics = {}
    for seq_type in ("normal_to_crowdy", "crowdy_to_panic"):
        seqs = build_synthetic_sequences(timelines, seq_type, a.max_sequences)
        transition_metrics[seq_type] = evaluate_transitions(seqs)
        # Log detected events + plot the first few sequences.
        for k, seq in enumerate(seqs):
            res = detect_alerts(seq["risk"])
            for al in res["alerts"]:
                latency = seq["transition_frame"] - al["frame"]
                events.append(
                    dict(
                        sequence_type=seq_type,
                        a_video=seq["a_video"],
                        b_video=seq["b_video"],
                        transition_type=al["type"],
                        frame_detected=al["frame"],
                        confidence=round(al["confidence"], 3),
                        latency=latency,
                        transition_frame=seq["transition_frame"],
                    )
                )
            if k < 3:
                plot_risk_curve(
                    seq["risk"],
                    res["risk_smoothed"],
                    res["alerts"],
                    plot_dir / f"synthetic_{seq_type}_{k}.png",
                    f"Synthetic {seq_type}: {seq['a_video']} -> {seq['b_video']}",
                    transition_frame=seq["transition_frame"],
                )

    # --- Real single-class timelines (false positives + smoothness) ---
    fp_metrics = evaluate_false_positives(timelines)
    # Plot a few representative real timelines (one per class if available).
    plotted_labels = set()
    for vid, g in timelines.items():
        lab = g.label.iloc[0]
        if lab in plotted_labels:
            continue
        plotted_labels.add(lab)
        res = detect_alerts(g.risk.to_numpy())
        plot_risk_curve(
            g.risk.to_numpy(),
            res["risk_smoothed"],
            res["alerts"],
            plot_dir / f"real_{lab.replace(' ', '_')}_{vid}.png",
            f"Real timeline ({lab}): {vid}",
        )

    pd.DataFrame(events).to_csv(a.output / "escalation_events.csv", index=False)

    # Data-calibrated thresholds: show the achievable trade-off when band edges
    # are matched to this leakage-free model's risk calibration (Panic and
    # Crowdy risk overlap heavily, so the spec's 0.8 edge is rarely reached).
    calibrated = _calibrate_thresholds(feats)
    calibrated_eval = _run_escalation_eval(timelines, a.max_sequences, calibrated)

    report = {
        "eval_split": a.eval_split,
        "spec_thresholds": {
            "transition_detection": transition_metrics,
            "false_positives_and_smoothness": {
                k: v for k, v in fp_metrics.items() if k != "per_video"
            },
        },
        "calibrated_thresholds": {
            "thresholds": calibrated,
            **calibrated_eval,
            "note": "Band edges derived from the test risk distribution "
            "(Youden's J). Reported to show the achievable detection/false-"
            "alarm trade-off given Crowdy/Panic risk overlap; the spec edges "
            "(0.5 / 0.8) are the primary result.",
        },
        "total_alerts_logged": len(events),
    }
    (a.output / "stage2_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
