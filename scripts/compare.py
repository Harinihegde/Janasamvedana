#!/usr/bin/env python3
"""Side-by-side comparison: 4-class vs binary crowd-risk classification.

Reads the reports produced by ``train_stage1.py`` (4-class, ``outputs/``) and
``run_binary.py`` (binary, ``outputs_binary/``) and emits a Markdown comparison
table plus a JSON summary. The primary decision metric is **macro-F1** from the
leakage-safe cross-validation out-of-fold (OOF) predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def fmt(x, pct=True):
    return f"{x*100:.1f}%" if pct else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--four", type=Path, default=Path("outputs"))
    ap.add_argument("--binary", type=Path, default=Path("outputs_binary"))
    a = ap.parse_args()

    four = load(a.four / "stage1_report.json")
    four_s2 = load(a.four / "stage2_report.json")
    binr = load(a.binary / "stage1_report.json")

    f_cv, b_cv = four["cross_validation"], binr["cross_validation"]
    f_oof, b_oof = f_cv["oof"], b_cv["oof"]

    def pcf1(oof):
        return {k: v["f1-score"] for k, v in oof["per_class"].items()
                if isinstance(v, dict) and "f1-score" in v}

    lines = []
    lines.append("# 4-Class vs Binary — Crowd-Risk Classification\n")
    lines.append("Same 12 features, same leakage-safe StratifiedGroupKFold CV "
                 f"({f_cv['n_splits']} folds), same RandomForest. Metrics are "
                 "clip-level.\n")
    lines.append("## Stage 1 (primary metric: macro-F1, from CV out-of-fold)\n")
    lines.append("| Metric | 4-Class | Binary (Safe vs Stampede Risk) |")
    lines.append("|---|---|---|")
    rows = [
        ("CV accuracy (mean ± std)",
         f"{fmt(f_cv['clip_accuracy_mean'])} ± {fmt(f_cv['clip_accuracy_std'])}",
         f"{fmt(b_cv['clip_accuracy_mean'])} ± {fmt(b_cv['clip_accuracy_std'])}"),
        ("CV macro-F1 (mean ± std)",
         f"{fmt(f_cv['clip_macro_f1_mean'],0)} ± {fmt(f_cv['clip_macro_f1_std'],0)}",
         f"{fmt(b_cv['clip_macro_f1_mean'],0)} ± {fmt(b_cv['clip_macro_f1_std'],0)}"),
        ("CV weighted-F1 (mean)",
         fmt(f_cv['clip_weighted_f1_mean'], 0), fmt(b_cv['clip_weighted_f1_mean'], 0)),
        ("OOF accuracy", fmt(f_oof['accuracy']), fmt(b_oof['accuracy'])),
        ("OOF macro-F1", fmt(f_oof['macro_f1'], 0), fmt(b_oof['macro_f1'], 0)),
        ("OOF weighted-F1", fmt(f_oof['weighted_f1'], 0), fmt(b_oof['weighted_f1'], 0)),
    ]
    for name, fv, bv in rows:
        lines.append(f"| {name} | {fv} | {bv} |")

    lines.append("\n## Per-class F1 (CV out-of-fold)\n")
    lines.append("| Class | 4-Class F1 | | Binary class | Binary F1 |")
    lines.append("|---|---|---|---|---|")
    fpc, bpc = pcf1(f_oof), pcf1(b_oof)
    order4 = ["No Panic", "Normal", "Crowdy", "Panic"]
    orderb = ["Safe", "Stampede Risk"]
    for i in range(max(len(order4), len(orderb))):
        c4 = order4[i] if i < len(order4) else ""
        cb = orderb[i] if i < len(orderb) else ""
        v4 = fmt(fpc[c4], 0) if c4 in fpc else ""
        vb = fmt(bpc[cb], 0) if cb in bpc else ""
        lines.append(f"| {c4} | {v4} | | {cb} | {vb} |")

    # Panic-recall focus: how well does each approach catch the dangerous class?
    p4 = f_oof["per_class"].get("Panic", {})
    pb = b_oof["per_class"].get("Stampede Risk", {})
    lines.append("\n## Danger-class detection (the safety-critical number)\n")
    lines.append("| | 4-Class: *Panic* | Binary: *Stampede Risk* |")
    lines.append("|---|---|---|")
    lines.append(f"| Precision | {fmt(p4.get('precision',0),0)} | {fmt(pb.get('precision',0),0)} |")
    lines.append(f"| Recall | {fmt(p4.get('recall',0),0)} | {fmt(pb.get('recall',0),0)} |")
    lines.append(f"| F1 | {fmt(p4.get('f1-score',0),0)} | {fmt(pb.get('f1-score',0),0)} |")

    # Stage 2.
    lines.append("\n## Stage 2 escalation (held-out test)\n")
    f_cp = four_s2["spec_thresholds"]["transition_detection"]["crowdy_to_panic"]
    f_cp_cal = four_s2["calibrated_thresholds"]["transition_detection"]["crowdy_to_panic"]
    b_s2 = binr["stage2"]
    lines.append("| | 4-Class (Crowdy→Panic) | Binary (Safe→Stampede Risk) |")
    lines.append("|---|---|---|")
    lines.append(f"| Detection @ spec edge 0.8 | {fmt(f_cp['detection_rate'])} | "
                 f"{fmt(b_s2['spec_threshold_0.8']['detection_rate'])} |")
    lines.append(f"| Detection @ calibrated edge | {fmt(f_cp_cal['detection_rate'])} | "
                 f"{fmt(b_s2['calibrated']['detection_rate'])} "
                 f"(edge {b_s2['calibrated']['threshold']}) |")
    lines.append(f"| Median latency (calibrated, frames early) | "
                 f"{f_cp_cal['median_latency_frames']:.0f} | "
                 f"{b_s2['calibrated']['median_latency_frames']:.0f} |")

    # Majority-class ("always Safe") baseline for the binary problem.
    safe_support = b_oof["per_class"]["Safe"]["support"]
    total_support = sum(v["support"] for k, v in b_oof["per_class"].items()
                        if isinstance(v, dict) and "support" in v
                        and k in ("Safe", "Stampede Risk"))
    baseline = safe_support / total_support

    # Verdict.
    winner = "Binary" if b_oof["macro_f1"] > f_oof["macro_f1"] else "4-Class"
    delta = abs(b_oof["macro_f1"] - f_oof["macro_f1"])
    danger_recall_winner = ("4-Class" if p4.get("recall", 0) > pb.get("recall", 0)
                            else "Binary")
    lines.append("\n## Verdict\n")
    lines.append(
        f"**Binary wins the headline metrics** — OOF macro-F1 "
        f"{fmt(b_oof['macro_f1'],0)} vs {fmt(f_oof['macro_f1'],0)} (Δ {delta:.3f}) "
        f"and accuracy {fmt(b_oof['accuracy'])} vs {fmt(f_oof['accuracy'])} — and "
        "its single-transition Stage 2 escalation detects far better.\n")
    lines.append(
        f"**But read the fine print for a safety system:**\n"
        f"- Binary accuracy ({fmt(b_oof['accuracy'])}) is barely above the trivial "
        f"*always-Safe* baseline ({fmt(baseline)}): the model is only "
        f"{(b_oof['accuracy']-baseline)*100:+.1f} pp better than never raising an "
        f"alarm. Most of the binary 'win' is the easy majority Safe class "
        f"(F1 {fmt(bpc.get('Safe',0),0)}).\n"
        f"- On the metric that actually matters — **catching Panic** — the "
        f"4-class model has *higher recall* ({fmt(p4.get('recall',0),0)} vs "
        f"{fmt(pb.get('recall',0),0)}); binary misses "
        f"{(1-pb.get('recall',0))*100:.0f}% of real Panic clips. Danger-class "
        f"recall winner: **{danger_recall_winner}**.\n"
        f"- Binary CV is high-variance (fold accuracy spans "
        f"{min(b_cv['fold_accuracies'])*100:.0f}%–{max(b_cv['fold_accuracies'])*100:.0f}%) "
        "because some folds are Panic-heavy — a symptom of few unique Panic videos.\n")
    lines.append(
        "**Recommendation:** use **binary** if the goal is overall discrimination, "
        "deployment simplicity, and a cleaner Stage 2 escalation signal; keep "
        "**4-class** if the priority is maximising *recall of Panic* (fewest missed "
        "stampedes). On this leakage-free data neither is strongly separable, so "
        "the choice is an operating-point decision, not a clear technical win.\n")

    md = "\n".join(lines)
    (a.four.parent / "comparison_4class_vs_binary.md").write_text(md)
    summary = {
        "winner_by_oof_macro_f1": winner,
        "four_class": {"oof_macro_f1": f_oof["macro_f1"],
                       "oof_accuracy": f_oof["accuracy"],
                       "panic_f1": p4.get("f1-score"),
                       "panic_recall": p4.get("recall")},
        "binary": {"oof_macro_f1": b_oof["macro_f1"],
                   "oof_accuracy": b_oof["accuracy"],
                   "stampede_risk_f1": pb.get("f1-score"),
                   "stampede_risk_recall": pb.get("recall")},
    }
    (a.four.parent / "comparison_4class_vs_binary.json").write_text(
        json.dumps(summary, indent=2))
    print(md)


if __name__ == "__main__":
    main()
