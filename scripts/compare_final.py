#!/usr/bin/env python3
"""Master comparison across all four trained models.

    4-Class(12)  = outputs/
    4-Class(19)  = outputs_compression/
    Binary(12)   = outputs_binary/
    Binary(19)   = outputs_binary_19/

All share the identical leakage-safe split and 5-fold StratifiedGroupKFold CV.
The "danger class" is Panic (4-class) or Stampede Risk (binary). Primary metric
is danger-class F1 / recall (does the system catch the dangerous state?).
"""
from __future__ import annotations

import json
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text())


def pct(x):
    return f"{x*100:.1f}%"


def f3(x):
    return f"{x:.3f}"


def danger_metrics(report, danger_label):
    oof = report["cross_validation"]["oof"]
    d = oof["per_class"].get(danger_label, {})
    return {
        "f1": d.get("f1-score", 0.0),
        "recall": d.get("recall", 0.0),
        "precision": d.get("precision", 0.0),
        "macro_f1": oof["macro_f1"],
        "accuracy": oof["accuracy"],
    }


def stage2_4class(s2):
    return {
        "spec": s2["spec_thresholds"]["transition_detection"]["crowdy_to_panic"]["detection_rate"],
        "cal": s2["calibrated_thresholds"]["transition_detection"]["crowdy_to_panic"]["detection_rate"],
    }


def stage2_binary(report):
    s2 = report["stage2"]
    return {"spec": s2["spec_threshold_0.8"]["detection_rate"],
            "cal": s2["calibrated"]["detection_rate"]}


def main():
    root = Path(".")
    four12 = load(root / "outputs" / "stage1_report.json")
    four19 = load(root / "outputs_compression" / "stage1_report.json")
    bin12 = load(root / "outputs_binary" / "stage1_report.json")
    bin19 = load(root / "outputs_binary_19" / "stage1_report.json")
    four12_s2 = stage2_4class(load(root / "outputs" / "stage2_report.json"))
    four19_s2 = stage2_4class(load(root / "outputs_compression" / "stage2_report.json"))

    m = {
        "4-Class (12)": {**danger_metrics(four12, "Panic"), "s2": four12_s2},
        "4-Class (19)": {**danger_metrics(four19, "Panic"), "s2": four19_s2},
        "Binary (12)": {**danger_metrics(bin12, "Stampede Risk"), "s2": stage2_binary(bin12)},
        "Binary (19)": {**danger_metrics(bin19, "Stampede Risk"), "s2": stage2_binary(bin19)},
    }
    cols = list(m.keys())

    L = []
    L.append("# Final model comparison — 4-Class vs Binary, 12 vs 19 features\n")
    L.append("Identical leakage-safe split & 5-fold CV. Danger class = Panic "
             "(4-class) / Stampede Risk (binary). Clip-level, CV out-of-fold.\n")

    def row(name, key, fmt):
        return "| " + name + " | " + " | ".join(fmt(m[c][key]) for c in cols) + " |"

    L.append("| Metric | " + " | ".join(cols) + " |")
    L.append("|" + "---|" * (len(cols) + 1))
    L.append(row("**Danger-class F1**", "f1", f3))
    L.append(row("**Danger-class recall**", "recall", f3))
    L.append(row("Danger-class precision", "precision", f3))
    L.append(row("CV macro-F1 (OOF)", "macro_f1", f3))
    L.append(row("CV accuracy (OOF)", "accuracy", pct))
    L.append("| Stage 2 detect (spec 0.8) | " +
             " | ".join(pct(m[c]["s2"]["spec"]) for c in cols) + " |")
    L.append("| Stage 2 detect (calibrated) | " +
             " | ".join(pct(m[c]["s2"]["cal"]) for c in cols) + " |")

    # Binary 12 vs 19 focus.
    b12, b19 = m["Binary (12)"], m["Binary (19)"]
    L.append("\n## Binary: 12 vs 19 features (the requested delta)\n")
    L.append("| Metric | Binary (12) | Binary (19) | Δ |")
    L.append("|---|---|---|---|")
    for lbl, k, fmt in [("Stampede Risk F1", "f1", f3),
                        ("Stampede Risk recall", "recall", f3),
                        ("Stampede Risk precision", "precision", f3),
                        ("CV macro-F1 (OOF)", "macro_f1", f3),
                        ("CV accuracy (OOF)", "accuracy", pct)]:
        L.append(f"| {lbl} | {fmt(b12[k])} | {fmt(b19[k])} | {b19[k]-b12[k]:+.3f} |")
    L.append(f"| Stage 2 detect (calibrated) | {pct(b12['s2']['cal'])} | "
             f"{pct(b19['s2']['cal'])} | {(b19['s2']['cal']-b12['s2']['cal'])*100:+.0f} pp |")

    # Always-Safe baseline for binary.
    bin_oof = bin19["cross_validation"]["oof"]["per_class"]
    supp = {k: v.get("support", 0) for k, v in bin_oof.items() if isinstance(v, dict)}
    baseline = supp.get("Safe", 0) / max(1, supp.get("Safe", 0) + supp.get("Stampede Risk", 0))

    # Verdict.
    best_recall = max(cols, key=lambda c: m[c]["recall"])
    best_f1 = max(cols, key=lambda c: m[c]["f1"])
    L.append("\n## Verdict\n")
    L.append(f"- **Best danger-class recall** (fewest missed danger events): "
             f"**{best_recall}** ({f3(m[best_recall]['recall'])}).")
    L.append(f"- **Best danger-class F1**: **{best_f1}** ({f3(m[best_f1]['f1'])}).")
    L.append(f"- Binary accuracy must beat the *always-Safe* baseline "
             f"({pct(baseline)}); Binary(19) OOF accuracy is {pct(b19['accuracy'])} "
             f"({(b19['accuracy']-baseline)*100:+.1f} pp).")
    L.append("")

    md = "\n".join(L)
    (root / "comparison_final.md").write_text(md)
    (root / "comparison_final.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "s2"} | {"stage2": v["s2"]}
         for k, v in m.items()}, indent=2))
    print(md)


if __name__ == "__main__":
    main()
