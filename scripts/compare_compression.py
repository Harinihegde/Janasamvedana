#!/usr/bin/env python3
"""Compare the base 12-feature model against the +compression 19-feature model.

Both are 4-class RandomForests trained on the identical leakage-safe split and
CV folds; the only difference is the 7 added compression/occlusion features.
Reports the metrics that matter for the Panic-vs-Crowdy problem and shows where
the compression features rank by importance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

COMPRESSION = {
    "bbox_overlap_ratio", "bbox_overlap_trend", "detection_confidence_mean",
    "detection_confidence_std", "bbox_area_variance", "bbox_area_mean",
    "spatial_density_mismatch",
}


def load(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def pct(x):
    return f"{x*100:.1f}%"


def f3(x):
    return f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=Path("outputs"))
    ap.add_argument("--comp", type=Path, default=Path("outputs_compression"))
    a = ap.parse_args()

    base = load(a.base / "stage1_report.json")
    comp = load(a.comp / "stage1_report.json")
    base_s2 = load(a.base / "stage2_report.json")
    comp_s2 = load(a.comp / "stage2_report.json")

    b_oof = base["cross_validation"]["oof"]
    c_oof = comp["cross_validation"]["oof"]
    b_cv = base["cross_validation"]
    c_cv = comp["cross_validation"]

    def panic(oof, k="Panic"):
        return oof["per_class"].get(k, {})

    bp, cp = panic(b_oof), panic(c_oof)

    L = []
    L.append("# Base (12 features) vs +Compression (19 features)\n")
    L.append("4-class RandomForest, identical leakage-safe split & 5-fold "
             "StratifiedGroupKFold CV. Compression adds 7 occlusion/crush "
             "features. Metrics are clip-level, from CV out-of-fold predictions.\n")

    L.append("## Headline\n")
    L.append("| Metric | 12 Features | 19 Features (+Compression) | Δ |")
    L.append("|---|---|---|---|")
    rows = [
        ("Panic F1", bp.get("f1-score", 0), cp.get("f1-score", 0), False),
        ("Panic recall", bp.get("recall", 0), cp.get("recall", 0), False),
        ("Panic precision", bp.get("precision", 0), cp.get("precision", 0), False),
        ("CV macro-F1 (OOF)", b_oof["macro_f1"], c_oof["macro_f1"], False),
        ("CV accuracy (OOF)", b_oof["accuracy"], c_oof["accuracy"], True),
        ("CV macro-F1 (fold mean)", b_cv["clip_macro_f1_mean"],
         c_cv["clip_macro_f1_mean"], False),
    ]
    for name, bv, cv, is_pct in rows:
        fmt = pct if is_pct else f3
        d = cv - bv
        L.append(f"| {name} | {fmt(bv)} | {fmt(cv)} | {d:+.3f} |")

    L.append("\n## Per-class F1 (CV out-of-fold)\n")
    L.append("| Class | 12 Features | 19 Features | Δ |")
    L.append("|---|---|---|---|")
    for cls in ["No Panic", "Normal", "Crowdy", "Panic"]:
        bf = b_oof["per_class"].get(cls, {}).get("f1-score", 0)
        cf = c_oof["per_class"].get(cls, {}).get("f1-score", 0)
        L.append(f"| {cls} | {f3(bf)} | {f3(cf)} | {cf-bf:+.3f} |")

    # Crowdy<->Panic confusion (the target of the compression features).
    def cell(oof, actual, pred):
        labs = oof["confusion_labels"]
        cm = oof["confusion_matrix"]
        return cm[labs.index(actual)][labs.index(pred)]

    L.append("\n## Crowdy↔Panic confusion (CV OOF) — the pair we targeted\n")
    L.append("| | 12 Features | 19 Features |")
    L.append("|---|---|---|")
    L.append(f"| Panic→Panic (correct) | {cell(b_oof,'Panic','Panic')} | {cell(c_oof,'Panic','Panic')} |")
    L.append(f"| Panic→Crowdy (miss) | {cell(b_oof,'Panic','Crowdy')} | {cell(c_oof,'Panic','Crowdy')} |")
    L.append(f"| Crowdy→Panic (false) | {cell(b_oof,'Crowdy','Panic')} | {cell(c_oof,'Crowdy','Panic')} |")
    L.append(f"| Crowdy→Crowdy (correct) | {cell(b_oof,'Crowdy','Crowdy')} | {cell(c_oof,'Crowdy','Crowdy')} |")

    # Feature importances: where do compression features land?
    imp = comp["feature_importances"]
    ranked = list(imp.items())  # already sorted desc
    L.append("\n## Feature importances (19-feature model) — compression features **bold**\n")
    L.append("| Rank | Feature | Importance |")
    L.append("|---|---|---|")
    comp_ranks = []
    for i, (name, val) in enumerate(ranked, 1):
        tag = f"**{name}**" if name in COMPRESSION else name
        if name in COMPRESSION:
            comp_ranks.append(i)
        L.append(f"| {i} | {tag} | {val:.3f} |")
    in_top10 = [r for r in comp_ranks if r <= 10]
    L.append(f"\nCompression features in the top 10: **{len(in_top10)} of 7** "
             f"(ranks {sorted(comp_ranks)}).\n")

    # Stage 2.
    def cp_det(s2, key):
        return s2[key]["transition_detection"]["crowdy_to_panic"]["detection_rate"]

    L.append("## Stage 2 — Crowdy→Panic detection (held-out test)\n")
    L.append("| Threshold | 12 Features | 19 Features |")
    L.append("|---|---|---|")
    L.append(f"| Spec edge 0.8 | {pct(cp_det(base_s2,'spec_thresholds'))} | "
             f"{pct(cp_det(comp_s2,'spec_thresholds'))} |")
    L.append(f"| Calibrated | {pct(cp_det(base_s2,'calibrated_thresholds'))} | "
             f"{pct(cp_det(comp_s2,'calibrated_thresholds'))} |")

    # Verdict.
    d_panic = cp.get("f1-score", 0) - bp.get("f1-score", 0)
    d_macro = c_oof["macro_f1"] - b_oof["macro_f1"]
    better = "improve" if d_panic > 0 else "do not improve"
    L.append("\n## Verdict\n")
    L.append(
        f"Compression features **{better}** Panic F1 "
        f"({f3(bp.get('f1-score',0))} → {f3(cp.get('f1-score',0))}, "
        f"Δ {d_panic:+.3f}) and move CV macro-F1 by {d_macro:+.3f}. "
        f"{len(in_top10)}/7 compression features rank in the top 10 by "
        "importance.\n")

    md = "\n".join(L)
    (a.base.parent / "comparison_compression.md").write_text(md)
    (a.base.parent / "comparison_compression.json").write_text(json.dumps({
        "panic_f1": {"base": bp.get("f1-score"), "compression": cp.get("f1-score")},
        "panic_recall": {"base": bp.get("recall"), "compression": cp.get("recall")},
        "cv_macro_f1_oof": {"base": b_oof["macro_f1"], "compression": c_oof["macro_f1"]},
        "compression_feature_ranks": sorted(comp_ranks),
        "compression_in_top10": len(in_top10),
    }, indent=2))
    print(md)


if __name__ == "__main__":
    main()
