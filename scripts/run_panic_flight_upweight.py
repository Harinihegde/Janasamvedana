#!/usr/bin/env python3
"""Compare the 4-class Stage 1 model with vs. without Panic-flight upweighting.

Motivation: data/panic_sublabels.csv splits kept Panic clips into crush (164
clips / 17 videos) vs. flight (39 clips / 4 videos) via manual audit. Only 4
source videos carry the flight sub-label - too few to evaluate as its own
leakage-safe class (a video-grouped CV fold can't meaningfully spread 4
videos across folds) - so instead of adding a class,
stampede.sublabels.panic_flight_sample_weight upweights flight frames within
Panic training (via the sample_weight plumbing added to
stampede/classifier.py). This script checks whether that changes anything,
using the same leakage-safe 5-fold pooled out-of-fold (OOF) protocol as the
rest of the project's comparisons (e.g. results/csrnet_comparison.md).

Outputs -> results/panic_flight_upweight.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from stampede.classifier import train
from stampede.config import FEATURE_COLUMNS
from stampede.dataset import stratified_group_folds
from stampede.evaluate import clip_vote
from stampede.sublabels import load_panic_sublabels, panic_flight_sample_weight

FEATURES_CSV = Path("outputs/frame_features.csv")
OUT_JSON = Path("results/panic_flight_upweight.json")
OUT_MD = Path("results/panic_flight_upweight.md")
N_SPLITS = 5


def pooled_oof(feats: pd.DataFrame, upweight: bool, n_splits: int = N_SPLITS) -> pd.DataFrame:
    """Leakage-safe pooled out-of-fold clip-level predictions (5-fold)."""
    clips = feats[["path", "video_id", "label"]].drop_duplicates("path").reset_index(drop=True)
    folds = stratified_group_folds(clips, n_splits=n_splits)
    parts = []
    for fi, (tr_idx, te_idx) in enumerate(folds, 1):
        tr_paths = set(clips.loc[tr_idx, "path"])
        te_paths = set(clips.loc[te_idx, "path"])
        tr = feats[feats.path.isin(tr_paths)]
        te = feats[feats.path.isin(te_paths)]
        sw = panic_flight_sample_weight(tr) if upweight else None
        model = train(tr, sample_weight=sw)
        pred = model.predict(te)
        parts.append(clip_vote(te, pred))
        tag = "upweighted" if upweight else "baseline"
        print(f"  [{tag}] fold {fi}/{len(folds)} done", flush=True)
    return pd.concat(parts, ignore_index=True)


def summarize(oof: pd.DataFrame) -> dict:
    acc = accuracy_score(oof.actual, oof.predicted)
    macro_f1 = f1_score(oof.actual, oof.predicted, average="macro", zero_division=0)
    report = classification_report(oof.actual, oof.predicted, output_dict=True, zero_division=0)
    panic = report.get("Panic", {})
    return dict(
        accuracy=float(acc),
        macro_f1=float(macro_f1),
        panic_precision=float(panic.get("precision", 0.0)),
        panic_recall=float(panic.get("recall", 0.0)),
        panic_f1=float(panic.get("f1-score", 0.0)),
        panic_support=int(panic.get("support", 0)),
    )


def sublabel_breakdown(oof: pd.DataFrame) -> dict:
    """Descriptive-only: Panic recall split by crush/flight sub-label.

    NOT a validated per-class metric - only 4 source videos carry the flight
    sub-label, so however they land across the 5 folds, this is a small and
    possibly unbalanced sample. Reported for transparency, not as a claim.
    """
    sublabels = load_panic_sublabels()
    panic = oof[oof.actual == "Panic"].copy()
    panic["clip_name"] = panic["path"].map(lambda p: Path(p).name)
    panic["sublabel"] = panic["clip_name"].map(sublabels)
    out = {}
    for sub in ("panic_crush", "panic_flight"):
        g = panic[panic.sublabel == sub]
        if len(g) == 0:
            continue
        out[sub] = dict(n_clips=int(len(g)), recall=float((g.predicted == "Panic").mean()))
    return out


def write_report(baseline: dict, upweighted: dict, baseline_sub: dict, upweighted_sub: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            dict(
                baseline=baseline,
                upweighted=upweighted,
                baseline_sublabel_breakdown=baseline_sub,
                upweighted_sublabel_breakdown=upweighted_sub,
            ),
            indent=2,
        )
    )

    def row(name: str, b: float, u: float) -> str:
        return f"| {name} | {b:.3f} | {u:.3f} | {u - b:+.3f} |"

    lines = [
        "# Panic crush/flight upweighting (Stage 1, 4-class)",
        "",
        "Pooled 5-fold leakage-safe out-of-fold clip-level results (n=1,330 "
        "clips), comparing the baseline RandomForest against the same model "
        "trained with Panic-flight frames upweighted to match Panic-crush's "
        "aggregate training weight (see `stampede/sublabels.py`).",
        "",
        "| Metric | Baseline | Upweighted | Delta |",
        "|---|---|---|---|",
        row("CV clip accuracy", baseline["accuracy"], upweighted["accuracy"]),
        row("CV clip macro-F1", baseline["macro_f1"], upweighted["macro_f1"]),
        row("Panic precision", baseline["panic_precision"], upweighted["panic_precision"]),
        row("Panic recall", baseline["panic_recall"], upweighted["panic_recall"]),
        row("Panic F1", baseline["panic_f1"], upweighted["panic_f1"]),
        "",
        "## Descriptive-only: Panic recall by crush/flight sub-label",
        "",
        "**Not a validated per-subclass metric.** Only 4 of 23 Panic source "
        "videos carry a confirmed flight sub-label (39 clips vs. 164 crush "
        "clips) - far too few to trust a video-grouped fold's flight recall "
        "on its own. Shown for transparency only, not as a claim.",
        "",
        "| Sub-label | n clips | Baseline recall | Upweighted recall |",
        "|---|---|---|---|",
    ]
    for sub in ("panic_crush", "panic_flight"):
        b, u = baseline_sub.get(sub), upweighted_sub.get(sub)
        if b is None or u is None:
            continue
        lines.append(f"| {sub} | {b['n_clips']} | {b['recall']:.0%} | {u['recall']:.0%} |")
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    feats = pd.read_csv(FEATURES_CSV)
    # This cached CSV predates the CSRNet feature columns; per config.py's
    # documented schema contract, a heuristic-only extraction should carry
    # them as 0.0 rather than omit them.
    for col in FEATURE_COLUMNS:
        if col not in feats.columns:
            feats[col] = 0.0
    print(f"Loaded {len(feats)} frames from {FEATURES_CSV}", flush=True)

    print("Running baseline (no upweighting)...", flush=True)
    oof_base = pooled_oof(feats, upweight=False)
    print("Running upweighted (Panic-flight upweighted within training)...", flush=True)
    oof_up = pooled_oof(feats, upweight=True)

    baseline = summarize(oof_base)
    upweighted = summarize(oof_up)
    baseline_sub = sublabel_breakdown(oof_base)
    upweighted_sub = sublabel_breakdown(oof_up)

    write_report(baseline, upweighted, baseline_sub, upweighted_sub)
    print(json.dumps(dict(baseline=baseline, upweighted=upweighted), indent=2))
    print(f"\nWrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
