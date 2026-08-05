#!/usr/bin/env python3
"""Before/after: does applying data/kept_panic.txt's label-audit allowlist
change Stage 1 or Stage 2 results?

Context: data/kept_panic.txt is a manually-curated allowlist (data/README.md)
removing blank outro cards, news-desk interviews, and other non-panic
contamination from the raw 281-clip Panic class - but until now it was never
read by any pipeline code (stampede/dataset.py's inventory() and
filter_kept_panic() fix that). One flagged example
(VID-20251025-WA0003_clip_003.mp4, audited as "OUTRO CARD ... no people") was
confirmed to have 15/30 zero-person frames while still labeled Panic in
training. Diagnosing Stage 2's false positives (see results/
stage2_baseline_margin_sweep.md) found several trip on near-blank/black
frames or stock-footage watermark cards in Normal/No Panic videos - consistent
with the model having learned "near-zero features -> Panic" from this
contamination.

This compares Stage 1 (5-fold leakage-safe CV) and Stage 2 (false-positive
rate on the same 8 known-bad videos) before vs. after applying the filter.

Outputs -> results/kept_panic_filter_comparison.{json,md}
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from stampede.classifier import train
from stampede.config import CONFIRM_FRAMES, FEATURE_COLUMNS
from stampede.dataset import filter_kept_panic, stratified_group_folds
from stampede.escalation import clip_timelines, evaluate_false_positives

FEATURES_CSV = Path("outputs/frame_features.csv")
OUT_JSON = Path("results/kept_panic_filter_comparison.json")
OUT_MD = Path("results/kept_panic_filter_comparison.md")
N_SPLITS = 5

# The 8 stable (No Panic/Normal) videos already identified as false-positive
# sources at confirm_frames=3 (see results/stage2_baseline_margin_sweep.md).
KNOWN_FP_VIDEOS = {
    "VID-20250924-WA0036", "VID-20251025-WA0006", "VID-20250924-WA0030",
    "VID-20250924-WA0034", "VID-20250924-WA0032", "VID-20250924-WA0029",
    "VID-20250924-WA0035", "VID-20250924-WA0031",
}


def pooled_oof(feats: pd.DataFrame, n_splits: int = N_SPLITS) -> pd.DataFrame:
    """Leakage-safe pooled out-of-fold predictions + continuous risk score."""
    clips = feats[["path", "video_id", "label"]].drop_duplicates("path").reset_index(drop=True)
    folds = stratified_group_folds(clips, n_splits=n_splits)
    parts = []
    for fi, (tr_idx, te_idx) in enumerate(folds, 1):
        tr_paths = set(clips.loc[tr_idx, "path"])
        te_paths = set(clips.loc[te_idx, "path"])
        tr = feats[feats.path.isin(tr_paths)]
        te = feats[feats.path.isin(te_paths)].copy()
        model = train(tr, feature_cols=FEATURE_COLUMNS)
        te["predicted"] = model.predict(te)
        te["risk"] = model.risk_score(te)
        parts.append(te[["path", "video_id", "label", "frame_index", "predicted", "risk"]])
        print(f"  fold {fi}/{len(folds)} done ({len(te)} frames)", flush=True)
    return pd.concat(parts, ignore_index=True)


def clip_vote_summary(oof: pd.DataFrame) -> dict:
    tmp = oof[["path", "video_id", "label", "predicted"]].copy()
    votes = (
        tmp.groupby("path")
        .agg(video_id=("video_id", "first"), actual=("label", "first"),
             predicted=("predicted", lambda s: s.value_counts().idxmax()))
    )
    acc = accuracy_score(votes.actual, votes.predicted)
    macro_f1 = f1_score(votes.actual, votes.predicted, average="macro", zero_division=0)
    report = classification_report(votes.actual, votes.predicted, output_dict=True, zero_division=0)
    panic = report.get("Panic", {})
    return dict(
        accuracy=float(acc),
        macro_f1=float(macro_f1),
        panic_precision=float(panic.get("precision", 0.0)),
        panic_recall=float(panic.get("recall", 0.0)),
        panic_f1=float(panic.get("f1-score", 0.0)),
        panic_support=int(panic.get("support", 0)),
    )


def fp_recheck(oof: pd.DataFrame) -> dict:
    """Re-run the Stage 2 false-positive check, but only on the 8 videos
    already known to trip alerts before this fix, to see which (if any) stop."""
    timelines = clip_timelines(oof)
    still_alerting = []
    for vid in KNOWN_FP_VIDEOS:
        if vid not in timelines:
            continue
        g = timelines[vid]
        from stampede.escalation import detect_alerts
        res = detect_alerts(g.risk.to_numpy(), confirm_frames=CONFIRM_FRAMES)
        if res["alerts"]:
            still_alerting.append(vid)
    # Also the full official false-positive rate across all stable videos.
    fp = evaluate_false_positives(timelines, confirm_frames=CONFIRM_FRAMES)
    return dict(
        known_fp_videos_checked=len(KNOWN_FP_VIDEOS & timelines.keys()),
        known_fp_videos_still_alerting=still_alerting,
        n_known_fp_fixed=len(KNOWN_FP_VIDEOS & timelines.keys()) - len(still_alerting),
        overall_false_positive_rate=fp["false_positive_rate"],
        overall_n_stable_videos=fp["n_stable_videos"],
    )


def write_report(before: dict, after: dict, fp_before: dict, fp_after: dict) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(dict(before=before, after=after, fp_before=fp_before, fp_after=fp_after), indent=2)
    )

    def row(name: str, b: float, a: float) -> str:
        return f"| {name} | {b:.3f} | {a:.3f} | {a - b:+.3f} |"

    lines = [
        "# kept_panic.txt filter: before/after (Stage 1 + Stage 2)",
        "",
        "Pooled 5-fold leakage-safe out-of-fold results, comparing the raw "
        "281-clip Panic class against the label-audit-filtered set "
        "(`data/kept_panic.txt`, dropping confirmed contamination clips).",
        "",
        "## Stage 1 (4-class classifier)",
        "",
        "| Metric | Before (raw 281 Panic) | After (kept_panic filtered) | Delta |",
        "|---|---|---|---|",
        row("CV clip accuracy", before["accuracy"], after["accuracy"]),
        row("CV clip macro-F1", before["macro_f1"], after["macro_f1"]),
        row("Panic precision", before["panic_precision"], after["panic_precision"]),
        row("Panic recall", before["panic_recall"], after["panic_recall"]),
        row("Panic F1", before["panic_f1"], after["panic_f1"]),
        f"| Panic support (n clips) | {before['panic_support']} | {after['panic_support']} | "
        f"{after['panic_support'] - before['panic_support']:+d} |",
        "",
        "## Stage 2: the 8 known false-positive videos (confirm_frames=3)",
        "",
        f"- Before: {len(fp_before['known_fp_videos_still_alerting'])}/"
        f"{fp_before['known_fp_videos_checked']} still alerting",
        f"- After: {len(fp_after['known_fp_videos_still_alerting'])}/"
        f"{fp_after['known_fp_videos_checked']} still alerting",
        f"- Fixed by this change: {fp_after['n_known_fp_fixed'] - fp_before['n_known_fp_fixed']} "
        "(relative to the 0 fixed by definition before)",
        f"- Still alerting after the fix: {fp_after['known_fp_videos_still_alerting']}",
        "",
        f"- Overall stable-video false-positive rate: {fp_before['overall_false_positive_rate']:.0%} "
        f"-> {fp_after['overall_false_positive_rate']:.0%} "
        f"(n={fp_after['overall_n_stable_videos']} stable videos)",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    feats = pd.read_csv(FEATURES_CSV)
    for col in FEATURE_COLUMNS:
        if col not in feats.columns:
            feats[col] = 0.0
    print(f"Loaded {len(feats)} frames from {FEATURES_CSV}", flush=True)

    print("Running BEFORE (raw, unfiltered Panic class)...", flush=True)
    oof_before = pooled_oof(feats)
    before = clip_vote_summary(oof_before)
    fp_before = fp_recheck(oof_before)

    print("Running AFTER (kept_panic.txt filter applied)...", flush=True)
    feats_filtered = filter_kept_panic(feats)
    oof_after = pooled_oof(feats_filtered)
    after = clip_vote_summary(oof_after)
    fp_after = fp_recheck(oof_after)

    write_report(before, after, fp_before, fp_after)
    print(json.dumps(dict(before=before, after=after), indent=2))
    print(json.dumps(dict(fp_before=fp_before, fp_after=fp_after), indent=2))
    print(f"\nWrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
