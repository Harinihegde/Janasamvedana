#!/usr/bin/env python3
"""Crush-focused retrain: does making Panic = crush-only improve the model?

Same 21 features, same RF architecture, LOVO cross-validation. Two schemes:
  * FULL  - Panic = all original 281 clips (baseline, as before)
  * CLEAN - Panic = crush clips only (flight WA0017/0020/0022/0023-0924 and
            the videos/clips the user removed are dropped from Panic entirely;
            No-Panic / Normal / Crowdy unchanged)

Reports, for each: per-class precision/recall/F1, Crowdy<->Panic confusion,
accuracy, macro-F1. Plus an ISOLATION metric: recall on the SAME crush clips
under FULL-training vs CLEAN-training (holds the eval set fixed, so it measures
the training-label effect, not just "we dropped the hard clips").
Output -> outputs_improved/crush_lovo_report.{json,md}
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from stampede.classifier import train
from stampede.config import CLASSES, FEATURE_COLUMNS

FEATS = Path("outputs_compression2/frame_features.csv")
KEPT = Path("kept_panic.txt")
OUT = Path("outputs_improved")
FLIGHT_VIDEOS = {"VID-20250924-WA0017", "VID-20250924-WA0020",
                 "VID-20250924-WA0022", "VID-20250924-WA0023"}


def lovo(df):
    """4-class leave-one-video-out; return clip-level OOF (path, actual, pred)."""
    vids = df.video_id.unique().tolist()
    parts = []
    t0 = time.time()
    for n, v in enumerate(vids, 1):
        tr = df[df.video_id != v]
        te = df[df.video_id == v]
        model = train(tr, feature_cols=FEATURE_COLUMNS)
        te = te.assign(pred=model.predict(te))
        agg = (te.groupby("path")
               .agg(video_id=("video_id", "first"), actual=("label", "first"),
                    pred=("pred", lambda s: s.value_counts().idxmax())).reset_index())
        parts.append(agg)
        if n % 15 == 0 or n == len(vids):
            print(f"  LOVO {n}/{len(vids)} ({time.time()-t0:.0f}s)", flush=True)
    return pd.concat(parts, ignore_index=True)


def metrics(oof):
    y, p = oof.actual.values, oof.pred.values
    rep = classification_report(y, p, labels=list(CLASSES), output_dict=True,
                                zero_division=0)
    cm = confusion_matrix(y, p, labels=list(CLASSES))
    ci = {c: i for i, c in enumerate(CLASSES)}
    cr, pa = ci["Crowdy"], ci["Panic"]
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "per_class": {c: {k: rep[c][k] for k in ("precision", "recall", "f1-score", "support")}
                      for c in CLASSES},
        "crowdy_panic_block": {
            "Panic->Panic": int(cm[pa, pa]), "Panic->Crowdy": int(cm[pa, cr]),
            "Crowdy->Panic": int(cm[cr, pa]), "Crowdy->Crowdy": int(cm[cr, cr])},
        "confusion": cm.tolist(),
    }


def main():
    df = pd.read_csv(FEATS)
    df["clip"] = df.path.apply(lambda p: Path(p).name)
    kept = set(re.findall(r"VID-\d+-WA\d+_clip_\d+\.mp4", KEPT.read_text()))

    is_panic = df.label == "Panic"
    is_flight = df.video_id.isin(FLIGHT_VIDEOS)
    is_kept = df["clip"].isin(kept)
    # crush Panic clips = kept, panic, not a flight video
    crush_mask = is_panic & is_kept & ~is_flight
    crush_paths = set(df.loc[crush_mask, "path"])

    # composition report
    pan = df[is_panic].drop_duplicates("path")
    comp = {
        "panic_clips_total": int(pan.shape[0]),
        "crush_clips": int(crush_mask.drop_duplicates().sum() if False else
                           df.loc[crush_mask].path.nunique()),
        "flight_clips_dropped": int(df.loc[is_panic & is_flight].path.nunique()),
        "removed_clips": int(df.loc[is_panic & ~is_kept & ~is_flight].path.nunique()),
        "crush_videos": int(df.loc[crush_mask].video_id.nunique()),
    }
    print("Composition:", json.dumps(comp), flush=True)
    print("Crush videos:", sorted(df.loc[crush_mask].video_id.unique()), flush=True)

    # datasets
    full_df = df.copy()                                   # Panic = all 281
    clean_df = df[(df.label != "Panic") | crush_mask].copy()  # Panic = crush only

    print("\n=== FULL (Panic = all 281) LOVO ===", flush=True)
    full_oof = lovo(full_df)
    print("=== CLEAN (Panic = crush only) LOVO ===", flush=True)
    clean_oof = lovo(clean_df)

    full_m, clean_m = metrics(full_oof), metrics(clean_oof)

    # isolation: recall on the SAME crush clips, FULL-training vs CLEAN-training
    full_crush = full_oof[full_oof.path.isin(crush_paths)]
    clean_crush = clean_oof[clean_oof.path.isin(crush_paths)]
    iso = {
        "crush_clips_evaluated": int(len(full_crush)),
        "crush_recall_FULL_training": float((full_crush.pred == "Panic").mean()),
        "crush_recall_CLEAN_training": float((clean_crush.pred == "Panic").mean()),
    }

    report = {"composition": comp, "full": full_m, "clean": clean_m, "isolation": iso}
    (OUT / "crush_lovo_report.json").write_text(json.dumps(report, indent=2))

    def pc(m, c, k): return m["per_class"][c][k]
    L = ["# Crush-focused retrain (4-class LOVO, 21 features)\n",
         f"Composition: {comp['crush_clips']} crush clips ({comp['crush_videos']} videos) kept as Panic; "
         f"{comp['flight_clips_dropped']} flight + {comp['removed_clips']} removed dropped.\n",
         "## Per-class (LOVO OOF) — FULL Panic vs CLEAN (crush-only) Panic\n",
         "| Metric | FULL | CLEAN | Δ |", "|---|---|---|---|"]
    for c in ["Crowdy", "Panic"]:
        for k in ["precision", "recall", "f1-score"]:
            fv, cv = pc(full_m, c, k), pc(clean_m, c, k)
            L.append(f"| {c} {k} | {fv:.3f} | {cv:.3f} | {cv-fv:+.3f} |")
    for k in ["accuracy", "macro_f1"]:
        L.append(f"| overall {k} | {full_m[k]:.3f} | {clean_m[k]:.3f} | {clean_m[k]-full_m[k]:+.3f} |")
    L += ["", "## Crowdy↔Panic confusion (LOVO OOF)\n",
          f"FULL : {full_m['crowdy_panic_block']}", "",
          f"CLEAN: {clean_m['crowdy_panic_block']}", "",
          "## Isolation — crush-clip recall on the SAME clips (training effect only)\n",
          f"| | crush recall |", "|---|---|",
          f"| trained with FULL Panic (flight incl.) | {iso['crush_recall_FULL_training']:.3f} |",
          f"| trained with CLEAN Panic (crush only)  | {iso['crush_recall_CLEAN_training']:.3f} |",
          f"| Δ | {iso['crush_recall_CLEAN_training']-iso['crush_recall_FULL_training']:+.3f} |"]
    (OUT / "crush_lovo_report.md").write_text("\n".join(L))
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
