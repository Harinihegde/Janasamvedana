#!/usr/bin/env python3
"""Stage 0: build the leakage-safe manifest and extract per-frame features.

Usage::

    python extract_features.py --dataset /path/to/crowd_panic \\
        --weights /path/to/best_combined.pt --output outputs

Outputs (under ``--output``):
  * ``manifest.csv``            - one row per clip, with video_id and split
  * ``video_properties_by_class.csv``
  * ``frame_features.csv``      - one row per sampled frame with all features
Re-running with an existing ``frame_features.csv`` is skipped unless
``--overwrite`` is passed, so extraction is a one-time cost.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from stampede.dataset import grouped_split, inventory
from stampede.detector import Detector
from stampede.features import frame_features


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--output", type=Path, default=Path("outputs"))
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit-per-class", type=int, default=0,
                    help="cap clips per class for a fast smoke test (0 = all)")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)

    # 1. Inventory + leakage-safe split (grouped by video_id).
    manifest = inventory(a.dataset)
    manifest = grouped_split(manifest)
    if a.limit_per_class:
        manifest = (
            manifest.groupby("label").head(a.limit_per_class).reset_index(drop=True)
        )
    manifest.to_csv(a.output / "manifest.csv", index=False)
    manifest.groupby("label")[
        ["fps", "frames", "width", "height", "duration_seconds"]
    ].agg(["count", "min", "median", "max"]).to_csv(
        a.output / "video_properties_by_class.csv"
    )
    print(
        "Split (unique videos): "
        + ", ".join(
            f"{lab}: {manifest[manifest.label == lab].video_id.nunique()} vids "
            f"({(manifest[(manifest.label == lab) & (manifest.split == 'test')].video_id.nunique())} test)"
            for lab in manifest.label.unique()
        ),
        flush=True,
    )

    # 2. Per-frame feature extraction.
    fp = a.output / "frame_features.csv"
    if fp.exists() and not a.overwrite:
        print(f"{fp} exists; skipping extraction (use --overwrite to redo).")
        return

    detector = Detector(a.weights, imgsz=a.imgsz)
    frames = []
    n = len(manifest)
    t0 = time.time()
    for j, row in manifest.reset_index(drop=True).iterrows():
        feats = frame_features(row.path, detector)
        if feats.empty:
            print(f"  [warn] no frames read from {row.path}")
            continue
        feats.insert(0, "split", row.split)
        feats.insert(0, "video_id", row.video_id)
        feats.insert(0, "label", row.label)
        feats.insert(0, "path", row.path)
        frames.append(feats)
        elapsed = time.time() - t0
        eta = elapsed / (j + 1) * (n - j - 1)
        print(
            f"features: {j + 1}/{n}  ({row.label})  "
            f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
            flush=True,
        )
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(fp, index=False)
    print(f"Wrote {len(out)} frame rows -> {fp}")


if __name__ == "__main__":
    main()
