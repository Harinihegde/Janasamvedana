#!/usr/bin/env python3
"""Dump montages for all 86 leaked Panic clips with an auto-triage flag.

Categories (heuristic, for fast human review - not ground truth):
  ARCHIVAL_BW    near-grayscale footage (different visual domain / stock)
  STATIC_WIDE    colour, very low motion (fixed-cam wide/aerial; distant
                 crowd often under-counted by YOLO -> looks low-density)
  MOTION_RUNNING colour + high optical flow (people moving/scattering)
  AMBIGUOUS      mid motion - needs eyeballing

Signals (no new inference): mean HSV saturation (grayscale test) + per-clip
optical-flow magnitude & directional coherence from the extracted features.
Montages are named ``<CATEGORY>__<clip>.png`` so the folder self-sorts.
Output -> outputs_improved/panic_lowdensity_all/
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

OUT = Path("outputs_improved/panic_lowdensity_all")
LEAKED_CSV = Path("outputs_improved/panic_lowdensity/leaked_panic_clips.csv")
FEATURES_CSV = Path("outputs_compression2/frame_features.csv")
SAT_BW = 10.0  # mean saturation below this => genuinely grayscale/archival
              # (washed-out colour wide-shots sit ~20-30 and are NOT B&W)


def sample_frames(path: str, fracs=(0.2, 0.5, 0.8)):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for f in fracs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * f))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    return frames


def montage(frames, label, dest):
    h = 260
    imgs = [cv2.resize(f, (int(f.shape[1] * h / f.shape[0]), h)) for f in frames]
    strip = cv2.hconcat(imgs)
    bar = np.full((28, strip.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(dest), cv2.vconcat([bar, strip]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    leaked = pd.read_csv(LEAKED_CSV)
    leaked_names = set(leaked['clip'])

    feats = pd.read_csv(FEATURES_CSV)
    feats["clip"] = feats.path.apply(lambda p: Path(p).name)
    agg = (feats[feats['clip'].isin(leaked_names)]
           .groupby("clip")
           .agg(path=("path", "first"),
                count=("person_count", "mean"),
                flow=("flow_mag_mean", "mean"),
                coherence=("direction_consistency", "mean")).reset_index())
    agg = agg.merge(leaked[["clip", "predicted"]], on="clip")

    # Flow tertiles (colour clips only get static/motion split).
    flo_lo, flo_hi = agg.flow.quantile([0.33, 0.66])

    rows = []
    for _, r in agg.iterrows():
        frames = sample_frames(r.path)
        if not frames:
            continue
        sat = float(np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2HSV)[..., 1].mean()
                             for f in frames]))
        if sat < SAT_BW:
            cat = "ARCHIVAL_BW"
        elif r.flow < flo_lo:
            cat = "STATIC_WIDE"
        elif r.flow > flo_hi:
            cat = "MOTION_RUNNING"
        else:
            cat = "AMBIGUOUS"
        label = (f"[{cat}] PANIC->{r.predicted} | cnt={r['count']:.0f} "
                 f"flow={r.flow:.4f} coh={r.coherence:.2f} sat={sat:.0f} | {r['clip']}")
        montage(frames, label, OUT / f"{cat}__{Path(r['clip']).stem}.png")
        rows.append(dict(clip=r['clip'], category=cat, predicted=r.predicted,
                         count=round(r['count'], 1), flow=round(r.flow, 4),
                         coherence=round(r.coherence, 3), saturation=round(sat, 1)))

    tri = pd.DataFrame(rows).sort_values(["category", "flow"])
    tri.to_csv(OUT / "triage.csv", index=False)
    print(f"Wrote {len(tri)} montages -> {OUT}/")
    print("\nCategory counts:")
    print(tri.category.value_counts().to_string())
    print("\nBy category:")
    for cat, g in tri.groupby("category"):
        print(f"\n== {cat} ({len(g)}) ==")
        for _, r in g.iterrows():
            print(f"  {r['clip']:36s} cnt={r['count']:5.1f} flow={r.flow:.4f} "
                  f"coh={r.coherence:.2f} sat={r.saturation:.0f} ->{r.predicted}")


if __name__ == "__main__":
    main()
