#!/usr/bin/env python3
"""Stage A + B diagnostics: per-video camera-distance and camera-stability.

A (distance): mean detected person bbox HEIGHT as a fraction of frame height,
   over confidently-detected boxes (YOLO conf >= 0.4). Small = far/aerial/CCTV.
B (stability): per-frame-pair global camera translation (RANSAC affine on
   tracked feature points), magnitude / frame-diagonal. ~0 = fixed camera
   (CCTV/tripod), large = handheld pan/shake.

Both computed on the SAME ~15 frame-pairs sampled across each video's clips.
Output -> outputs_improved/camera_axes.csv  (per video)
NOTE (Stage B caveat): in frames fully packed with moving crowd there is little
static background, so the affine estimate there is upward-biased.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Path to the custom YOLO person detector (see README's "You'll need to get
# separately" section). Override with the YOLO_WEIGHTS env var rather than
# editing this file.
WEIGHTS = os.environ.get("YOLO_WEIGHTS", "best_combined.pt")
MANIFEST = Path("outputs_compression2/manifest.csv")
OUT = Path("outputs_improved/camera_axes.csv")
CONF_HI = 0.40          # "confidently detected" for the distance measure
N_CLIPS = 4             # clips sampled per video
N_POS = 4               # positions sampled per clip (each -> a frame pair)


def clipnum(p):
    m = re.search(r"_clip_(\d+)", Path(p).stem)
    return int(m.group(1)) if m else 0


def camera_shift(f1, f2, diag):
    """Global camera translation magnitude between two frames / diagonal."""
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(g1, maxCorners=300, qualityLevel=0.01,
                                  minDistance=7)
    if pts is None or len(pts) < 12:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, pts, None)
    if nxt is None:
        return None
    ok = st.ravel() == 1
    p0, p1 = pts[ok], nxt[ok]
    if len(p0) < 12:
        return None
    M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                         ransacReprojThreshold=3)
    if M is None:
        return None
    dx, dy = M[0, 2], M[1, 2]
    return float(np.hypot(dx, dy) / diag)


def main():
    man = pd.read_csv(MANIFEST)
    man["num"] = man.path.apply(clipnum)
    model = YOLO(WEIGHTS)
    rows = []
    t0 = time.time()
    vids = man.video_id.unique()
    for vi, (vid, g) in enumerate(man.groupby("video_id"), 1):
        label = g.label.iloc[0]
        paths = g.sort_values("num").path.tolist()
        pick = [paths[k] for k in np.linspace(0, len(paths) - 1,
                                              min(N_CLIPS, len(paths)), dtype=int)]
        heights, shifts = [], []
        for p in pick:
            cap = cv2.VideoCapture(p)
            total = int(cap.get(7)) or 1
            fh = float(cap.get(4)) or 1.0
            fw = float(cap.get(3)) or 1.0
            diag = float(np.hypot(fw, fh)) or 1.0
            for fr in np.linspace(0.15, 0.85, N_POS):
                idx = int(total * fr)
                cap.set(1, idx)
                ok1, a = cap.read()
                ok2, b = cap.read()
                if not ok1:
                    continue
                # A: bbox heights on confident detections
                r = model(a, classes=[0], conf=CONF_HI, imgsz=512, verbose=False)[0]
                if r.boxes is not None and len(r.boxes):
                    bb = r.boxes.xyxy.cpu().numpy()
                    heights.extend(((bb[:, 3] - bb[:, 1]) / fh).tolist())
                # B: camera shift
                if ok2:
                    s = camera_shift(a, b, diag)
                    if s is not None:
                        shifts.append(s)
            cap.release()
        rows.append(dict(
            video_id=vid, label=label, n_clips=len(paths),
            n_boxes=len(heights),
            bbox_height_ratio=float(np.mean(heights)) if heights else np.nan,
            n_pairs=len(shifts),
            camera_shift_ratio=float(np.median(shifts)) if shifts else np.nan,
        ))
        print(f"{vi:2d}/{len(vids)} {vid[:34]:34s} label={label:8s} "
              f"h={rows[-1]['bbox_height_ratio']!s:.6} shift={rows[-1]['camera_shift_ratio']!s:.6} "
              f"({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
