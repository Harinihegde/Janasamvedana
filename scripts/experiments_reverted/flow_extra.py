#!/usr/bin/env python3
"""Fast flow-only pass: compute flight/scatter motion features (no YOLO).

For every clip, replicating the main pipeline's frame sampling (stride +
320px downscale) exactly so rows align on (path, frame_index):

  * flow_dir_divergence - magnitude-WEIGHTED circular variance of optical-flow
    direction (1 - weighted resultant length). Unlike the existing
    ``direction_consistency`` (unweighted over all pixels, incl. static
    background), this is dominated by the *moving* pixels: ~0 = everyone flows
    one way (march/Crowdy), ~1 = motion scattered in many directions (flight).
  * flight_score = frame flow-magnitude (speed) x flow_dir_divergence
    (the "fast AND scattering" flight-panic signature).

Output -> outputs_improved/flow_extra.csv  (path, frame_index, 2 features)
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from stampede.config import FLOW_WIDTH, FRAME_STRIDE, MAX_FRAMES_PER_CLIP

MANIFEST = Path("outputs_compression2/manifest.csv")
OUT = Path("outputs_improved/flow_extra.csv")


def clip_flow_feats(path: str):
    cap = cv2.VideoCapture(path)
    prev = None
    raw = kept = 0
    rows = []
    while kept < MAX_FRAMES_PER_CLIP:
        ok, frame = cap.read()
        if not ok:
            break
        if raw % FRAME_STRIDE != 0:
            raw += 1
            continue
        fi = raw
        raw += 1
        kept += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(
            gray, (FLOW_WIDTH, max(1, int(FLOW_WIDTH * gray.shape[0] / gray.shape[1]))))
        diag = float(np.hypot(*gray.shape[:2])) or 1.0
        div = flight = 0.0
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            m = mag.ravel()
            msum = float(m.sum())
            if msum > 1e-6:
                cx = float((m * np.cos(ang).ravel()).sum()) / msum
                sy = float((m * np.sin(ang).ravel()).sum()) / msum
                R = float(np.hypot(cx, sy))  # magnitude-weighted resultant
            else:
                R = 1.0
            div = 1.0 - R
            speed = float(mag.mean()) / diag
            flight = speed * div
        prev = gray
        rows.append(dict(path=path, frame_index=fi,
                         flow_dir_divergence=div, flight_score=flight))
    cap.release()
    return rows


def main():
    manifest = pd.read_csv(MANIFEST)
    rows = []
    n = len(manifest)
    t0 = time.time()
    for j, r in manifest.iterrows():
        rows.extend(clip_flow_feats(r.path))
        if (j + 1) % 100 == 0 or j + 1 == n:
            el = time.time() - t0
            print(f"flow {j+1}/{n}  elapsed={el:.0f}s eta={el/(j+1)*(n-j-1):.0f}s",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows -> {OUT}")


if __name__ == "__main__":
    main()
