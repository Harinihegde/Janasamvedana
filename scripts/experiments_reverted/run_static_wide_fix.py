#!/usr/bin/env python3
"""Fix undercounting on the STATIC_WIDE Panic clips (distant/aerial crowds).

Compares three detectors on the SAME sampled frames of the 29 STATIC_WIDE clips:
  * baseline  - YOLO imgsz=512 (current pipeline setting)
  * highres   - YOLO imgsz=1280 (one pass, larger input)
  * tiled     - 2x2 tiles (10% overlap), YOLO imgsz=640 per tile, boxes mapped
                back to full-frame coords + global NMS to dedup seam duplicates

Reports per-clip and overall mean person-count and crowd-density
(count / frame-area * 1e5) before vs after.
Output -> outputs_improved/static_wide_fix.{csv,md}
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.ops import nms
from ultralytics import YOLO

WEIGHTS = "/Users/harinihegde/Downloads/best_combined.pt"
TRIAGE = Path("outputs_improved/panic_lowdensity_all/triage.csv")
FEATURES_CSV = Path("outputs_compression2/frame_features.csv")
OUT = Path("outputs_improved")
N_FRAMES = 12       # evenly-spaced frames per clip for the diagnostic
CONF = 0.25


def detect_count(model, frame, imgsz):
    r = model(frame, classes=[0], conf=CONF, imgsz=imgsz, verbose=False)[0]
    return 0 if r.boxes is None else len(r.boxes)


def tiled_count(model, frame, tiles=(2, 2), overlap=0.10, imgsz=640):
    H, W = frame.shape[:2]
    th, tw = H // tiles[0], W // tiles[1]
    oh, ow = int(th * overlap), int(tw * overlap)
    boxes, scores = [], []
    for i in range(tiles[0]):
        for j in range(tiles[1]):
            y0, y1 = max(0, i * th - oh), min(H, (i + 1) * th + oh)
            x0, x1 = max(0, j * tw - ow), min(W, (j + 1) * tw + ow)
            tile = frame[y0:y1, x0:x1]
            r = model(tile, classes=[0], conf=CONF, imgsz=imgsz, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            b = r.boxes.xyxy.cpu().numpy().copy()
            c = r.boxes.conf.cpu().numpy()
            b[:, [0, 2]] += x0
            b[:, [1, 3]] += y0
            boxes.append(b)
            scores.append(c)
    if not boxes:
        return 0
    b = np.vstack(boxes)
    s = np.concatenate(scores)
    keep = nms(torch.tensor(b, dtype=torch.float32),
               torch.tensor(s, dtype=torch.float32), 0.5)
    return int(len(keep))


def sample_frames(path, n=N_FRAMES):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    area = float(cap.get(3) * cap.get(4)) or 1.0
    idxs = np.linspace(0, max(total - 1, 0), min(n, total), dtype=int)
    out = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if ok:
            out.append(fr)
    cap.release()
    return out, area


def main():
    tri = pd.read_csv(TRIAGE)
    static = tri[tri.category == "STATIC_WIDE"]
    feats = pd.read_csv(FEATURES_CSV)
    feats["clip"] = feats.path.apply(lambda p: Path(p).name)
    pathmap = feats.drop_duplicates("clip").set_index("clip").path.to_dict()

    model = YOLO(WEIGHTS)
    rows = []
    for n, clip in enumerate(static['clip'], 1):
        path = pathmap[clip]
        frames, area = sample_frames(path)
        if not frames:
            continue
        base = np.mean([detect_count(model, f, 512) for f in frames])
        high = np.mean([detect_count(model, f, 1280) for f in frames])
        tile = np.mean([tiled_count(model, f) for f in frames])
        rows.append(dict(
            clip=clip, area=area,
            count_before=round(base, 1),
            count_highres=round(high, 1),
            count_tiled=round(tile, 1),
            density_before=round(base / area * 1e5, 2),
            density_highres=round(high / area * 1e5, 2),
            density_tiled=round(tile / area * 1e5, 2),
        ))
        print(f"{n:2d}/{len(static)} {clip:34s} "
              f"cnt {base:5.1f} -> hi {high:5.1f} / tile {tile:5.1f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "static_wide_fix.csv", index=False)

    L = ["# STATIC_WIDE undercounting fix — detector comparison (29 clips)\n",
         f"Sampled {N_FRAMES} frames/clip. Counts are per-frame means.\n",
         "## Overall (mean across clips)\n",
         "| Detector | Mean count | Mean crowd-density | ×vs baseline |",
         "|---|---|---|---|"]
    b = df.count_before.mean()
    for name, ccol, dcol in [("baseline (512)", "count_before", "density_before"),
                             ("highres (1280)", "count_highres", "density_highres"),
                             ("tiled 2x2 (640)", "count_tiled", "density_tiled")]:
        L.append(f"| {name} | {df[ccol].mean():.1f} | {df[dcol].mean():.2f} | "
                 f"{df[ccol].mean()/b:.2f}× |")
    L += ["", "## Per-clip counts (before → highres / tiled)\n",
          "| clip | before | highres | tiled |", "|---|---|---|---|"]
    for _, r in df.sort_values("count_before").iterrows():
        L.append(f"| {r['clip']} | {r.count_before:.0f} | {r.count_highres:.0f} | "
                 f"{r.count_tiled:.0f} |")
    (OUT / "static_wide_fix.md").write_text("\n".join(L))
    print("\n" + "\n".join(L[:12]))


if __name__ == "__main__":
    main()
