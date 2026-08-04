#!/usr/bin/env python3
"""Build one review montage per Crowdy source video (spanning its clip sequence).

For each video: sort its clips by number, pick ~5 evenly-spaced sample points
across the sequence, take one mid-frame from each, and lay them out with labels
so a human can eyeball the whole video's evolution at a glance.
Output -> outputs_improved/crowdy_review/<video>.png
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

OUT = Path("outputs_improved/crowdy_review")
MANIFEST = Path("outputs_compression2/manifest.csv")


def clip_num(path):
    m = re.search(r"_clip_(\d+)", Path(path).stem)
    return int(m.group(1)) if m else 0


def mid_frame(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def frames_across_clip(path, n):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    out = []
    for f in np.linspace(0.1, 0.9, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * f))
        ok, fr = cap.read()
        if ok:
            out.append(fr)
    cap.release()
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="Crowdy")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global OUT
    OUT = Path(a.out) if a.out else Path(f"outputs_improved/{a.label.lower()}_review")
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(MANIFEST)
    c = m[m.label == a.label].copy()
    c["num"] = c.path.apply(clip_num)

    for vid, g in c.groupby("video_id"):
        g = g.sort_values("num")
        paths = g.path.tolist()
        picks = []  # (label, frame)
        if len(paths) >= 5:
            idxs = np.linspace(0, len(paths) - 1, 5, dtype=int)
            for k in idxs:
                fr = mid_frame(paths[k])
                if fr is not None:
                    picks.append((f"clip{g.num.tolist()[k]}", fr))
        else:
            # few clips: spread frames within each clip
            per = max(1, 5 // len(paths))
            for p in paths:
                for j, fr in enumerate(frames_across_clip(p, per)):
                    picks.append((f"clip{clip_num(p)}.{j}", fr))
        if not picks:
            continue
        h = 240
        imgs = []
        for lab, fr in picks:
            im = cv2.resize(fr, (int(fr.shape[1] * h / fr.shape[0]), h))
            cv2.putText(im, lab, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2, cv2.LINE_AA)
            imgs.append(im)
        strip = cv2.hconcat(imgs)
        bar = np.full((26, strip.shape[1], 3), 30, np.uint8)
        cv2.putText(bar, f"{vid}  ({len(paths)} clips)", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", vid)
        cv2.imwrite(str(OUT / f"{safe}.png"), cv2.vconcat([bar, strip]))
    print(f"Wrote montages for {c.video_id.nunique()} videos -> {OUT}/")


if __name__ == "__main__":
    main()
