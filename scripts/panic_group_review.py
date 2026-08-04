#!/usr/bin/env python3
"""One montage per (group, video) for the 5 Panic sub-groups, for visual audit.
Output -> outputs_improved/panic_subfolders/<group>/<video>.png
"""
import re
from pathlib import Path
import cv2, numpy as np, pandas as pd

G = pd.read_csv("outputs_improved/panic_groups.csv")
OUT = Path("outputs_improved/panic_subfolders")


def num(p):
    m = re.search(r"_clip_(\d+)", Path(p).stem)
    return int(m.group(1)) if m else 0


def midframe(p):
    c = cv2.VideoCapture(p); t = int(c.get(7)) or 1
    c.set(1, t // 2); ok, f = c.read(); c.release()
    return f if ok else None


def across(p, n):
    c = cv2.VideoCapture(p); t = int(c.get(7)) or 1; out = []
    for fr in np.linspace(0.15, 0.85, n):
        c.set(1, int(t * fr)); ok, im = c.read()
        if ok: out.append(im)
    c.release(); return out


for (group, vid), g in G.groupby(["group", "video_id"]):
    d = OUT / group; d.mkdir(parents=True, exist_ok=True)
    paths = g.sort_values("path", key=lambda s: s.map(num)).path.tolist()
    picks = []
    if len(paths) >= 5:
        for k in np.linspace(0, len(paths) - 1, 5, dtype=int):
            fr = midframe(paths[k])
            if fr is not None: picks.append((f"c{num(paths[k])}", fr))
    else:
        per = max(1, 5 // len(paths))
        for p in paths:
            for j, fr in enumerate(across(p, per)): picks.append((f"c{num(p)}.{j}", fr))
    if not picks: continue
    h = 240; imgs = []
    for lab, fr in picks:
        im = cv2.resize(fr, (int(fr.shape[1] * h / fr.shape[0]), h))
        cv2.putText(im, lab, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        imgs.append(im)
    strip = cv2.hconcat(imgs)
    bar = np.full((26, strip.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, f"{group} | {vid} ({len(paths)} clips)", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", vid)
    cv2.imwrite(str(d / f"{safe}.png"), cv2.vconcat([bar, strip]))
print("done");
for g in sorted(G.group.unique()):
    print(g, len(list((OUT / g).glob("*.png"))), "montages")
