#!/usr/bin/env python3
"""Find Panic clips the 21-feature 4-class model predicts as No-Panic/Normal
(the low-density 'leak'), and dump frame montages for visual inspection.

Output -> outputs_improved/panic_lowdensity/ (montage PNGs + a listing CSV).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from run_compression2 import AFTER_21, cv_oof
from stampede.dataset import stratified_group_folds

OUT = Path("outputs_improved/panic_lowdensity")
FEATURES_CSV = Path("outputs_compression2/frame_features.csv")


def montage(path: str, label_text: str, dest: Path, n=3):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    picks = [int(total * f) for f in (0.2, 0.5, 0.8)][:n]
    imgs = []
    for fi in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        h = 260
        fr = cv2.resize(fr, (int(fr.shape[1] * h / fr.shape[0]), h))
        imgs.append(fr)
    cap.release()
    if not imgs:
        return False
    strip = cv2.hconcat(imgs)
    bar = np.full((28, strip.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, label_text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(dest), cv2.vconcat([bar, strip]))
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(FEATURES_CSV)
    clips = (feats[["path", "video_id", "label"]].drop_duplicates("path")
             .reset_index(drop=True))
    folds = stratified_group_folds(clips, n_splits=5)
    oof = cv_oof(feats, AFTER_21, folds, clips)

    # Per-clip mean person count + hull density for context.
    ctx = (feats.groupby("path")
           .agg(count=("person_count", "mean"),
                hull_density=("hull_density", "mean"),
                nn=("nn_distance_mean", "mean")).reset_index())
    leaked = oof[(oof.actual == "Panic") &
                 (oof.predicted.isin(["No Panic", "Normal"]))].merge(ctx, on="path")
    leaked = leaked.sort_values("count")
    leaked["clip"] = leaked.path.apply(lambda p: Path(p).name)
    leaked[["clip", "video_id", "predicted", "count", "hull_density", "nn"]].to_csv(
        OUT / "leaked_panic_clips.csv", index=False)

    print(f"Panic clips predicted No-Panic/Normal: {len(leaked)} "
          f"(of {int((oof.actual=='Panic').sum())} Panic clips)")
    all_panic_count = ctx.merge(oof[oof.actual == "Panic"][["path"]])["count"].mean()
    print(f"mean person-count of leaked clips: {leaked['count'].mean():.1f} "
          f"(vs {all_panic_count:.1f} all-Panic)")
    print("\nExample leaked Panic clips (lowest density first):")
    # Sample across the density range: mix of lowest and mid.
    picks = pd.concat([leaked.head(6), leaked.iloc[len(leaked)//2:len(leaked)//2+2]])
    made = []
    for _, r in picks.iterrows():
        label = (f"PANIC->{r['predicted']} | count={r['count']:.0f} "
                 f"hull_d={r['hull_density']:.0f} nn={r['nn']:.3f} | {r['clip']}")
        dest = OUT / f"{Path(r.path).stem}.png"
        if montage(r.path, label, dest):
            made.append(str(dest))
            print(f"  {label}")
    print("\nMontages:")
    for m in made:
        print(" ", m)


if __name__ == "__main__":
    main()
