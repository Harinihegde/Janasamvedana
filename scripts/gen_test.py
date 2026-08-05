#!/usr/bin/env python3
"""External generalization test: train on ALL current data, test on the new
'Abnormal High-density Crowds' dataset (Times Square / Las Vegas / Love Parade /
Italy), streamed frame-by-frame from archive.zip (no extraction; disk is full).

Model: binary Safe-vs-Stampede-Risk RF on the full 21-feature current dataset
(all 71 videos, no hold-out). New data: Test/abnormal sequences -> expected
Panic; Train/normal -> expected Safe. Reports per-incident catch rate + the
false-positive rate on normal segments, and whether crush (Love Parade) is
caught more reliably than flight (Vegas / Italy).
Output -> outputs_improved/gen_test.{json,md} and gen_features.csv
"""
from __future__ import annotations

import json
import os
import re
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from stampede.classifier import train
from stampede.config import (FEATURE_COLUMNS, FLOW_WIDTH, FRAME_STRIDE,
                             MAX_FRAMES_PER_CLIP)
from stampede.detector import Detector
from stampede.features import (CentroidTracker, _add_window_features,
                               compute_bbox_area_stats,
                               compute_bbox_overlap_ratio, compute_compression,
                               compute_detection_confidence,
                               compute_spatial_density_mismatch)

ZIP = "archive.zip"
ROOT = "Abnormal High-density Crowds"
# Path to the custom YOLO person detector (see README's "You'll need to get
# separately" section). Override with the YOLO_WEIGHTS env var rather than
# editing this file.
WEIGHTS = os.environ.get("YOLO_WEIGHTS", "best_combined.pt")
OUT = Path("outputs_improved")
CHUNK = 150            # frames per synthetic clip (~5-10 s)
POS, NEG = "Stampede Risk", "Safe"

# (incident, video_id, expected, fps, [sequence folder prefixes], type)
SEQUENCES = [
    ("Times Square", "flight", 29.97, [
        ("View_1/Train", "normal"), ("View_1/Test", "abnormal"),
        ("View_2/Train", "normal"), ("View_2/Test", "abnormal"),
        ("View_3/Train", "normal"), ("View_3/Test", "abnormal")], "1_Times_Square"),
    ("Las Vegas", "flight", 15.17, [
        ("Train", "normal"), ("Test_1", "abnormal"), ("Test_2", "abnormal"),
        ("Test_3", "abnormal"), ("Test_4", "abnormal")], "2_Las_Vegas"),
    ("Love Parade", "crush", 25.0, [
        ("Train_1", "normal"), ("Train_2", "normal"), ("Test", "abnormal")],
     "3_Love Parade"),
    ("Italy", "flight", 25.0, [
        ("View_1/Test", "abnormal"), ("View_2/Test", "abnormal")], "4_Italy"),
]


def rows_from_frames(frames, fps, detector):
    """Replicates features._instant_rows over pre-sampled frames (stride already
    applied), so the 21 features match the training pipeline exactly."""
    h, w = frames[0].shape[:2]
    diag = float(np.hypot(w, h)) or 1.0
    area = float(w * h) or 1.0
    dt = FRAME_STRIDE / fps
    tracker = CentroidTracker(max_dist=0.12 * max(w, h))
    prev_gray = None
    rows = []
    for i, frame in enumerate(frames):
        boxes, conf, count, fb = detector(frame)
        disp = tracker.update(boxes)
        overlap = compute_bbox_overlap_ratio(boxes)
        cmn, cstd = compute_detection_confidence(conf)
        amean, avar = compute_bbox_area_stats(boxes, area)
        smm = compute_spatial_density_mismatch(frame, boxes)
        cent = (boxes[:, :2] + boxes[:, 2:]) / 2.0 if len(boxes) else np.empty((0, 2))
        hd, nnd = compute_compression(cent, area, diag)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (FLOW_WIDTH, max(1, int(FLOW_WIDTH * h / w))))
        fdiag = float(np.hypot(*gray.shape[:2])) or 1.0
        fmm = fmv = coh = 0.0
        if prev_gray is not None:
            fl = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(fl[..., 0], fl[..., 1])
            fmm = float(mag.mean()) / fdiag
            fmv = float(mag.var()) / (fdiag ** 2)
            coh = float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))
        prev_gray = gray
        if disp.size:
            sp = disp / diag / dt
            vpp, vv = float(sp.mean()), float(sp.var())
        else:
            vpp = vv = 0.0
        tp = tracker.prev
        disp_sp = float(np.hypot(tp[:, 0].std(), tp[:, 1].std()) / diag) if (tp is not None and len(tp) > 1) else 0.0
        rows.append(dict(frame_index=i, used_fallback=int(fb), person_count=float(count),
                         crowd_density=count / area * 1e5, density_norm=float(1 - np.exp(-count / 50.0)),
                         flow_mag_mean=fmm, flow_mag_var=fmv, velocity_per_person=vpp, velocity_var=vv,
                         direction_consistency=coh, trajectory_dispersion=disp_sp,
                         bbox_overlap_ratio=overlap, detection_confidence_mean=cmn,
                         detection_confidence_std=cstd, bbox_area_variance=avar, bbox_area_mean=amean,
                         spatial_density_mismatch=smm, hull_density=hd, nn_distance_mean=nnd))
    df = _add_window_features(pd.DataFrame(rows))
    for c in FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = 0.0
    return df[FEATURE_COLUMNS].fillna(0.0)


def seq_frames(z, prefix):
    names = [n for n in z.namelist()
             if n.startswith(prefix + "/") and re.search(r"/scene-\d+\.png$", n)
             and "/Label/" not in n]
    return sorted(names)


def main():
    OUT.mkdir(exist_ok=True)
    z = zipfile.ZipFile(ZIP)
    det = Detector(WEIGHTS, imgsz=512)
    rows = []
    t0 = time.time()
    for incident, kind, fps, seqs, folder in SEQUENCES:
        for sub, typ in seqs:
            prefix = f"{ROOT}/{folder}/{sub}"
            names = seq_frames(z, prefix)
            if not names:
                print(f"  [warn] no frames: {prefix}", flush=True)
                continue
            for ci, start in enumerate(range(0, len(names), CHUNK)):
                chunk = names[start:start + CHUNK][::FRAME_STRIDE][:MAX_FRAMES_PER_CLIP]
                if len(chunk) < 3:
                    continue
                frames = [cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
                          for n in chunk]
                feats = rows_from_frames(frames, fps, det)
                agg = feats.mean().to_dict()  # clip-level = mean of frames (for context)
                feats["path"] = f"{incident}/{sub}/chunk{ci}"
                feats["incident"] = incident
                feats["kind"] = kind
                feats["type"] = typ
                rows.append(feats)
            print(f"{incident:12s} {sub:14s} {typ:8s} {len(names):4d} frames "
                  f"({time.time()-t0:.0f}s)", flush=True)
    allf = pd.concat(rows, ignore_index=True)
    allf.to_csv(OUT / "gen_features.csv", index=False)

    # --- train binary Safe-vs-Risk on ALL current data (21 feat) ---
    cur = pd.read_csv("outputs_compression2/frame_features.csv")
    cur["label"] = np.where(cur.label == "Panic", POS, NEG)
    model = train(cur, feature_cols=FEATURE_COLUMNS,
                  risk_anchors={NEG: 0.0, POS: 1.0})

    allf["risk"] = model.risk_score(allf)
    # clip-level: mean risk per synthetic clip
    clip = allf.groupby(["incident", "kind", "type", "path"]).risk.mean().reset_index()
    clip["pred_panic_0.5"] = clip.risk >= 0.5

    def summary(df):
        return {"n_clips": int(len(df)), "mean_risk": round(float(df.risk.mean()), 3),
                "flagged_panic_rate": round(float((df.risk >= 0.5).mean()), 3)}

    report = {"by_incident": {}, "overall": {}}
    for inc, g in clip.groupby("incident"):
        report["by_incident"][inc] = {
            "kind": g.kind.iloc[0],
            "ABNORMAL (want panic)": summary(g[g.type == "abnormal"]),
            "NORMAL (want safe)": summary(g[g.type == "normal"]) if (g.type == "normal").any() else "n/a",
        }
    ab = clip[clip.type == "abnormal"]; nm = clip[clip.type == "normal"]
    report["overall"] = {
        "abnormal_catch_rate": round(float((ab.risk >= 0.5).mean()), 3),
        "normal_false_positive_rate": round(float((nm.risk >= 0.5).mean()), 3) if len(nm) else None,
        "n_abnormal_clips": int(len(ab)), "n_normal_clips": int(len(nm))}
    (OUT / "gen_test.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
