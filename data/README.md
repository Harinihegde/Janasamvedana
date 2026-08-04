# Data

Raw videos are **not** committed to this repository (they are large and
redistribution-restricted). This folder documents how the data is structured
and includes only small curation metadata.

## Primary dataset — "Crowd Panic"
Short crowd clips organised by class, each clip cut from a WhatsApp/news source
video. Clips from the same source video share a `video_id` prefix
(e.g. `VID-20250924-WA0037_clip_003.mp4`), which is essential: **all train/test
splits group by `video_id`** so near-duplicate clips from one source never span
the split (the leakage fix).

Expected layout (point `scripts/extract_features.py --dataset` here):
```
crowd_panic/
├── No Panic/   *.mp4      (7 source videos, 364 clips)
├── Normal/     *.mp4      (13 videos, 526 clips)
├── Crowdy/     *.mp4      (34 videos, 159 clips)
└── Panic/      *.mp4      (23 videos, 281 clips)
```
Total: **71 unique source videos, 1,330 clips.**

## External generalization set — "Abnormal High-density Crowds"
Frame sequences (PNG) from 4 real incidents — **1_Times_Square, 2_Las_Vegas,
3_Love Parade, 4_Italy** — each split into Train (normal) and Test (abnormal)
sequences, plus the original `Footage` videos and (for Love Parade) PASCAL-VOC /
YOLO frame labels. Used only for held-out testing. `scripts/gen_test.py` reads
frames **directly from the distributed `archive.zip`** (no extraction needed).
Contact for that dataset: Samar Mahmoud (see the archive's `Read Me.txt`).

## Curation metadata included here
- **`kept_panic.txt`** — the manually reviewed set of Panic clips retained after
  the label audit (news-desk / outro-card contamination removed).
- **`panic_sublabels.csv`** — per-clip **crush vs flight** sub-labels derived
  from the manual visual audit (metadata only; the model does not currently
  split on it — see README "Future work").

## Reproducing features
Feature CSVs (`frame_features.csv`, etc.) are **not** committed — regenerate with
`scripts/extract_features.py` given the dataset path and YOLO weights.
