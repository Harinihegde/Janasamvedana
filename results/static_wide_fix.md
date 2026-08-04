# STATIC_WIDE undercounting fix — RESULT: imgsz/tiling did NOT help

12 frames/clip, per-frame mean counts. **Both higher imgsz and tiling LOWERED
counts** — refuting the 'YOLO undercounts dense distant crowds' premise.

## Overall (mean across 29 clips)

| Detector | Mean count | ×vs baseline |
|---|---|---|
| baseline (imgsz 512) | 7.9 | 1.00× |
| highres (imgsz 1280) | 2.3 | 0.29× |
| tiled 2×2 (imgsz 640) | 4.2 | 0.53× |

Tiling vs baseline: 10 clips up, 14 down, 5 ~same.
Highres vs baseline: 1 up, 20 down.

## Why (root causes)
1. **Low source resolution** (frames are 360–640px wide). imgsz=1280 only
   upscales/blurs; there is no hidden detail to recover.
2. **Baseline counts were partly false positives** — at higher effective
   resolution the detector rejects blob/texture/graphics detections (many
   clips drop toward 0).
3. **Genuine dense stampedes are an unresolvable blur** (e.g. Patna Gandhi
   Maidan CCTV on ABP News): a real Panic crowd of tiny blurry heads inside
   news graphics — bbox detection cannot count it at ANY imgsz. Needs a
   crowd DENSITY-MAP model (CSRNet), not detection.

## Per-clip (sorted by baseline count)

| clip | before(512) | highres(1280) | tiled |
|---|---|---|---|
| VID-20250924-WA0021_clip_034.mp4 | 22 | 5 | 11 |
| VID-20250924-WA0025_clip_019.mp4 | 19 | 0 | 0 |
| VID-20250924-WA0022_clip_007.mp4 | 16 | 3 | 5 |
| VID-20250924-WA0022_clip_008.mp4 | 16 | 2 | 3 |
| VID-20250924-WA0022_clip_009.mp4 | 15 | 3 | 4 |
| VID-20250924-WA0025_clip_025.mp4 | 15 | 0 | 1 |
| VID-20250924-WA0025_clip_026.mp4 | 15 | 0 | 1 |
| VID-20250924-WA0025_clip_006.mp4 | 15 | 1 | 1 |
| VID-20250924-WA0022_clip_010.mp4 | 12 | 4 | 7 |
| VID-20250924-WA0022_clip_013.mp4 | 9 | 7 | 9 |
| VID-20250924-WA0021_clip_041.mp4 | 8 | 2 | 4 |
| VID-20250924-WA0022_clip_012.mp4 | 8 | 6 | 9 |
| VID-20250924-WA0021_clip_035.mp4 | 7 | 1 | 3 |
| VID-20250924-WA0025_clip_018.mp4 | 7 | 1 | 1 |
| VID-20250924-WA0021_clip_010.mp4 | 6 | 2 | 4 |
| VID-20250924-WA0021_clip_016.mp4 | 6 | 2 | 7 |
| VID-20250924-WA0025_clip_024.mp4 | 5 | 0 | 0 |
| VID-20250924-WA0021_clip_029.mp4 | 5 | 4 | 7 |
| VID-20250924-WA0022_clip_011.mp4 | 5 | 3 | 5 |
| VID-20250924-WA0022_clip_014.mp4 | 5 | 3 | 7 |
| VID-20250924-WA0022_clip_015.mp4 | 4 | 3 | 4 |
| VID-20250924-WA0021_clip_032.mp4 | 3 | 4 | 6 |
| VID-20250924-WA0021_clip_040.mp4 | 2 | 2 | 4 |
| VID-20250924-WA0021_clip_019.mp4 | 2 | 1 | 3 |
| VID-20250924-WA0021_clip_018.mp4 | 1 | 2 | 4 |
| VID-20250924-WA0021_clip_020.mp4 | 1 | 1 | 4 |
| VID-20250924-WA0021_clip_025.mp4 | 0 | 1 | 1 |
| VID-20250924-WA0021_clip_017.mp4 | 0 | 1 | 3 |
| VID-20251025-WA0002_clip_003.mp4 | 0 | 0 | 1 |