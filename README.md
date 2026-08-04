# Janasamvedana — Early Crowd-Crush & Stampede Risk Detection

A two-stage video system that rates how dangerous a crowd is and raises an
early warning when the situation escalates toward a crush or stampede.

## Project overview
Crowd crushes (Love Parade, Hillsborough, Itaewon, Kumbh) build up before they
turn fatal. Janasamvedana watches crowd video and, frame by frame:
1. **Stage 1 — Crowd-risk classification.** Rates the crowd state
   (No Panic / Normal / Crowdy / Panic), and as a deployment-oriented binary,
   **Safe vs Stampede-Risk**, producing a continuous risk score in [0, 1].
2. **Stage 2 — Escalation / early warning.** Smooths the risk over time and
   raises an alert when it rises through danger thresholds (with hysteresis to
   suppress flicker).

Everything is evaluated **leakage-safe**: because clips are cut from a smaller
set of source videos, all splits are grouped by source video (an earlier
clip-level split leaked near-duplicate frames and inflated accuracy to ~80.8%).

## Architecture
- **Features (per frame):** YOLO person detection + Farnebäck optical flow →
  **21 features** spanning density, motion, temporal instability, and
  compression/occlusion. Resolution/fps-normalised so clips are comparable.
- **Stage 1:** class-balanced RandomForest. Two heads — 4-class and binary
  (Safe vs Stampede-Risk). Risk score = class-probability expectation over
  ordinal risk anchors.
- **Stage 2:** causal smoothing + risk-velocity + a hysteresis rule
  (k consecutive frames above threshold) → Safe→Risk escalation alerts.
- **Evaluation:** grouped 80/20 split **and** Leave-One-Video-Out (LOVO).

## Dataset
- **71 unique source videos → 1,330 clips**

  | Class | Videos | Clips |
  |---|---|---|
  | No Panic | 7 | 364 |
  | Normal | 13 | 526 |
  | Crowdy | 34 | 159 |
  | Panic | 23 | 281 |

- Raw videos are **not** committed (see [`data/README.md`](data/README.md) for
  sourcing and structure).
- **Augmentation** (scarce classes only — Panic + No-Panic): speed perturbation
  (0.8×, 1.2×), centre crop/zoom, and brightness jitter. Under LOVO this raised
  Panic recall (~0.50 → 0.75 clip-level) at a precision cost — an
  operating-point shift, reported as an experiment (not folded into the
  headline model).
- **External validation:** 4 real incidents (Times Square, Las Vegas, Love
  Parade, Italy) streamed frame-by-frame from a separate dataset
  (`scripts/gen_test.py`).

## Results (leakage-safe; honest)

**Binary — Safe vs Stampede-Risk (5-fold grouped out-of-fold, 21 features):**
- Accuracy **86.3%** (vs 78.9% always-"Safe" baseline) · macro-F1 **0.76**
- Stampede-Risk: precision 0.76 · recall 0.51 · F1 0.61

**4-class (5-fold grouped out-of-fold, 21 features):**
- Accuracy 68.0% · macro-F1 0.66 · Panic F1 0.57

*(The two compression-ratio features aid the 4-class Crowdy↔Panic boundary
[+0.04 Panic F1] and are neutral-to-slightly-negative for the binary head; kept
for one consistent feature set.)*

**LOVO — leave-one-video-out (most rigorous):**
- Panic recall ~**0.41** (video-level) / 0.50 (clip); No-Panic 1.00,
  Normal 0.90, Crowdy 0.94

**External generalization (4 unseen real incidents, no tuning tricks):**
- **38%** panic caught at **0% false positives** with a fixed threshold
- **71%** caught at **0% false positives** after per-site normal-baseline
  threshold calibration (a standard deployment step)

## Known limitations
- **Sparse flight-panic** (people fleeing across open space) is largely missed:
  it is low-density, and the model leans on density.
- **Very dense / distant / low-res crushes** can be undercounted by the person
  detector (it caps at 300 boxes; blurry CCTV saturates).
- **Root causes:** only **23 unique Panic source videos**, and "Panic" conflates
  two opposite phenomena — **crush** (very dense) and **flight** (sparse, fast).
  In-domain, the densest crowds are actually *Crowdy festivals* (labelled Safe),
  so density alone cannot flag a crush.

## What was tried and didn't work (rigorous negative results)
Reported transparently — these shaped the final design. Code lives in
`scripts/experiments_reverted/`.
- **Motion / flight features** (flow divergence + speed): **no improvement**
  (Δ Panic-F1 ≈ 0.00). Flight is directionally *coherent* (a crowd flees one
  way), so "divergence" was the wrong signal, and density dominates the model.
- **Crush-only pruning of the Panic class:** **hurt** — LOVO crush recall fell
  0.69 → 0.26. Shrinking an already-scarce class removes signal.
- **Density risk floor** ("flag any extreme density as danger"): **invalid,
  reverted.** It was tuned with knowledge of the external test, and under LOVO
  it flagged ~3–4× more Safe (dense Crowdy festivals) than Risk clips
  (precision 0.81 → 0.67).
- **Detector tiling / higher input resolution** for undercounting: did not
  recover counts on low-resolution footage.

## Getting started (for collaborators)

**1. Clone and set up the environment**
```bash
git clone https://github.com/Harinihegde/Janasamvedana.git
cd Janasamvedana
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Get the things that aren't in the repo** (large / restricted → gitignored):
- **Dataset** — the "Crowd Panic" videos, arranged as in
  [`data/README.md`](data/README.md). Keep the clip naming
  (`VID-<id>_clip_NNN.mp4`) so all splits stay leakage-safe by `video_id`.
- **YOLO weights** — a person-detection `.pt` (COCO or a fine-tuned person
  model). Save it anywhere; pass its path with `--weights`.
- *(optional)* the external "Abnormal High-density Crowds" `archive.zip` (for
  `gen_test.py`) — place it at the repo root.

**3. Run the pipeline**
```bash
# Stage 0 — extract the 21 per-frame features (one-time; writes outputs/frame_features.csv)
python scripts/extract_features.py --dataset /path/to/crowd_panic --weights /path/to/best.pt --output outputs

# Stage 1 — train + leakage-safe evaluation (4-class)
python scripts/train_stage1.py --output outputs

# Stage 2 — escalation / early-warning alerts
python scripts/run_stage2.py --output outputs

# Binary Safe-vs-Risk head + leave-one-video-out (LOVO)
python scripts/run_binary.py --features outputs/frame_features.csv --output outputs_binary
python scripts/run_lovo.py   --oof-cache outputs/lovo_oof.csv

# External generalization test (reads frames straight from archive.zip)
python scripts/gen_test.py
```

**Notes for teammates**
- Feature CSVs, trained models, and everything under `outputs*/` are
  **regenerated** by the steps above and are gitignored — don't commit them.
- The committed `results/` folder holds the **reference numbers from our runs**
  (LOVO, generalization, calibration, audits) — compare your runs against these.
- Feature extraction (YOLO + optical flow over ~1,330 clips) is the slow step
  (~1–1.5 h). It's cached, so later stages are fast. On Apple-Silicon,
  `ultralytics`/`torch` use MPS automatically; CPU-only works but is slower.
- `scripts/experiments_reverted/` are **documented negative results** — read its
  README before re-trying those ideas (see also the top-level "What was tried"
  section).

## Future work
1. **Collect more distinct Panic source videos — especially flight-panic**
   (currently ~2 distinct scenes). This is the single lever with real headroom;
   features and thresholds have hit diminishing returns.
2. Split Panic into **crush vs flight** sub-labels (metadata in
   [`data/panic_sublabels.csv`](data/panic_sublabels.csv)) once enough flight
   footage exists to train/evaluate them separately.
3. A **crowd-density-map counter** (e.g. CSRNet) for distant/blurry dense crowds
   the detector undercounts.

## Repository layout
```
stampede/    pipeline package (config, dataset, detector, features,
             classifier, evaluate, escalation, visualize)
scripts/     CLI + audit + external-validation + exploration scripts
             experiments_reverted/  documented negative results
data/         dataset notes + crush/flight curation metadata (no videos)
results/      LOVO / generalization / calibration / audit reports + plots
```

## License
MIT — see [LICENSE](LICENSE).
