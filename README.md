# Janasamvedana — Early Crowd-Crush & Stampede Detection

## What this project does

Crowd crushes and stampedes (like Love Parade, Hillsborough, Kumbh Mela) don't happen instantly — they build up. This project watches crowd video and tries to catch that buildup early, in two steps:

1. **Stage 1 — "How dangerous is this crowd right now?"**
   Looks at each moment of video and rates it: No Panic, Normal, Crowdy, or Panic. Also gives a simpler yes/no: Safe or Risk.

2. **Stage 2 — "Should we sound an alarm?"**
   Watches how the risk level changes over time and raises an early warning if things are clearly getting worse — before it becomes a full panic.

## How it works (simple version)

For every video, we detect people (using YOLO) and track their movement (optical flow) to compute 21 numbers per frame — things like how crowded it is, how fast people are moving, how compressed/packed they are. A machine learning model (RandomForest) looks at these numbers and predicts the risk level.

## Our dataset

We collected 71 real videos, split into 1,330 short clips:

| Category | Videos | Clips |
|---|---|---|
| No Panic | 7 | 364 |
| Normal | 13 | 526 |
| Crowdy | 34 | 159 |
| Panic | 23 | 281 |

**Important:** Panic footage is genuinely hard to find (only 23 unique videos), so we used data augmentation (speed changes, cropping, brightness changes) to stretch this limited data further.

**Note on testing:** We made sure to always test on videos the model never saw during training (not just different video clips — entire different source videos). This is called "leakage-safe" testing, and it's stricter than what a lot of similar projects do. An earlier, less careful test method showed a fake ~81% accuracy — the real, honest number is lower, but trustworthy.

## Our results (the honest numbers)

### Stage 1 — Risk classification

**Simple Safe vs Risk detection:**
- 86.3% accuracy (compared to 78.9% if you just always guessed "safe")
- When it says "risk," it's right 76% of the time (precision), and catches about half (51%) of real dangerous situations (recall)

**Detailed 4-category detection (No Panic/Normal/Crowdy/Panic):**
- 69% accuracy overall (up from 68% after fixing a label-contamination bug — see below)
- Panic specifically: ~61% reliable, catches ~58% of real panic cases

**Testing on whole videos it's never seen (strictest test):**
- Almost perfect at recognizing "definitely safe" videos (90-100%)
- Only catches ~41% of full Panic videos — this is our biggest weakness

**Testing on 4 completely new, real-world incidents (Times Square, Las Vegas, Love Parade, Italy):**
- Without any adjustment: catches 38% of real danger, with zero false alarms
- After calibrating the alarm sensitivity per location (a normal real-world setup step): catches 71% of real danger, still zero false alarms
- Caveat: this 71% number should be treated cautiously — some individual "catches" in this test may have been coincidental rather than the model genuinely recognizing danger. Treat 41-58% (our own dataset numbers) as the more dependable range.

### Stage 2 — Early-warning alerts

**Important context:** we don't have real footage of a crowd smoothly turning from calm into panic in one continuous video. So we tested this by artificially gluing a calm clip to a panic clip and checking if the system notices the switch. This is a reasonable stand-in, but it's a simulated test, not a real one.

**Table 1 — Base model (trained on original data only):**

| Alarm sensitivity | Catches real transitions | False alarms on calm videos | Warning timing |
|---|---|---|---|
| Standard (0.50) | 63% | 57% | 4 frames late |
| Stricter (0.60) | 41% | 47% | 5 frames late |
| Strictest (0.70) | 35% | 31% | 6 frames late |

**Table 2 — Same test, but trained on extra (augmented) Panic data:**

| Alarm sensitivity | Catches real transitions | False alarms on calm videos | Warning timing |
|---|---|---|---|
| Standard (0.50) | 96% | 84% | 7 frames early |
| Stricter (0.60) | 83% | 69% | 4 frames late |
| Strictest (0.70) | 72% | 53% | 5 frames late |

**How to read these two tables together:** adding extra training data (via augmentation) makes the system catch far more real escalations — up to 96% — and even warn *early* instead of late, at the standard sensitivity. But it comes at a real cost: false alarms jump to 84%, meaning it would cry wolf on the vast majority of calm crowds too. This isn't the model "understanding panic better" — it's the model becoming more trigger-happy overall, which mechanically raises both detection and false alarms together.

**The best middle-ground we found:** using the augmented model with a stricter "wait for 5 confirmations before alerting" rule, at the standard 0.50 sensitivity: catches 89% of real escalations, still with a high false-alarm rate (71%), but warns about 11 frames early.

**Bottom line for Stage 2:** the augmented version clearly detects more and warns earlier — but at a false-alarm rate too high for real deployment as-is. Whether the base model (fewer false alarms, but late/less sensitive) or the augmented model (more sensitive, more false alarms) is more useful depends on the deployment context — for now, both are shown honestly so the tradeoff is visible, not hidden behind one "best" number. It also inherits whatever mistakes Stage 1 already makes, since it depends on Stage 1's risk scores.

**Note:** the tables above are from a separate binary Safe/Risk study (`scripts/run_lovo.py`). The actual shipped Stage 2 module (`stampede/escalation.py`, run via `scripts/run_stage2.py`) originally fired on a single risk-crossing frame with no confirmation debounce. We added a `confirm_frames` option (default 3) requiring the risk to stay above the alert band for 3 consecutive frames before firing — evaluated on leakage-safe pooled out-of-fold risk scores across all 71 videos (see `results/stage2_confirm_sweep.md`), it moves the false-positive rate **67% → 53%** with detection rates essentially unchanged (70%→68% Normal→Crowdy, 17.5%→17.5% Crowdy→Panic) and only ~2-4 extra sampled frames of latency. Going higher (4-6 frames) gives no further false-positive improvement and starts costing Crowdy→Panic detection — so 3 is the sweet spot, not a fully-solved problem.
**Caveat on this number:** treat it as a promising internal signal, not a validated result. With only 15 stable-class videos in the pool, 67%→53% is just **2 videos** flipping outcome. And like the tables above, the "40 sequences" used for detection-rate stats are synthetic pairings built by cycling through a small pool of source videos (13 Normal / 34 Crowdy / 23 Panic) — several sequences reuse the same underlying video, so the effective independent sample size is well under 40. No confidence intervals were computed, and `confirm_frames=3` was picked because it looked best among the 5 values swept on this same data, with no separate confirmatory check. This same caveat — synthetic sequences built from a small, reused video pool — applies to the Table 1/2 numbers above too, just less severely (their Safe/Risk pooling spans more source videos).

**A real Stage 1 fix, found while diagnosing Stage 2's false positives:** `data/kept_panic.txt` is a manually-curated allowlist (from a label audit that flagged blank outro cards, news-desk interviews, and other non-panic contamination in the raw 281-clip Panic class — see `data/README.md`) that was **never actually applied anywhere in the pipeline** until now (`stampede/dataset.py`'s `inventory()` and new `filter_kept_panic()`, wired into `scripts/train_stage1.py`). One confirmed example: `VID-20251025-WA0003_clip_003.mp4`, audited as *"OUTRO CARD... no people"*, still trained as Panic with 15/30 zero-person frames. Applying the filter (281 → 212 Panic clips) is a real, clean win — **+1.1% macro-F1, and Panic precision *and* recall both improved together** (+2.9pt / +1.6pt — better than the CSRNet swap's own gain, see `results/kept_panic_filter_comparison.md`). It did *not*, however, fix the specific Stage 2 false positives it was aimed at (spot-checked: risk on one literal black frame dropped from 0.874 → 0.479, essentially cured, but the video-level alert outcome didn't flip because these are long videos with many separate excursions) — so the fix is shipped for Stage 1's sake, not as a Stage 2 solution.

**We also tried to push the false-positive rate down further and hit what looks like a real ceiling for Stage-2-only fixes.** Diagnosing the 8 videos still false-alerting at `confirm_frames=3` showed the excursions aren't brief noise a longer debounce could catch — one Normal video spends 431 consecutive frames above the alert threshold, and another's *mean* risk score (0.525) is already above the 0.50 alert line. We added a relative-rise gate (`NC_MIN_RISE`/`CP_MIN_RISE` in `stampede/config.py`, off by default) requiring risk to climb a minimum amount above its own recent baseline, not just cross a fixed cutoff — reasoning that a genuine transition should rise sharply relative to itself while a systematically-elevated-but-fluctuating video shouldn't. It didn't work: swept up to a rise requirement 1.5x the alert band's own width, nothing changed (see `results/stage2_baseline_margin_sweep.md`) — the false-positive videos rise by just as much, relative to their own baseline, as genuine transitions do. Stage 2 cannot tell them apart using the risk timeline alone; the false positives are inherited from specific Stage 1 blind spots on specific videos, not a Stage 2 debouncing or thresholding shortcoming. **53% is likely the floor without either improving Stage 1's accuracy on that content or giving Stage 2 richer per-frame input than a single scalar risk score.**

## What doesn't work well yet (and why)

**1. People running/fleeing in open, spread-out areas (like Times Square).**
Our model mainly judges danger by how packed a crowd is. A few people sprinting across an open plaza doesn't look "crowded" — so the model misses it, even though it's genuinely dangerous.

**2. Extremely dense crowds filmed from far away or in low quality (like Love Parade).**
When people are packed too tightly and the camera is too far/blurry, our person-detector can't tell individuals apart anymore — it undercounts, and the crowd looks less dangerous than it really is.

**The root cause of both problems:** we only have 23 unique Panic videos to learn from, and "Panic" actually contains two very different behaviors — people crushed together (dense) and people fleeing (sparse) — that look almost opposite to each other. More Panic footage, especially fleeing/running footage, is the #1 thing that would fix this.

## Things we tried that DIDN'T work (important — read this before retrying these ideas!)

We tested several fixes and are reporting honestly what failed, so nobody repeats the same dead ends:

- **Adding motion/speed features to catch fleeing people** — no improvement. Turns out fleeing crowds move together in one direction (not scattered like we assumed), and our model trusts crowd density way more than motion anyway.
- **Removing "messy" Panic clips to keep only clean crush footage** — made things worse. We already had too little Panic data; removing more of it hurt everything.
- **Adding a rule that says "extreme density always = danger"** — seemed promising at first, but it backfired: in our data, the densest crowds are actually calm festival crowds (labeled Safe), not panic. This rule would've caused more false alarms than it prevented. Reverted.
- **Trying to sharpen/zoom the video to see individual people better in blurry footage** — didn't help. The footage itself doesn't have enough real detail to recover; you can't zoom into information that isn't there.
- **Upweighting Panic-flight clips during training** (`data/panic_sublabels.csv` already has a manual crush-vs-flight audit; we tried scaling flight frames' training weight to match crush's, via `stampede/sublabels.py` + a new `sample_weight` param on `Stage1Model.train`) — made things slightly worse, including on flight recall itself (32% → 29%, see `results/panic_flight_upweight.md`). Only 4 of 23 Panic source videos carry the flight sub-label, so upweighting concentrates training on ~2-3 videos' idiosyncratic look rather than a generalisable flight signal — it can't manufacture video diversity that isn't there. The `sample_weight` plumbing is kept (generic and useful elsewhere) but defaults off (`panic_flight_upweight=False`).

## How to run it

**Setup (needed for both options):**
```bash
git clone https://github.com/Harinihegde/Janasamvedana.git
cd Janasamvedana
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**You'll need to get separately** (too large to include in this repo):
- The video dataset (see `data/README.md` for details).
- **The exact YOLO weights we used: `best_combined.pt`** — a *custom, single-class
  ('person')* YOLO detector fine-tuned at imgsz 512 (~52 MB;
  SHA-256 `4a80ac54d8a129edabb0a1a35ed183e68ff4f0622682b2b8e878673e1fd72dea`).
  This is not "any" person model — this specific file produced the numbers above.
  → **[DOWNLOAD LINK — TO BE ADDED]**
- (Optional) the external test dataset used for `gen_test.py`.

There are **two ways to run this**:

### Option A — Use precomputed outputs (fast; recommended for teammates)
Skips the slow (~1–1.5 h) feature-extraction step by downloading our precomputed
per-frame features + trained Safe/Risk model.

1. Download **`precomputed_outputs.zip`** → **shared in whatsapp**
2. Unzip it at the repo root (it creates `outputs/`, `outputs_compression2/`,
   and `outputs_binary/`).
3. Go straight to the evaluation/results scripts — **no dataset or YOLO weights
   needed** for these:
```bash
python scripts/train_stage1.py --output outputs          # 4-category model (~2 min from precomputed features)
python scripts/run_stage2.py   --output outputs          # early-warning alerts
python scripts/run_binary.py   --features outputs/frame_features.csv --output outputs_binary
python scripts/run_lovo.py     --oof-cache outputs/lovo_oof.csv   # strictest (leave-one-video-out) test
```
The zip contains the 21-feature `frame_features.csv` (the expensive artifact)
and the trained Safe/Risk model; the scripts above only *train* quick models
from those features (minutes) — feature extraction is what you skip.
(`gen_test.py` additionally needs the external `archive.zip`.)

### Option B — Run the full pipeline from scratch
Needs the dataset **and** `best_combined.pt`.
```bash
# Step 1: Extract features from all videos (~1-1.5 hours, one-time)
python scripts/extract_features.py --dataset /path/to/videos --weights /path/to/best_combined.pt --output outputs

# Step 2: Train the 4-category model
python scripts/train_stage1.py --output outputs

# Step 3: Run the early-warning alert system
python scripts/run_stage2.py --output outputs

# Step 4: Train the simpler Safe/Risk model + strictest test
python scripts/run_binary.py --features outputs/frame_features.csv --output outputs_binary
python scripts/run_lovo.py --oof-cache outputs/lovo_oof.csv

# Step 5 (optional): Test on real-world unseen incidents
python scripts/gen_test.py
```

## Notes for teammates

- Don't commit anything in `outputs*/` folders — those get regenerated automatically when you run the scripts above.
- The `results/` folder has our official reference numbers — compare your own runs against these.
- Feature extraction is the slow part (~1-1.5 hrs) but only needs to run once; everything after is fast.
- `scripts/experiments_reverted/` contains the failed experiments above, with more detail — read before retrying any of them.

## What would make this better (future work)

1. **More Panic videos, especially people fleeing/running** — this is the single biggest lever left. We've tested several technical fixes; none beat simply having more real examples.
2. **Split "Panic" into two separate labels** (crush vs. flight) once there's enough footage of each to train on separately.
3. **A proper crowd-density model** (like CSRNet) for handling very dense, blurry, distant footage that our current person-detector can't count accurately. **Update:** `stampede/csrnet.py` now implements this — a real CSRNet (Li et al., CVPR 2018) that can replace the old edge-density heuristic in `Detector`'s dense-crowd fallback, via `extract_features.py --csrnet-weights /path/to/checkpoint.pth`. Loading the official ShanghaiA-pretrained checkpoint (`gdown https://drive.google.com/uc?id=1Z-atzS5Y2pOd-nEWqZRVBDMYJDreGWHH`, SHA-256 `68383c1053be371ad54b0061ab99fed08c2bfff39de73937f2a4388041ab0128`, from [leeyeehoo/CSRNet-pytorch](https://github.com/leeyeehoo/CSRNet-pytorch) — no LICENSE file in that repo, so treat the weights as research-use-only, not redistributable) and spot-checking it against 5 real fallback-triggered frames confirmed it moves in the right direction on both known failure modes: on a genuinely packed festival crush frame (YOLO found 2 boxes, the old heuristic guessed 23, CSRNet estimated 108 — much closer to the ~100+ people visible), and on a busy-but-empty news-broadcast frame that the heuristic falsely flagged as dense (heuristic guessed 12 people where there were ~0, CSRNet said 4). We re-extracted all 1,330 clips with `--csrnet-weights` and retrained/re-evaluated (`results/csrnet_comparison.md`): on the pooled 5-fold out-of-fold clip-level result (n=281 Panic, n=159 Crowdy), **Panic recall moves 0.580 → 0.605** (~7 more clips caught) — the intended direction — but Crowdy recall drops slightly (0.623 → 0.610) and Panic precision drops slightly too, so overall macro-F1 only moves +0.006 and CV accuracy +0.003. **Honest verdict: a real but small, mixed improvement, not a breakthrough** — kept as opt-in (`--csrnet-weights`) rather than the new default, since the ~3x slower extraction isn't clearly justified by this size of gain on the current dataset.
4. **Reduce Stage 2's false-alarm rate** — currently the biggest blocker to real deployment; needs either better underlying risk scores (fixing Stage 1's blind spots) or a smarter alerting rule.

## License
MIT — see LICENSE file.
