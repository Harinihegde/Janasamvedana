Here's a simplified version, same info, easier for teammates to actually read and understand:

---

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

**Simple Safe vs Risk detection:**
- 86.3% accuracy (compared to 78.9% if you just always guessed "safe")
- When it says "risk," it's right 76% of the time, and catches about half (51%) of real dangerous situations

**Detailed 4-category detection (No Panic/Normal/Crowdy/Panic):**
- 68% accuracy overall
- Panic specifically: ~57% reliable, catches ~58% of real panic cases

**Testing on whole videos it's never seen (strictest test):**
- Almost perfect at recognizing "definitely safe" videos (90-100%)
- Only catches ~41% of full Panic videos — this is our biggest weakness

**Testing on 4 completely new, real-world incidents (Times Square, Las Vegas, Love Parade, Italy):**
- Without any adjustment: catches 38% of real danger, with zero false alarms
- After calibrating the alarm sensitivity per location (a normal real-world setup step): catches 71% of real danger, still zero false alarms

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

## How to run it

```bash
git clone https://github.com/Harinihegde/Janasamvedana.git
cd Janasamvedana
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**You'll need to get separately** (too large to include in this repo):
- The video dataset (see `data/README.md` for details)
- YOLO detection weights (any person-detection `.pt` file)
- (Optional) the external test dataset used for `gen_test.py`

**Then run these steps in order:**

```bash
# Step 1: Extract features from all videos (~1-1.5 hours, one-time)
python scripts/extract_features.py --dataset /path/to/videos --weights /path/to/best.pt --output outputs

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
3. **A proper crowd-density model** (like CSRNet) for handling very dense, blurry, distant footage that our current person-detector can't count accurately.

## License
MIT — see LICENSE file.
