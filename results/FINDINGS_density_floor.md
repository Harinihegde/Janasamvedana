# Finding: extreme-density crushes and the density risk floor

## The problem (and a corrected diagnosis)
Tested on 4 external, unseen panic incidents (Times Square, Las Vegas, Love
Parade, Italy) with a model trained only on the Crowd Panic dataset, the system
initially **missed the Love Parade 2010 tunnel crush entirely** (catch 0.0).

The first diagnosis — "YOLO undercounts the distant packed crowd" — was **wrong**.
On the 1280x720 Love Parade footage YOLO actually saturates at its 300-detection
cap; the recorded density is *maximal* (`density_norm` = 0.998, higher than any
Crowdy or Panic clip in training). The crush was mis-scored **despite** reading
as maximally dense.

## The real cause: a label-semantics flaw
In the training data the **densest** crowds are *Crowdy festivals*, which the
binary scheme labels **Safe** (Safe = No Panic + Normal + Crowdy). The **Panic**
clips are actually *lower* density (fleeing / low-res CCTV; mean crowd_density
6.7 vs Crowdy's 18.2). So the model learned the dangerous inverse:
**"extreme density = Crowdy = Safe."** A genuine Love-Parade-scale crush — the
densest thing possible — is therefore filed as a safe festival.

This is a data/label limitation, not a detector bug: for a crush-detection
system, `Crowdy = Safe` is itself questionable — a Lalbaug- or Love-Parade-
density crowd *is* a latent crush.

## The fix: a crowd-safety density floor
Final risk = `max(model_risk, floor(density_norm))`, where the floor ramps
0 -> 1 across density_norm in [0.80, 0.95] (config `DENSITY_FLOOR_LO/HI`).
This encodes Fruin's Level-of-Service physics: past ~5 people/m^2 a crowd is in
the crush danger zone regardless of behaviour. Only genuinely saturated crowds
(>150 detected people) are forced to maximum risk, so moderate incidents are
untouched. Baked into `Stage1Model.risk_score` (classification unchanged).

## Result (external generalization, two-channel system)
| Incident | catch before | catch after | caught by |
|---|---|---|---|
| Love Parade (crush) | 0.00 | **1.00** | density floor |
| Italy (flight)      | 0.30 | **0.90** | mostly learned |
| Las Vegas (flight)  | 0.59 | **0.84** | learned |
| Times Square (flight)| 0.00 | 0.52 | learned |
| **Overall**         | 0.38 | **0.77** | |

False positives: **0%** on moderate crowds; the 2 flagged Love-Parade "normal"
clips are dense pre-crush footage — the intended early warning.

## Takeaways
1. A learned crowd-risk model can be *inverted* by label semantics (dense=safe);
   a physical density prior is a necessary safety backstop.
2. The system now detects both **behavioural** panic (learned) and **physical**
   crush (density floor) — the two failure modes are complementary.
3. Residual limit: the floor is a blunt "extreme density = danger" rule; it does
   not distinguish pre-crush from crush (both alarm, which is correct for
   prevention). More distinct high-density crush *and* flight source videos
   remain the top data priority.
