# LOVO: A5 RF top-12 — no-augmentation vs +augmentation (speed/crop/brightness)

Hysteresis k=3. Augmented variants of Panic+No-Panic added to TRAINING only; testing on original held-out videos. Leakage-safe (grouped by video_id).

## Stage 1 (LOVO clip-level, Stampede-Risk class)

| Metric | No-aug | +Aug | Δ |
|---|---|---|---|
| Accuracy | 0.863 | 0.835 | -0.028 |
| Risk precision | 0.773 | 0.587 | -0.187 |
| Risk recall | 0.498 | 0.747 | +0.249 |
| Risk F1 | 0.606 | 0.657 | +0.051 |
| **Panic video recall** | 0.409 | 0.773 | +0.364 |

## Stage 2 escalation — threshold sweep (detection / FP / median latency)

| Threshold | Det (no-aug) | Det (+aug) | FP (no-aug) | FP (+aug) | Lat (+aug) |
|---|---|---|---|---|---|
| 0.50 | 0.63 | **0.96** | 0.57 | 0.84 | +7f |
| 0.55 | 0.50 | **0.94** | 0.51 | 0.76 | +10f |
| 0.60 | 0.41 | **0.83** | 0.47 | 0.69 | -4f |
| 0.65 | 0.39 | **0.78** | 0.39 | 0.61 | -5f |
| 0.70 | 0.35 | **0.72** | 0.31 | 0.53 | -5f |