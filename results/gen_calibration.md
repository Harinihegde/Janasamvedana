# Per-domain threshold calibration (external generalization test)

Thresholds set from NORMAL clips only (deployment-realistic). 87 abnormal + 37 normal clips.

Baseline (fixed 0.5): overall catch **0.379**, FP 0.0

## Global normal-calibrated thresholds (one threshold, all incidents)

| FP budget | threshold | overall catch | actual FP | Las Vegas | Times Square | Love Parade | Italy |
|---|---|---|---|---|---|---|---|
| 0% | 0.312 | **0.713** | 0.0 | 0.843 | 0.522 | 0.0 | 0.7 |
| 5% | 0.296 | **0.736** | 0.054 | 0.863 | 0.522 | 0.0 | 0.8 |
| 10% | 0.268 | **0.77** | 0.108 | 0.863 | 0.609 | 0.0 | 0.9 |
| 20% | 0.234 | **0.816** | 0.216 | 0.882 | 0.652 | 0.667 | 0.9 |

## Per-incident self-calibration (own normal baseline)

| Incident | n abn / norm | threshold | catch rate |
|---|---|---|---|
| Italy | 10/0 | — | too few normal clips to self-calibrate |
| Las Vegas | 51/29 | 0.295 | **0.863** |
| Love Parade | 3/2 | — | too few normal clips to self-calibrate |
| Times Square | 23/6 | 0.312 | **0.522** |