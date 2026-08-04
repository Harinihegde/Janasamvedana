# Base (12 features) vs +Compression (19 features)

4-class RandomForest, identical leakage-safe split & 5-fold StratifiedGroupKFold CV. Compression adds 7 occlusion/crush features. Metrics are clip-level, from CV out-of-fold predictions.

## Headline

| Metric | 12 Features | 19 Features (+Compression) | Δ |
|---|---|---|---|
| Panic F1 | 0.473 | 0.535 | +0.062 |
| Panic recall | 0.434 | 0.544 | +0.110 |
| Panic precision | 0.519 | 0.526 | +0.007 |
| CV macro-F1 (OOF) | 0.550 | 0.645 | +0.095 |
| CV accuracy (OOF) | 57.3% | 65.8% | +0.085 |
| CV macro-F1 (fold mean) | 0.468 | 0.551 | +0.083 |

## Per-class F1 (CV out-of-fold)

| Class | 12 Features | 19 Features | Δ |
|---|---|---|---|
| No Panic | 0.551 | 0.665 | +0.114 |
| Normal | 0.641 | 0.714 | +0.073 |
| Crowdy | 0.536 | 0.667 | +0.131 |
| Panic | 0.473 | 0.535 | +0.062 |

## Crowdy↔Panic confusion (CV OOF) — the pair we targeted

| | 12 Features | 19 Features |
|---|---|---|
| Panic→Panic (correct) | 122 | 153 |
| Panic→Crowdy (miss) | 40 | 32 |
| Crowdy→Panic (false) | 45 | 46 |
| Crowdy→Crowdy (correct) | 78 | 97 |

## Feature importances (19-feature model) — compression features **bold**

| Rank | Feature | Importance |
|---|---|---|
| 1 | **bbox_overlap_ratio** | 0.119 |
| 2 | direction_consistency | 0.107 |
| 3 | **bbox_area_mean** | 0.076 |
| 4 | flow_mag_mean | 0.076 |
| 5 | **detection_confidence_mean** | 0.065 |
| 6 | **spatial_density_mismatch** | 0.062 |
| 7 | **bbox_area_variance** | 0.055 |
| 8 | density_norm | 0.054 |
| 9 | person_count | 0.052 |
| 10 | crowd_density | 0.050 |
| 11 | trajectory_dispersion | 0.043 |
| 12 | **detection_confidence_std** | 0.042 |
| 13 | flow_mag_var | 0.040 |
| 14 | stop_go | 0.038 |
| 15 | motion_instability | 0.033 |
| 16 | velocity_var | 0.027 |
| 17 | velocity_per_person | 0.027 |
| 18 | density_trend | 0.018 |
| 19 | **bbox_overlap_trend** | 0.017 |

Compression features in the top 10: **5 of 7** (ranks [1, 3, 5, 6, 7, 12, 19]).

## Stage 2 — Crowdy→Panic detection (held-out test)

| Threshold | 12 Features | 19 Features |
|---|---|---|
| Spec edge 0.8 | 0.0% | 20.0% |
| Calibrated | 60.0% | 45.0% |

## Verdict

Compression features **improve** Panic F1 (0.473 → 0.535, Δ +0.062) and move CV macro-F1 by +0.095. 5/7 compression features rank in the top 10 by importance.
