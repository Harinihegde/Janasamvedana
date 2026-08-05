# Stage 2 confirmation-frame sweep (4-class production model)

Pooled 5-fold leakage-safe out-of-fold risk scores across all 71 videos (not just the 3-video test split), evaluated with detect_alerts()'s new `confirm_frames` debounce at the spec thresholds (0.5 / 0.8).

| confirm_frames | Normal->Crowdy detect | latency | Crowdy->Panic detect | latency | False-positive rate | n_stable_videos |
|---|---|---|---|---|---|---|
| 1 | 70% | 0.1f | 18% | -1.9f | 67% | 15 |
| 3 | 68% | -1.6f | 18% | -3.9f | 53% | 15 |
| 4 | 68% | -2.6f | 12% | 0.4f | 53% | 15 |
| 5 | 68% | -3.6f | 12% | -5.6f | 53% | 15 |
| 6 | 68% | -4.6f | 12% | -6.6f | 53% | 15 |

## Caveats (read before citing these numbers)

- `n_stable_videos=15`: the 67%→53% false-positive move is only **2 videos** flipping outcome (10/15 → 8/15). Small absolute base.
- The 40 "sequences" per transition type are synthetic pairings cycled through a small pool of source videos (13 Normal / 34 Crowdy / 23 Panic for the two transition types) — several sequences reuse the same underlying video on one side, so the effective independent sample size is well under 40.
- No confidence intervals or significance test were computed.
- `confirm_frames=3` was chosen because it looked best among the 5 values swept on this exact evaluation set — there is no separate confirmatory holdout it was checked against afterward.

Treat this as a promising internal signal worth keeping as the new default, not a validated result.
