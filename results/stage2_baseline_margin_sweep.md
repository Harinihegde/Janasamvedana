# Stage 2 relative-rise gate sweep (4-class production model)

Pooled 5-fold leakage-safe out-of-fold risk scores across all 71 videos, `confirm_frames=3` (current default) fixed, sweeping the new `nc_min_rise`/`cp_min_rise`/`baseline_window` relative-rise gate on top of it.

| baseline_window | min_rise (NC/CP) | Normal->Crowdy detect | Crowdy->Panic detect | False-positive rate | stable videos alerting |
|---|---|---|---|---|---|
| 90 | 0.00/0.00 | 70% | 20% | 53% | 8/15 |
| 60 | 0.10/0.10 | 70% | 20% | 53% | 8/15 |
| 90 | 0.10/0.10 | 70% | 20% | 53% | 8/15 |
| 150 | 0.10/0.10 | 70% | 20% | 53% | 8/15 |
| 90 | 0.15/0.15 | 70% | 20% | 53% | 8/15 |
| 150 | 0.15/0.15 | 70% | 20% | 53% | 8/15 |
| 90 | 0.20/0.20 | 35% | 18% | 47% | 7/15 |
| 150 | 0.20/0.20 | 45% | 18% | 47% | 7/15 |

(Row 1, `min_rise=0.00`, is the gate switched off - i.e. current shipped
behavior - and is the baseline the other rows are compared against. Its
70%/20%/53% differs slightly from `results/stage2_confirm_sweep.md`'s
67%/18%/53% published numbers for the same `confirm_frames=3` config; that's
run-to-run noise in `stratified_group_folds`' fold assignment, not a change -
same order of magnitude, same conclusion.)

## Honest read

**This doesn't work either, and the reason is more fundamental than "wrong
parameters": Stage 2 cannot tell these false positives apart from real
transitions using the risk timeline alone.** Up to `min_rise=0.15` (1.5x the
alert band's own width), *nothing* changes - all 8 previously-alerting stable
videos still clear the bar. At `min_rise=0.20` one video finally gets
suppressed (8/15 -> 7/15, FP 53% -> 47%), but Normal->Crowdy detection on the
*genuine* synthetic transitions collapses from 70% to 35-45% in the same
pass - roughly two real detections lost for every one false alarm removed.

Why: a real Normal->Crowdy transition in the synthetic test data is a hard
splice between two constant-class clips, smoothed into a sharp rise from a
low baseline to a high plateau. The false-positive videos identified during
diagnosis (e.g. one Normal video spends 431 consecutive frames above the
alert threshold; another's mean risk is 0.525, already above the 0.50 line)
show the *same shape* - a sustained rise from a low local baseline to a high
plateau - just because Stage 1 is wrong about that content, not because
anything is escalating. Relative-rise and absolute-threshold gates both only
look at the shape of the risk curve, and both shapes are the same. There is
no risk-timeline-only heuristic left to try here; distinguishing them needs
information Stage 2 doesn't have (e.g. Stage 1 being more accurate on that
specific content, or Stage 2 seeing more than a single scalar risk score per
frame).

**Recommendation:** don't ship this gate (`NC_MIN_RISE`/`CP_MIN_RISE` stay at
`0.0`, the code path is opt-in via the `thresholds` dict for future
experiments). Combined with `results/stage2_confirm_sweep.md` already showing
`confirm_frames` plateaus at the same 53%, **53% is likely a real ceiling for
Stage-2-only fixes on top of the current Stage 1 model** - the false positives
are inherited from specific Stage 1 blind spots on specific videos, not a
Stage 2 debouncing or thresholding shortcoming.
