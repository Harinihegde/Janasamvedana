# Flight/scatter motion features — 4-class before/after (CV OOF)

New: ['flow_dir_divergence', 'flight_score']  ranks/importance: {'flow_dir_divergence': (21, np.float64(0.018)), 'flight_score': (17, np.float64(0.0267))}

## MOTION_RUNNING subset (25 clips, all actual Panic)

| | Before (21) | After (23) |
|---|---|---|
| Recall (predicted Panic) | 0.000 | 0.000 |
| Predicted-label breakdown | {'Normal': 16, 'No Panic': 9} | {'Normal': 17, 'No Panic': 8} |

## Overall (CV OOF, clip-level)

| Metric | Before (21) | After (23) | Δ |
|---|---|---|---|
| Panic precision | 0.566 | 0.558 | -0.008 |
| Panic recall | 0.580 | 0.580 | +0.000 |
| Panic F1 | 0.573 | 0.569 | -0.004 |
| Crowdy F1 | 0.671 | 0.673 | +0.002 |
| Accuracy | 0.680 | 0.677 | -0.003 |
| Macro-F1 | 0.664 | 0.661 | -0.003 |