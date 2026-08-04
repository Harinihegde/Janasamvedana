# Compression-ratio features — 4-class before/after (CV OOF, clip-level)

New features: ['hull_density', 'nn_distance_mean']  (ranks/importance: {'hull_density': (2, np.float64(0.0713)), 'nn_distance_mean': (16, np.float64(0.0291))})

## Crowdy<->Panic confusion block (rows=actual, cols=pred)

### Before (19 features)
```
      pred:Crowdy pred:Panic
  Crowdy      97        46
  Panic       32       153
```
### After (21 features)
```
      pred:Crowdy pred:Panic
  Crowdy      99        45
  Panic       32       163
```

## Per-class & overall (Crowdy / Panic are the target)

| Metric | Before (19) | After (21) | Δ |
|---|---|---|---|
| Crowdy precision | 0.735 | 0.728 | -0.007 |
| Crowdy recall | 0.610 | 0.623 | +0.013 |
| Crowdy F1 | 0.667 | 0.671 | +0.005 |
| Panic precision | 0.526 | 0.566 | +0.040 |
| Panic recall | 0.544 | 0.580 | +0.036 |
| Panic F1 | 0.535 | 0.573 | +0.038 |
| Overall accuracy | 0.658 | 0.680 | +0.022 |
| Overall macro-F1 | 0.645 | 0.664 | +0.019 |
| Overall weighted-F1 | 0.657 | 0.678 | +0.021 |