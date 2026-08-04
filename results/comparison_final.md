# Final model comparison — 4-Class vs Binary, 12 vs 19 features

Identical leakage-safe split & 5-fold CV. Danger class = Panic (4-class) / Stampede Risk (binary). Clip-level, CV out-of-fold.

| Metric | 4-Class (12) | 4-Class (19) | Binary (12) | Binary (19) |
|---|---|---|---|---|
| **Danger-class F1** | 0.473 | 0.535 | 0.441 | 0.621 |
| **Danger-class recall** | 0.434 | 0.544 | 0.342 | 0.520 |
| Danger-class precision | 0.519 | 0.526 | 0.623 | 0.772 |
| CV macro-F1 (OOF) | 0.550 | 0.645 | 0.666 | 0.770 |
| CV accuracy (OOF) | 57.3% | 65.8% | 81.7% | 86.6% |
| Stage 2 detect (spec 0.8) | 0.0% | 20.0% | 30.0% | 35.0% |
| Stage 2 detect (calibrated) | 60.0% | 45.0% | 80.0% | 80.0% |

## Binary: 12 vs 19 features (the requested delta)

| Metric | Binary (12) | Binary (19) | Δ |
|---|---|---|---|
| Stampede Risk F1 | 0.441 | 0.621 | +0.180 |
| Stampede Risk recall | 0.342 | 0.520 | +0.178 |
| Stampede Risk precision | 0.623 | 0.772 | +0.149 |
| CV macro-F1 (OOF) | 0.666 | 0.770 | +0.104 |
| CV accuracy (OOF) | 81.7% | 86.6% | +0.049 |
| Stage 2 detect (calibrated) | 80.0% | 80.0% | +0 pp |

## Verdict

- **Best danger-class recall** (fewest missed danger events): **4-Class (19)** (0.544).
- **Best danger-class F1**: **Binary (19)** (0.621).
- Binary accuracy must beat the *always-Safe* baseline (78.9%); Binary(19) OOF accuracy is 86.6% (+7.7 pp).
