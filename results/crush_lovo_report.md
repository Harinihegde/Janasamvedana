# Crush-focused retrain (4-class LOVO, 21 features)

Composition: 162 crush clips (16 videos) kept as Panic; 54 flight + 65 removed dropped.

## Per-class (LOVO OOF) — FULL Panic vs CLEAN (crush-only) Panic

| Metric | FULL | CLEAN | Δ |
|---|---|---|---|
| Crowdy precision | 0.744 | 0.688 | -0.056 |
| Crowdy recall | 0.604 | 0.736 | +0.132 |
| Crowdy f1-score | 0.667 | 0.711 | +0.045 |
| Panic precision | 0.558 | 0.447 | -0.111 |
| Panic recall | 0.616 | 0.259 | -0.356 |
| Panic f1-score | 0.585 | 0.328 | -0.257 |
| overall accuracy | 0.665 | 0.683 | +0.017 |
| overall macro_f1 | 0.651 | 0.615 | -0.036 |

## Crowdy↔Panic confusion (LOVO OOF)

FULL : {'Panic->Panic': 173, 'Panic->Crowdy': 27, 'Crowdy->Panic': 48, 'Crowdy->Crowdy': 96}

CLEAN: {'Panic->Panic': 42, 'Panic->Crowdy': 46, 'Crowdy->Panic': 21, 'Crowdy->Crowdy': 117}

## Isolation — crush-clip recall on the SAME clips (training effect only)

| | crush recall |
|---|---|
| trained with FULL Panic (flight incl.) | 0.691 |
| trained with CLEAN Panic (crush only)  | 0.259 |
| Δ | -0.432 |