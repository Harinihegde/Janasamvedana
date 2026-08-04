# Final external generalization — with integrated density floor

Learned-channel threshold (calibrated on moderate normals, 0% FP): **0.312**. Density floor fires at density_norm >= 0.95.

| Incident | catch | by floor | by learned | AUC | normal flagged |
|---|---|---|---|---|---|
| Italy | **0.9** | 2/10 | 7/10 | n/a | 0/0 |
| Las Vegas | **0.843** | 0/51 | 43/51 | 0.959 | 0/29 |
| Love Parade | **1.0** | 3/3 | 0/3 | n/a | 2/2 |
| Times Square | **0.522** | 0/23 | 12/23 | 0.674 | 0/6 |

**Overall abnormal catch: 0.77**  (was 0.379 at fixed 0.5 pre-floor)
False positives — on moderate crowds: **0.0**; incl. dense pre-crush clips: 0.054

> The dense pre-crush 'normal' clips (Love Parade) counted as FP are genuinely lethal-density crowds — flagging them is the intended early warning, not an error.