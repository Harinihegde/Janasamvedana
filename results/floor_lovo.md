# Density floor — leakage-safe LOVO evaluation (in-domain)

Binary Safe-vs-Risk, leave-one-video-out, clip-level.

| Metric | WITHOUT floor | WITH floor | Δ |
|---|---|---|---|
| AUC | 0.859 | 0.845 | -0.014 |
| risk_precision | 0.814 | 0.671 | -0.143 |
| risk_recall | 0.452 | 0.495 | +0.043 |
| risk_f1 | 0.581 | 0.57 | -0.011 |

Clips the floor newly flags as RISK (base<0.5 → ≥0.5): **39 SAFE** vs **12 RISK** → the floor flags more SAFE than RISK (hurts).

Threshold 0.80/0.95 was originally chosen with knowledge of the external test; this evaluation uses only in-domain held-out videos.