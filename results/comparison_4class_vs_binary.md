# 4-Class vs Binary — Crowd-Risk Classification

Same 12 features, same leakage-safe StratifiedGroupKFold CV (5 folds), same RandomForest. Metrics are clip-level.

## Stage 1 (primary metric: macro-F1, from CV out-of-fold)

| Metric | 4-Class | Binary (Safe vs Stampede Risk) |
|---|---|---|
| CV accuracy (mean ± std) | 62.5% ± 14.4% | 70.4% ± 32.0% |
| CV macro-F1 (mean ± std) | 0.468 ± 0.115 | 0.527 ± 0.231 |
| CV weighted-F1 (mean) | 0.643 | 0.696 |
| OOF accuracy | 57.3% | 81.7% |
| OOF macro-F1 | 0.550 | 0.666 |
| OOF weighted-F1 | 0.568 | 0.796 |

## Per-class F1 (CV out-of-fold)

| Class | 4-Class F1 | | Binary class | Binary F1 |
|---|---|---|---|---|
| No Panic | 0.551 | | Safe | 0.891 |
| Normal | 0.641 | | Stampede Risk | 0.441 |
| Crowdy | 0.536 | |  |  |
| Panic | 0.473 | |  |  |

## Danger-class detection (the safety-critical number)

| | 4-Class: *Panic* | Binary: *Stampede Risk* |
|---|---|---|
| Precision | 0.519 | 0.623 |
| Recall | 0.434 | 0.342 |
| F1 | 0.473 | 0.441 |

## Stage 2 escalation (held-out test)

| | 4-Class (Crowdy→Panic) | Binary (Safe→Stampede Risk) |
|---|---|---|
| Detection @ spec edge 0.8 | 0.0% | 30.0% |
| Detection @ calibrated edge | 60.0% | 80.0% (edge 0.49) |
| Median latency (calibrated, frames early) | 10 | 12 |

## Verdict

**Binary wins the headline metrics** — OOF macro-F1 0.666 vs 0.550 (Δ 0.116) and accuracy 81.7% vs 57.3% — and its single-transition Stage 2 escalation detects far better.

**But read the fine print for a safety system:**
- Binary accuracy (81.7%) is barely above the trivial *always-Safe* baseline (78.9%): the model is only +2.9 pp better than never raising an alarm. Most of the binary 'win' is the easy majority Safe class (F1 0.891).
- On the metric that actually matters — **catching Panic** — the 4-class model has *higher recall* (0.434 vs 0.342); binary misses 66% of real Panic clips. Danger-class recall winner: **4-Class**.
- Binary CV is high-variance (fold accuracy spans 9%–98%) because some folds are Panic-heavy — a symptom of few unique Panic videos.

**Recommendation:** use **binary** if the goal is overall discrimination, deployment simplicity, and a cleaner Stage 2 escalation signal; keep **4-class** if the priority is maximising *recall of Panic* (fewest missed stampedes). On this leakage-free data neither is strongly separable, so the choice is an operating-point decision, not a clear technical win.
