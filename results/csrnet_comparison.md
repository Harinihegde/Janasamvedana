# CSRNet fallback vs. edge-density heuristic (Stage 1, 4-class)

Full re-extraction of all 1,330 clips with `extract_features.py --csrnet-weights`
(official ShanghaiA-pretrained CSRNet, see README), retrained and evaluated the
same way as the baseline (`outputs/`). Compared at the pooled 5-fold
out-of-fold clip level (n=1,330, the stable/trustworthy number - not the
409-clip single holdout).

| Metric | Baseline (heuristic) | CSRNet | Delta |
|---|---|---|---|
| CV clip accuracy (mean) | 0.7179 | 0.7205 | +0.0026 |
| CV clip macro-F1 | 0.5657 | 0.5714 | +0.0057 |
| No Panic F1 (n=364) | 0.672 | 0.674 | +0.002 |
| Normal F1 (n=526) | 0.740 | 0.744 | +0.004 |
| Crowdy F1 (n=159) | 0.671 | 0.664 | -0.007 |
| Panic F1 (n=281) | 0.573 | 0.579 | +0.006 |
| **Panic recall (n=281)** | **0.580** | **0.605** | **+0.025 (~7 clips)** |
| Panic precision (n=281) | 0.566 | 0.556 | -0.010 |
| Crowdy recall (n=159) | 0.623 | 0.610 | -0.013 |

## Honest read

**This is a real but small, mixed improvement - not a breakthrough.** Panic
recall moved in exactly the direction the qualitative 5-frame check predicted
(CSRNet gives much higher, more plausible counts on genuinely dense/occluded
crush frames than the old heuristic) - about 7 more Panic clips get caught
out of 281. But it came with a small precision cost on Panic and a small
recall cost on Crowdy, so overall macro-F1 only moved +0.006 and CV accuracy
+0.003.

Why the effect is muted despite the fallback firing on ~22-23% of Panic/No
Panic *frames*: Stage 1 predicts per-frame then takes a majority vote per
*clip*, so a clip only flips its prediction if enough of its frames' features
change enough to shift the vote - a handful of fallback-triggered frames in an
80-frame clip isn't always decisive. The person-count/density features feed
into a RandomForest alongside 19 other features (motion, compression,
detection-confidence, etc.), so a better single feature doesn't dominate.

**Recommendation:** keep CSRNet as opt-in (`--csrnet-weights`), not the
default - the runtime cost (~3x slower extraction) isn't clearly justified by
this small a gain on the current dataset/label scheme. It's more likely to
matter if paired with more genuinely extreme-density footage (Love Parade-like
sources) than the current dataset mostly contains.
