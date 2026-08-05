# kept_panic.txt filter: before/after (Stage 1 + Stage 2)

Pooled 5-fold leakage-safe out-of-fold results, comparing the raw 281-clip Panic class against the label-audit-filtered set (`data/kept_panic.txt`, dropping confirmed contamination clips).

## Stage 1 (4-class classifier)

| Metric | Before (raw 281 Panic) | After (kept_panic filtered) | Delta |
|---|---|---|---|
| CV clip accuracy | 0.683 | 0.692 | +0.009 |
| CV clip macro-F1 | 0.669 | 0.680 | +0.011 |
| Panic precision | 0.582 | 0.611 | +0.029 |
| Panic recall | 0.569 | 0.585 | +0.016 |
| Panic F1 | 0.576 | 0.598 | +0.022 |
| Panic support (n clips) | 281 | 212 | -69 |

## Stage 2: the 8 known false-positive videos (confirm_frames=3)

- Before: 8/8 still alerting
- After: 8/8 still alerting
- Fixed by this change: 0 (relative to the 0 fixed by definition before)
- Still alerting after the fix: ['VID-20250924-WA0032', 'VID-20250924-WA0029', 'VID-20250924-WA0031', 'VID-20251025-WA0006', 'VID-20250924-WA0036', 'VID-20250924-WA0034', 'VID-20250924-WA0030', 'VID-20250924-WA0035']

- Overall stable-video false-positive rate: 53% -> 60% (n=15 stable videos)

## Honest read

**Stage 1 got a real, clean improvement — the best single change this
project has made (better than the CSRNet swap's +0.6% macro-F1): +1.1%
macro-F1, and Panic precision AND recall both moved up together (+2.9pt /
+1.6pt), not the usual precision/recall tradeoff.** This is a genuine
data-quality fix, not a modeling trick, and it's very likely to hold up
since it's just removing labeled-Panic clips that a manual audit already
confirmed contain zero people.

**But it did not fix the specific Stage 2 false positives it was aimed at,
and the overall FP rate moved the wrong way (53% -> 60%, though with only 15
stable videos that's 1 video's worth of noise).** Spot-checking confirmed the
underlying mechanism *is* real: on the literal black frame in
`VID-20251025-WA0006_clip_329.mp4` that peaked at risk 0.874 before the fix,
risk dropped to **0.479** after - essentially cured, and now below even the
Normal->Crowdy threshold. But the video-level alert outcome didn't flip,
because these are long videos (WA0006 alone is 12,502 frames) with *many*
separate excursions - fixing one moment doesn't fix a video's overall
trajectory. And two of the three checked frames (`WA0034`, `WA0029`) only
partially improved (0.878->0.630, and 0.821 - barely changed) - suggesting
those specific videos have a second, still-uninvestigated contributing
factor (plausibly the "large close-up occluding figure" pattern noted in the
same diagnosis for `WA0036`/`WA0032`/`WA0031`, which is unrelated to blank-
frame contamination and wasn't addressed by this fix).

**Recommendation: keep the fix (it's a real Stage 1 win, no reason not to
ship it) but don't expect it alone to move the Stage 2 headline number.**
The false positives have more than one cause; this closes one of them
partially. The close-up-occlusion pattern is the next lead worth digging
into, and is a good next-step handoff item.
