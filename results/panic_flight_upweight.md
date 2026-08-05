# Panic crush/flight upweighting (Stage 1, 4-class)

Pooled 5-fold leakage-safe out-of-fold clip-level results (n=1,330 clips), comparing the baseline RandomForest against the same model trained with Panic-flight frames upweighted to match Panic-crush's aggregate training weight (see `stampede/sublabels.py`).

| Metric | Baseline | Upweighted | Delta |
|---|---|---|---|
| CV clip accuracy | 0.683 | 0.683 | +0.000 |
| CV clip macro-F1 | 0.669 | 0.665 | -0.004 |
| Panic precision | 0.582 | 0.578 | -0.004 |
| Panic recall | 0.569 | 0.555 | -0.014 |
| Panic F1 | 0.576 | 0.566 | -0.009 |

## Descriptive-only: Panic recall by crush/flight sub-label

**Not a validated per-subclass metric.** Only 4 of 23 Panic source videos carry a confirmed flight sub-label (39 clips vs. 164 crush clips) - far too few to trust a video-grouped fold's flight recall on its own. Shown for transparency only, not as a claim.

| Sub-label | n clips | Baseline recall | Upweighted recall |
|---|---|---|---|
| panic_crush | 223 | 65% | 64% |
| panic_flight | 41 | 32% | 29% |

## Honest read

**This didn't help - if anything, it made things slightly worse across the
board**, including on the exact sub-group it targeted: flight recall dropped
32% -> 29% rather than improving, crush recall was flat, and overall Panic
recall/F1/macro-F1 all moved down a little (Panic recall -0.014, macro-F1
-0.004).

Best explanation: with only 4 source videos carrying the flight sub-label,
each 5-fold CV split puts roughly 3 of them in training and ~1 in the held-out
fold. Upweighting flight frames concentrates the RF's attention on a
training set of maybe 2-3 flight *videos* (not clips - clips from the same
video are near-duplicates), so it risks overfitting to those specific
videos' idiosyncratic look (lighting, camera angle, footage source) rather
than learning a generalisable "flight" signal - and that overfit doesn't
transfer to the held-out flight video(s) in the test fold. This is the same
underlying data-scarcity problem the README already names as the root cause
of both known failure modes, just showing up in a new place: reweighting
can't manufacture diversity that isn't there in only 4 videos.

**Recommendation:** don't enable `panic_flight_upweight` by default - the
`sample_weight` plumbing in `stampede/classifier.py` is kept (it's small,
generic, and useful for other experiments) but the flag defaults to `False`
everywhere it's threaded through (`evaluate_holdout`, `cross_validate`). This
is now a second technique-level fix (after motion features, the density-floor
rule, and image sharpening - see README's "didn't work" section) that fails
to substitute for more Panic-flight source footage. Adding this class of fix
to that list.
