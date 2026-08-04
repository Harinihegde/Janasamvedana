# Reverted experiments (documented negative results)

These scripts are kept for transparency. Each was tried rigorously and **did not
make it into the final model** — they're negative results that shaped the design,
not part of the pipeline. (Some import symbols that were later removed, so they
may not run as-is; they are preserved as a record.)

| Script | What it tried | Outcome |
|---|---|---|
| `run_flight.py`, `flow_extra.py` | Flight/scatter motion features (flow-direction divergence + speed) to catch sparse flight-panic | **No improvement** (Δ Panic-F1 ≈ 0.00). Flight is directionally *coherent*, so divergence was the wrong signal; density dominates. |
| `run_crush_lovo.py` | Prune Panic to crush-only clips to sharpen the class | **Hurt** — LOVO crush recall 0.69 → 0.26. Shrinking an already-scarce class removes signal. |
| `run_floor_lovo.py`, `gen_final.py` | A crowd-density "risk floor" (force high risk on extreme density) | **Invalid, reverted.** Threshold was test-informed; under LOVO it flagged ~3–4× more Safe (dense Crowdy festivals) than Risk clips (precision 0.81 → 0.67). `gen_final.py` references the removed floor constants. |
| `run_static_wide_fix.py` | Tiling / higher YOLO input resolution to fix undercounting of distant dense crowds | Did **not** recover counts on low-resolution footage. |

See the top-level `README.md` ("What was tried and didn't work") and the reports
in `results/` (`flight_report`, `crush_lovo_report`, `floor_lovo`,
`static_wide_fix`) for the full numbers.
