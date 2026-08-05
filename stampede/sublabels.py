"""Panic crush-vs-flight sub-labels from the manual audit (data/panic_sublabels.csv).

The 4-class Panic label bundles two behaviourally opposite crowd patterns: dense
crush and sparse flight. Only 4 of the 23 Panic source videos carry a confirmed
flight sub-label - far too few to evaluate as its own leakage-safe class (a
video-grouped CV fold can't meaningfully spread 4 videos). So instead of adding
a new class, we use the sub-label to upweight flight frames *within* Panic
training, so the classifier isn't trained on an almost entirely crush-shaped
picture of what "Panic" looks like.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

SUBLABELS_PATH = Path(__file__).resolve().parent.parent / "data" / "panic_sublabels.csv"
FLIGHT_SUBLABEL = "panic_flight"


def load_panic_sublabels(path: Path = SUBLABELS_PATH) -> dict[str, str]:
    """Map clip filename -> crush/flight sub-label. Empty dict if the file is missing."""
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {row["clip"]: row["sublabel"] for row in csv.DictReader(fh)}


def panic_flight_sample_weight(
    frame_df: pd.DataFrame, sublabels: dict[str, str] | None = None
) -> np.ndarray:
    """Per-row weight upweighting confirmed Panic-flight frames.

    All non-Panic rows and Panic rows without a confirmed flight sub-label get
    weight 1.0. Panic-flight rows are scaled so their total weight equals the
    total weight of the other Panic rows in ``frame_df``. Computed from
    ``frame_df`` itself (rather than a fixed constant) so it stays correct when
    called per CV fold, where the flight/crush ratio among that fold's training
    clips can differ from the full dataset.
    """
    sublabels = load_panic_sublabels() if sublabels is None else sublabels
    clip_name = frame_df["path"].map(lambda p: Path(p).name)
    is_flight = (frame_df["label"] == "Panic") & clip_name.map(sublabels).eq(FLIGHT_SUBLABEL)
    is_other_panic = (frame_df["label"] == "Panic") & ~is_flight

    weight = np.ones(len(frame_df), dtype=float)
    n_flight = int(is_flight.sum())
    n_other = int(is_other_panic.sum())
    if n_flight > 0 and n_other > 0:
        weight[is_flight.to_numpy()] = n_other / n_flight
    return weight
