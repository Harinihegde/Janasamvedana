"""Dataset inventory and **leakage-safe** train/test splitting.

The core correctness fix of this project lives here: clips such as
``VID-20250923-WA0008_clip_001.mp4`` and ``..._clip_002.mp4`` are cut from the
*same* source video and are therefore highly correlated. Splitting at the clip
level (as an earlier version of the pipeline did) leaks near-duplicate frames
across the train/test boundary and inflates accuracy. We instead group every
clip by its source ``video_id`` and split at the *video* level.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .config import CLASSES, RANDOM_STATE

# Matches the clip suffix for both naming schemes found in the dataset:
#   VID-20250923-WA0008_clip_001.mp4
#   WhatsApp Video 2025-09-23 at 18.49.36_79697ff5_clip_014.mp4
_CLIP_SUFFIX = re.compile(r"_clip_\d+$", re.IGNORECASE)


def video_id(path: str | Path) -> str:
    """Return the source-video identifier shared by all clips of one video.

    Everything up to and including the ``_clip_NNN`` marker is stripped, so all
    clips originating from the same recording collapse to one id.
    """
    stem = Path(path).stem
    return _CLIP_SUFFIX.sub("", stem)


def inventory(dataset: Path) -> pd.DataFrame:
    """Scan ``dataset`` for ``<label>/*.mp4`` clips and record video properties."""
    rows = []
    for label in CLASSES:
        for path in sorted((dataset / label).glob("*.mp4")):
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            rows.append(
                dict(
                    path=str(path.resolve()),
                    label=label,
                    video_id=video_id(path),
                    fps=fps,
                    frames=int(frames),
                    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    duration_seconds=frames / fps if fps else np.nan,
                )
            )
            cap.release()
    if not rows:
        raise FileNotFoundError(f"No MP4 files found beneath {dataset}")
    return pd.DataFrame(rows)


def grouped_split(manifest: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
    """Add a ``split`` column, assigning whole *videos* to train or test.

    All clips of a given ``video_id`` are kept together. Because some classes
    have very few unique videos, we split the *unique (label, video_id) groups*
    stratified by label; this guarantees at least one video per class in the
    test set whenever a class has >= 2 videos.
    """
    groups = (
        manifest[["label", "video_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    # Per-class video-level split so every class is represented in the test set.
    test_ids: set[str] = set()
    rng = np.random.RandomState(RANDOM_STATE)
    for label in CLASSES:
        vids = groups.loc[groups.label == label, "video_id"].tolist()
        if len(vids) < 2:
            # Only one source video: cannot hold any out without emptying train.
            continue
        n_test = max(1, int(round(len(vids) * test_size)))
        chosen = rng.choice(vids, size=n_test, replace=False)
        test_ids.update(chosen.tolist())
    out = manifest.copy()
    out["split"] = np.where(out.video_id.isin(test_ids), "test", "train")
    return out


def stratified_group_folds(
    manifest: pd.DataFrame, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield leakage-safe CV folds (train_idx, test_idx) into ``manifest``.

    Uses :class:`StratifiedGroupKFold` with ``video_id`` as the group so no
    video ever spans a fold boundary. ``n_splits`` is clamped to the smallest
    number of unique videos in any class.
    """
    per_class_videos = (
        manifest.groupby("label")["video_id"].nunique().min()
    )
    n_splits = int(max(2, min(n_splits, per_class_videos)))
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    idx = np.arange(len(manifest))
    return list(sgkf.split(idx, manifest.label.values, manifest.video_id.values))
