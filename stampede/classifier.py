"""Stage 1 crowd-risk classifier: RandomForest + normalisation + risk mapping.

The classifier is trained on per-frame features (each frame inherits its clip's
label). At inference it emits both a discrete class and a continuous risk score
in [0, 1] - the latter is the signal Stage 2 consumes. Normalisation parameters
are fit on the training split only and persisted alongside the model so new
videos can be scored without the training data.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .config import CLASSES, FEATURE_COLUMNS, RANDOM_STATE, RISK_ANCHOR


@dataclass
class Normalizer:
    """Min-max scaler to [0, 1] with mean/std retained for reporting."""

    cols: list[str]
    min_: np.ndarray
    max_: np.ndarray
    mean_: np.ndarray
    std_: np.ndarray

    @classmethod
    def fit(cls, df: pd.DataFrame, cols: list[str]) -> "Normalizer":
        x = df[cols].to_numpy(dtype=float)
        return cls(
            cols=list(cols),
            min_=np.nanmin(x, axis=0),
            max_=np.nanmax(x, axis=0),
            mean_=np.nanmean(x, axis=0),
            std_=np.nanstd(x, axis=0),
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.cols].to_numpy(dtype=float)
        span = np.where((self.max_ - self.min_) == 0, 1.0, self.max_ - self.min_)
        return np.clip((x - self.min_) / span, 0.0, 1.0)

    def to_dict(self) -> dict:
        return {
            "cols": self.cols,
            "min": self.min_.tolist(),
            "max": self.max_.tolist(),
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
        }


@dataclass
class Stage1Model:
    """Trained classifier bundle: RF + normaliser + class ordering.

    ``risk_anchors`` maps each class label to its scalar risk contribution; the
    continuous risk score is the class-probability expectation over these
    anchors. For the 4-class model these are the ordinal
    :data:`config.RISK_ANCHOR` values; for the binary model they are
    ``{"Safe": 0.0, "Stampede Risk": 1.0}`` so risk == P(Stampede Risk).
    """

    rf: RandomForestClassifier
    normalizer: Normalizer
    classes: list[str]
    risk_anchors: dict

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.rf.predict(self.normalizer.transform(df))

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        proba = self.rf.predict_proba(self.normalizer.transform(df))
        return pd.DataFrame(proba, columns=self.rf.classes_, index=df.index)

    def risk_score(self, df: pd.DataFrame) -> np.ndarray:
        """Continuous risk in [0, 1] as the class-probability expectation over
        the risk anchors."""
        proba = self.predict_proba(df)
        anchors = np.array([self.risk_anchors[c] for c in proba.columns])
        return (proba.to_numpy() * anchors).sum(axis=1)

    def save(self, path: Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: Path) -> "Stage1Model":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def train(
    train_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLUMNS,
    n_estimators: int = 400,
    min_samples_leaf: int = 2,
    risk_anchors: dict | None = None,
) -> Stage1Model:
    """Fit the normaliser and RandomForest on ``train_df`` (per-frame rows).

    ``risk_anchors`` defaults to the 4-class ordinal anchors; pass a custom map
    (e.g. the binary ``{"Safe": 0.0, "Stampede Risk": 1.0}``) for other label
    schemes.
    """
    normalizer = Normalizer.fit(train_df, feature_cols)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(normalizer.transform(train_df), train_df.label.values)
    anchors = dict(RISK_ANCHOR) if risk_anchors is None else dict(risk_anchors)
    classes = list(CLASSES) if risk_anchors is None else list(anchors.keys())
    return Stage1Model(
        rf=rf, normalizer=normalizer, classes=classes, risk_anchors=anchors
    )
