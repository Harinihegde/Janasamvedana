"""Plotting helpers: confusion matrix, risk curves, and alert overlays."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from .config import CLASSES, CP_HIGH, NC_HIGH


def plot_confusion(cm: list[list[int]], labels: list[str], path: Path, title: str):
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}\n{norm[i, j]:.0%}",
                ha="center",
                va="center",
                color="white" if norm[i, j] > 0.5 else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_risk_curve(
    risk_raw: np.ndarray,
    risk_smoothed: np.ndarray,
    alerts: list[dict],
    path: Path,
    title: str,
    transition_frame: int | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(risk_raw))
    ax.plot(x, risk_raw, color="#9aa0a6", lw=1, alpha=0.6, label="risk (raw)")
    ax.plot(x, risk_smoothed, color="#1a73e8", lw=2, label="risk (smoothed)")
    ax.axhline(NC_HIGH, color="#f9ab00", ls="--", lw=1, label="Crowdy threshold (0.5)")
    ax.axhline(CP_HIGH, color="#d93025", ls="--", lw=1, label="Panic threshold (0.8)")
    if transition_frame is not None:
        ax.axvline(
            transition_frame, color="#5f6368", ls=":", lw=1.5, label="true transition"
        )
    colors = {"normal_to_crowdy": "#f9ab00", "crowdy_to_panic": "#d93025"}
    for a in alerts:
        ax.axvline(a["frame"], color=colors.get(a["type"], "k"), lw=1.5, alpha=0.8)
        ax.scatter([a["frame"]], [a["risk"]], color=colors.get(a["type"], "k"), zorder=5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Frame (sampled)")
    ax.set_ylabel("Risk score")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_cv_summary(fold_accuracies: list[float], path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(1, len(fold_accuracies) + 1)
    ax.bar(x, fold_accuracies, color="#1a73e8", alpha=0.8)
    mean = float(np.mean(fold_accuracies))
    ax.axhline(mean, color="#d93025", ls="--", label=f"mean = {mean:.2%}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Clip-level accuracy")
    ax.set_title("Leakage-safe StratifiedGroupKFold CV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
