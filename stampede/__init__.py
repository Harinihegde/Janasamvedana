"""Early Stampede Detection System.

A two-stage crowd-safety pipeline:

* **Stage 1** - per-frame crowd-risk classification (No Panic / Normal / Crowdy /
  Panic) from density, motion, temporal-instability and control-loss features.
* **Stage 2** - escalation detection over continuous risk-score timelines,
  emitting early Normal->Crowdy and Crowdy->Panic alerts.

The package is deliberately modular so each stage can be reused on new videos
without retraining. See the top-level ``extract_features.py``, ``train_stage1.py``
and ``run_stage2.py`` scripts for the command-line entry points.
"""

from . import config  # noqa: F401

__all__ = ["config"]
