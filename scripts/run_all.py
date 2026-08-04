#!/usr/bin/env python3
"""Orchestrate the full pipeline: extract -> train Stage 1 -> run Stage 2.

Thin wrapper that shells out to the three stage scripts in order. Feature
extraction is skipped automatically if ``frame_features.csv`` already exists
(pass ``--overwrite`` to force re-extraction).

Example::

    python run_all.py --dataset /path/to/crowd_panic \\
        --weights /path/to/best_combined.pt --output outputs
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run(script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(HERE / script), *args]
    print(f"\n=== {script} ===\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--output", default="outputs")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    extract_args = ["--dataset", a.dataset, "--weights", a.weights, "--output", a.output]
    if a.overwrite:
        extract_args.append("--overwrite")
    run("extract_features.py", extract_args)
    run("train_stage1.py", ["--output", a.output])
    run("run_stage2.py", ["--output", a.output])
    print(f"\nAll stages complete. Artifacts in {a.output}/")


if __name__ == "__main__":
    main()
