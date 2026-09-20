"""Best-validation losses of the trained models (the validation-loss table).

    python -m sonority_probe.training_summary --runs-dir runs --out outputs/training

Reads every run's summary.json and writes best_validation_by_seed.csv (one row per
model) and best_validation_summary.csv (mean, minimum and maximum over seeds per
architecture and condition). Losses are cross-entropies per predicted token,
including EOS, on the validation split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sonority_probe.config import ARCH_LAYERS, CONDITIONS, TRAIN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("outputs/training"))
    args = parser.parse_args()

    rows = []
    for arch in ARCH_LAYERS:
        for condition in CONDITIONS:
            for seed in TRAIN.seeds:
                summary = json.loads((args.runs_dir / arch / condition / f"seed_{seed}" / "summary.json").read_text())
                rows.append({
                    "arch": arch,
                    "condition": condition,
                    "seed": seed,
                    "best_val_epoch": summary["best_val_epoch"],
                    "best_val_loss": summary["best_val_loss"],
                })
    by_seed = pd.DataFrame(rows)
    summary = by_seed.groupby(["arch", "condition"], as_index=False, sort=False).agg(
        mean_loss=("best_val_loss", "mean"),
        seed_min=("best_val_loss", "min"),
        seed_max=("best_val_loss", "max"),
        n_seeds=("seed", "size"),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    by_seed.to_csv(args.out / "best_validation_by_seed.csv", index=False)
    summary.to_csv(args.out / "best_validation_summary.csv", index=False)
    print(f"{len(by_seed)} runs -> {args.out}")


if __name__ == "__main__":
    main()
