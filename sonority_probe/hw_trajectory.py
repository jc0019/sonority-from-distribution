"""Violator-control discrimination across training checkpoints.

    python -m sonority_probe.hw_trajectory --runs-dir runs --out outputs/hw_trajectory

Scores the Hayes and White pairs as sonority_probe.evaluate does, but at the
untrained model and every developmental checkpoint (TRAJECTORY_EPOCHS) of every
run. hw_trajectory_scores.csv has one row per model, checkpoint and pair;
hw_trajectory_summary.csv gives the mean violator - control difference per epoch
(over seeds) for the natural pairs, the unnatural pairs, and the four
sonority-sequencing pairs, i.e. those violating a coda sonority constraint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from sonority_probe.analysis import load_hayes_white_pairs, mean_token_surprisal
from sonority_probe.config import HAYES_WHITE_DATA, TRAJECTORY_EPOCHS
from sonority_probe.evaluate import iter_run_dirs, load_model_from_checkpoint

SONORITY_SEQUENCING_CONSTRAINTS = {
    "*[-sonorant,+rhyme][+sonorant,+rhyme]",
    "*[+consonantal,+rhyme][-consonantal,+rhyme]",
}


def score_one_run(arch: str, condition: str, seed: int, run_dir: Path, pairs: pd.DataFrame) -> list[dict]:
    device = torch.device("cpu")
    rows = []
    for epoch in (0, *TRAJECTORY_EPOCHS):
        label = "step_0" if epoch == 0 else f"epoch_{epoch}"
        model, vocab, _ = load_model_from_checkpoint(run_dir / "checkpoints" / f"{label}.pt", device)
        for pair in pairs.itertuples():
            control = mean_token_surprisal(model, vocab, pair.control_phonemes, device=device)
            violator = mean_token_surprisal(model, vocab, pair.violator_phonemes, device=device)
            rows.append({
                "arch": arch,
                "condition": condition,
                "seed": seed,
                "checkpoint": label,
                "epoch": epoch,
                "naturalness": pair.naturalness,
                "pair_id": pair.pair_id,
                "sonority_seq": pair.sonority_seq,
                "delta_mean_token_surprisal": violator - control,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = load_hayes_white_pairs(HAYES_WHITE_DATA)
    pairs["sonority_seq"] = pairs["constraint_violated"].isin(SONORITY_SEQUENCING_CONSTRAINTS)
    rows = []
    for arch, condition, seed, run_dir in iter_run_dirs(args.runs_dir):
        rows += score_one_run(arch, condition, seed, run_dir, pairs)
        print(f"  {arch}/{condition}/seed_{seed}", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(args.out / "hw_trajectory_scores.csv", index=False)

    summaries = []
    for group, group_scores in (
        ("natural", scores[scores["naturalness"] == "natural"]),
        ("unnatural", scores[scores["naturalness"] == "unnatural"]),
        ("sonority_seq", scores[scores["sonority_seq"]]),
    ):
        per_seed = group_scores.groupby(["arch", "condition", "seed", "epoch"], as_index=False).agg(
            delta=("delta_mean_token_surprisal", "mean")
        )
        summary = per_seed.groupby(["arch", "condition", "epoch"], as_index=False).agg(
            delta_mean=("delta", "mean"),
            delta_min=("delta", "min"),
            delta_max=("delta", "max"),
            n_seeds=("delta", "size"),
        )
        summary.insert(0, "group", group)
        summaries.append(summary)
    pd.concat(summaries, ignore_index=True).to_csv(args.out / "hw_trajectory_summary.csv", index=False)
    print(f"{len(scores)} rows -> {args.out}")


if __name__ == "__main__":
    main()
