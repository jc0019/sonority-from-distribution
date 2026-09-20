"""Which consonants does the in-context consonants probe misrank?

    python -m sonority_probe.per_phoneme --runs-dir runs --out outputs/per_phoneme

For every model's best-validation checkpoint, the consonants probe (ridge
regression on the in-context representations, leave-one-out) is refitted and its
24 predictions are ranked. Sorted by true sonority class, the ranking positions
form blocks: obstruents 1-17, nasals 18-20, liquids 21-22, glides 23-24. A
consonant's displacement is 0 if its predicted position falls inside its own
block and otherwise the signed distance to the nearest edge of that block
(positive: ranked as more sonorous than it is, negative: as less sonorous).

per_phoneme_predictions.csv has one row per model and consonant;
per_phoneme_summary.csv averages over seeds (mean displacement and the fraction
of seeds displaced in each direction) and is what the heatmap figure reads.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

from sonority_probe.analysis import is_vowel, ridge_loocv_predictions, sonority_rank
from sonority_probe.config import TRAIN_SPLIT
from sonority_probe.data import load_training_examples
from sonority_probe.evaluate import iter_run_dirs, load_model_from_checkpoint
from sonority_probe.probe import in_context_representations

CLASS_NAME = {1: "obstruent", 2: "nasal", 3: "liquid", 4: "glide"}


def displacements(tokens: list[str], X: np.ndarray) -> pd.DataFrame:
    consonant = np.array([not is_vowel(token) for token in tokens])
    consonants = [token for token, c in zip(tokens, consonant) if c]
    ranks = np.array([sonority_rank(token) for token in consonants], dtype=float)
    preds = ridge_loocv_predictions(X[consonant], ranks)
    positions = rankdata(preds, method="ordinal")  # 1..24, ties broken by order
    rho = spearmanr(ranks, preds)[0]

    # The block of positions each class occupies when the consonants are sorted by rank.
    blocks, low = {}, 1
    for rank in sorted(set(ranks)):
        size = int((ranks == rank).sum())
        blocks[rank] = (low, low + size - 1)
        low += size

    rows = []
    for token, rank, pred, pos in zip(consonants, ranks, preds, positions):
        low, high = blocks[rank]
        displacement = 0 if low <= pos <= high else (pos - high if pos > high else pos - low)
        rows.append({
            "phoneme": token,
            "true_rank": int(rank),
            "sonority_class": CLASS_NAME[int(rank)],
            "pred": pred,
            "pred_pos": int(pos),
            "oob_displacement": float(displacement),
            "probe_rho": rho,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train_examples = load_training_examples(TRAIN_SPLIT)
    tables = []
    for arch, condition, seed, run_dir in iter_run_dirs(args.runs_dir):
        model, vocab, _ = load_model_from_checkpoint(run_dir / "checkpoints" / "best_val.pt", torch.device("cpu"))
        tokens, X = in_context_representations(model, vocab, train_examples)
        table = displacements(tokens, X)
        table.insert(0, "seed", seed)
        table.insert(0, "condition", condition)
        table.insert(0, "arch", arch)
        tables.append(table)
        print(f"  {arch}/{condition}/seed_{seed}", flush=True)
    predictions = pd.concat(tables, ignore_index=True)
    summary = predictions.groupby(["arch", "condition", "phoneme"], as_index=False).agg(
        true_rank=("true_rank", "first"),
        sonority_class=("sonority_class", "first"),
        mean_oob=("oob_displacement", "mean"),
        p_oob_positive=("oob_displacement", lambda d: float((d > 0).mean())),
        p_oob_negative=("oob_displacement", lambda d: float((d < 0).mean())),
        n_seeds=("seed", "nunique"),
    )
    predictions.to_csv(args.out / "per_phoneme_predictions.csv", index=False)
    summary.to_csv(args.out / "per_phoneme_summary.csv", index=False)
    print(f"{len(tables)} runs -> {args.out}")


if __name__ == "__main__":
    main()
