"""Sonority probes and RSA over the three representation spaces of every model.

    python -m sonority_probe.probe --mode endpoint   --runs-dir runs --out outputs/probe
    python -m sonority_probe.probe --mode trajectory --runs-dir runs --out outputs/probe

endpoint    the untrained model (step_0) and the best-validation checkpoint, with
            shuffled-label controls and RSA between the spaces
trajectory  step_0 and the developmental checkpoints (TRAJECTORY_EPOCHS)

Representation spaces, one vector per phoneme type:
    input               the phoneme's input vector
    hidden_bos          the hidden state after the sequence [BOS, phoneme]
    hidden_in_context   the hidden state at the phoneme, averaged over all of its
                        occurrences in the training split
Probes (ridge regression, leave-one-out, Spearman): full60 (sonority ranks 1-5 of
all 60 phonemes), cv_binary (consonant/vowel), consonants (ranks 1-4 of the 24
consonants). RSA: Spearman correlation of the pairwise correlation distances of
two spaces, over the consonants.

Every run is scored in its own worker and written to shards/; the shards are
then merged into probe_scores.csv, probe_summary.csv, rsa_scores.csv and
rsa_summary.csv.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from sonority_probe.analysis import is_vowel, probe_score, real_phoneme_tokens, sonority_rank
from sonority_probe.config import TRAIN_SPLIT, TRAJECTORY_EPOCHS
from sonority_probe.data import Example, PhonemeDataset, collate_batch, load_training_examples
from sonority_probe.evaluate import iter_run_dirs, load_model_from_checkpoint
from sonority_probe.vocab import PhonemeVocab

N_SHUFFLES = 20
SHUFFLE_SEED = 20260702
BATCH_SIZE = 2048
SPACES = ("input", "hidden_bos", "hidden_in_context")
RSA_PAIRS = (("input", "hidden_bos"), ("input", "hidden_in_context"), ("hidden_bos", "hidden_in_context"))


@torch.no_grad()
def in_context_representations(model, vocab: PhonemeVocab, train_examples: list[Example]) -> tuple[list[str], np.ndarray]:
    """Each phoneme type's hidden state averaged over its occurrences in the training words."""
    tokens = real_phoneme_tokens(vocab)
    ids = [vocab.token_to_id[token] for token in tokens]
    loader = DataLoader(
        PhonemeDataset(train_examples, vocab),
        batch_size=BATCH_SIZE,
        collate_fn=lambda batch: collate_batch(batch, pad_id=vocab.pad_id),
    )
    sums, counts = None, torch.zeros(len(tokens))
    for inputs, _ in loader:
        _, hidden = model(inputs)
        if sums is None:
            sums = torch.zeros(len(tokens), hidden.size(-1))
        flat_ids = inputs[:, 1:].reshape(-1)  # positions after BOS
        flat_hidden = hidden[:, 1:, :].reshape(-1, hidden.size(-1))
        for row, token_id in enumerate(ids):
            at_token = flat_ids.eq(token_id)
            sums[row] += flat_hidden[at_token].sum(dim=0)
            counts[row] += at_token.sum()
    return tokens, (sums / counts[:, None]).numpy()


@torch.no_grad()
def representation_spaces(model, vocab: PhonemeVocab, train_examples: list[Example]) -> tuple[list[str], dict[str, np.ndarray]]:
    tokens = real_phoneme_tokens(vocab)
    ids = torch.tensor([vocab.token_to_id[token] for token in tokens])
    input_space = model.embedding(ids).numpy()
    _, hidden = model(torch.stack([torch.full_like(ids, vocab.bos_id), ids], dim=1))
    bos_space = hidden[:, 1, :].numpy()
    _, in_context_space = in_context_representations(model, vocab, train_examples)
    return tokens, {"input": input_space, "hidden_bos": bos_space, "hidden_in_context": in_context_space}


def probe_rows(tokens: list[str], X: np.ndarray, *, shuffles: bool) -> list[dict]:
    ranks = np.array([sonority_rank(token) for token in tokens], dtype=float)
    consonant = np.array([not is_vowel(token) for token in tokens])
    rows = [
        {"probe": "full60", "spearman_rho": probe_score(X, ranks)},
        {"probe": "consonants", "spearman_rho": probe_score(X[consonant], ranks[consonant])},
        {"probe": "cv_binary", "spearman_rho": probe_score(X, (~consonant).astype(float))},
    ]
    if shuffles:  # the same probe on permuted labels, averaged over N_SHUFFLES permutations
        for name, X_probe, y in (("full60", X, ranks), ("consonants", X[consonant], ranks[consonant])):
            rng = np.random.default_rng(SHUFFLE_SEED)
            scores = [probe_score(X_probe, rng.permutation(y)) for _ in range(N_SHUFFLES)]
            rows.append({"probe": f"shuffled_{name}", "spearman_rho": float(np.mean(scores))})
    return rows


def rsa(X: np.ndarray, Y: np.ndarray) -> float:
    return float(spearmanr(pdist(X, metric="correlation"), pdist(Y, metric="correlation"))[0])


def checkpoint_labels(mode: str) -> list[str]:
    if mode == "endpoint":
        return ["step_0", "best_val"]
    return ["step_0", *(f"epoch_{epoch}" for epoch in TRAJECTORY_EPOCHS)]


def probe_one_run(arch: str, condition: str, seed: int, run_dir: str, mode: str, out_dir: str) -> None:
    torch.set_num_threads(1)  # one thread per worker
    train_examples = load_training_examples(TRAIN_SPLIT)
    probe_rows_, rsa_rows = [], []
    for label in checkpoint_labels(mode):
        model, vocab, ckpt = load_model_from_checkpoint(Path(run_dir) / "checkpoints" / f"{label}.pt", torch.device("cpu"))
        tokens, spaces = representation_spaces(model, vocab, train_examples)
        run = {"arch": arch, "condition": condition, "seed": seed, "checkpoint": label, "checkpoint_epoch": ckpt["epoch"]}
        for space, X in spaces.items():
            for row in probe_rows(tokens, X, shuffles=mode == "endpoint"):
                probe_rows_.append({**run, "extraction": space, **row})
        if mode == "endpoint" and label == "best_val":
            consonant = np.array([not is_vowel(token) for token in tokens])
            for a, b in RSA_PAIRS:
                rsa_rows.append({
                    "arch": arch,
                    "condition": condition,
                    "seed": seed,
                    "pair": f"{a}<->{b}",
                    "spearman_rho": rsa(spaces[a][consonant], spaces[b][consonant]),
                })
    shards = Path(out_dir) / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(probe_rows_).to_csv(shards / f"{arch}_{condition}_seed{seed}_{mode}.csv", index=False)
    if rsa_rows:
        pd.DataFrame(rsa_rows).to_csv(shards / f"{arch}_{condition}_seed{seed}_rsa.csv", index=False)


def merge_shards(out_dir: Path) -> None:
    shards = out_dir / "shards"
    probe_shards = sorted(shards.glob("*_endpoint.csv")) + sorted(shards.glob("*_trajectory.csv"))
    scores = pd.concat([pd.read_csv(shard) for shard in probe_shards], ignore_index=True)
    # step_0 is probed in both modes; keep one row per key.
    scores = scores.drop_duplicates(["arch", "condition", "seed", "checkpoint", "extraction", "probe"]).reset_index(drop=True)
    scores.to_csv(out_dir / "probe_scores.csv", index=False)
    summary = scores.groupby(["arch", "condition", "checkpoint", "extraction", "probe"], as_index=False).agg(
        mean_rho=("spearman_rho", "mean"),
        seed_min=("spearman_rho", "min"),
        seed_max=("spearman_rho", "max"),
        n_seeds=("spearman_rho", "size"),
    )
    summary.to_csv(out_dir / "probe_summary.csv", index=False)
    print(f"{len(probe_shards)} shards -> {out_dir / 'probe_scores.csv'} ({len(scores)} rows)")

    rsa_shards = sorted(shards.glob("*_rsa.csv"))
    if rsa_shards:
        rsa_scores = pd.concat([pd.read_csv(shard) for shard in rsa_shards], ignore_index=True)
        rsa_scores.to_csv(out_dir / "rsa_scores.csv", index=False)
        rsa_summary = rsa_scores.groupby(["arch", "condition", "pair"], as_index=False).agg(
            mean_rho=("spearman_rho", "mean"),
            seed_min=("spearman_rho", "min"),
            seed_max=("spearman_rho", "max"),
            n_seeds=("spearman_rho", "size"),
        )
        rsa_summary.to_csv(out_dir / "rsa_summary.csv", index=False)
        print(f"{len(rsa_shards)} RSA shards -> {out_dir / 'rsa_scores.csv'} ({len(rsa_scores)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("endpoint", "trajectory"), required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--merge-only", action="store_true", help="only rebuild the CSVs from existing shards")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.merge_only:
        runs = iter_run_dirs(args.runs_dir)
        todo = [
            run for run in runs
            if not (args.out / "shards" / f"{run[0]}_{run[1]}_seed{run[2]}_{args.mode}.csv").exists()
        ]
        print(f"{len(runs)} runs, {len(todo)} still to probe")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(probe_one_run, arch, condition, seed, str(run_dir), args.mode, str(args.out)): (arch, condition, seed)
                for arch, condition, seed, run_dir in todo
            }
            for i, future in enumerate(as_completed(futures), 1):
                future.result()
                arch, condition, seed = futures[future]
                print(f"  [{i}/{len(todo)}] {arch}/{condition}/seed_{seed}", flush=True)
    merge_shards(args.out)


if __name__ == "__main__":
    main()
