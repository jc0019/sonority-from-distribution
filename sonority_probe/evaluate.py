"""Behavioural evaluation of the trained models on the Daland et al. (2011) nonce
words and the Hayes and White (2013) violator/control pairs.

    python -m sonority_probe.evaluate --runs-dir runs --out outputs/eval

Each model's best-validation checkpoint scores every nonce word by its mean
per-token surprisal. Per item and model: daland_item_scores.csv and
hw_pair_scores.csv. Per model: daland_spearman_by_seed.csv (Spearman correlation
of -surprisal with the human 2AFC scores, pooled and within each attestedness
group) and hw_summary_by_seed.csv (mean violator - control surprisal difference
and the percentage of pairs with a positive difference, for natural and unnatural
constraints). Their means over seeds: daland_spearman_summary.csv and hw_summary.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from sonority_probe.analysis import load_hayes_white_pairs, mean_token_surprisal, vocab_from_checkpoint
from sonority_probe.config import DALAND_DATA, HAYES_WHITE_DATA
from sonority_probe.model import build_model
from sonority_probe.vocab import PhonemeVocab

DALAND_GROUPS = ("Attested", "Marginal", "Unattested")


def load_model_from_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, PhonemeVocab, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    vocab = vocab_from_checkpoint(ckpt)
    state = ckpt["model_state_dict"]
    model = build_model(ckpt["arch"], vocab, state.get("embedding.frozen_matrix"))
    model.load_state_dict(state)
    return model.to(device).eval(), vocab, ckpt


def iter_run_dirs(runs_root: Path) -> list[tuple[str, str, int, Path]]:
    """(arch, condition, seed, run_dir) for every run with a best_val checkpoint."""
    runs = []
    for ckpt in sorted(runs_root.glob("*/*/seed_*/checkpoints/best_val.pt")):
        run_dir = ckpt.parent.parent
        runs.append((run_dir.parent.parent.name, run_dir.parent.name, int(run_dir.name.removeprefix("seed_")), run_dir))
    return runs


def daland_correlations(item_scores: pd.DataFrame) -> pd.DataFrame:
    """Per model: Spearman correlation between model acceptability and the human 2AFC
    scores over all items and within each attestedness group."""
    rows = []
    for (arch, condition, seed), items in item_scores.groupby(["arch", "condition", "seed"]):
        for scope in ("pooled", *DALAND_GROUPS):
            scope_items = items if scope == "pooled" else items[items["attested"] == scope]
            rows.append({
                "arch": arch,
                "condition": condition,
                "seed": seed,
                "scope": scope,
                "n_items": len(scope_items),
                "spearman_rho": scope_items["model_acceptability"].corr(scope_items["expt2_2afc_score"], method="spearman"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    daland = pd.read_csv(DALAND_DATA)
    hw_pairs = load_hayes_white_pairs(HAYES_WHITE_DATA)
    runs = iter_run_dirs(args.runs_dir)
    print(f"{len(runs)} runs under {args.runs_dir}")

    item_rows, pair_rows = [], []
    for arch, condition, seed, run_dir in runs:
        model, vocab, ckpt = load_model_from_checkpoint(run_dir / "checkpoints" / "best_val.pt", device)
        run = {"arch": arch, "condition": condition, "seed": seed, "best_val_epoch": ckpt["epoch"]}
        for item in daland.itertuples():
            surprisal = mean_token_surprisal(model, vocab, item.cmu_transcription, device=device)
            item_rows.append({
                **run,
                "word": item.orthography,
                "sonority_group": item.sonority_group,
                "attested": item.attested,
                "mean_token_surprisal": surprisal,
                "model_acceptability": -surprisal,
                "expt2_2afc_score": item.expt2_2afc_score,
            })
        for pair in hw_pairs.itertuples():
            control = mean_token_surprisal(model, vocab, pair.control_phonemes, device=device)
            violator = mean_token_surprisal(model, vocab, pair.violator_phonemes, device=device)
            pair_rows.append({
                **run,
                "naturalness": pair.naturalness,
                "pair_id": pair.pair_id,
                "control_word": pair.control_word,
                "violator_word": pair.violator_word,
                "control_mean_token_surprisal": control,
                "violator_mean_token_surprisal": violator,
                "delta_mean_token_surprisal": violator - control,
            })
        print(f"  scored {arch}/{condition}/seed_{seed} (best epoch {ckpt['epoch']})", flush=True)

    item_scores = pd.DataFrame(item_rows)
    pair_scores = pd.DataFrame(pair_rows)
    correlations = daland_correlations(item_scores)
    hw_by_seed = pair_scores.groupby(["arch", "condition", "seed", "naturalness"], as_index=False).agg(
        n_pairs=("pair_id", "nunique"),
        delta_mean=("delta_mean_token_surprisal", "mean"),
        accuracy_pct=("delta_mean_token_surprisal", lambda d: 100 * float((d > 0).mean())),
    )
    correlation_summary = correlations.groupby(["arch", "condition", "scope"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        rho_mean=("spearman_rho", "mean"),
        rho_min=("spearman_rho", "min"),
        rho_max=("spearman_rho", "max"),
    )
    hw_summary = hw_by_seed.groupby(["arch", "condition", "naturalness"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        delta_mean=("delta_mean", "mean"),
        accuracy_pct=("accuracy_pct", "mean"),
    )
    item_scores.to_csv(args.out / "daland_item_scores.csv", index=False)
    pair_scores.to_csv(args.out / "hw_pair_scores.csv", index=False)
    correlations.to_csv(args.out / "daland_spearman_by_seed.csv", index=False)
    hw_by_seed.to_csv(args.out / "hw_summary_by_seed.csv", index=False)
    correlation_summary.to_csv(args.out / "daland_spearman_summary.csv", index=False)
    hw_summary.to_csv(args.out / "hw_summary.csv", index=False)
    print(f"wrote 6 CSVs to {args.out}")


if __name__ == "__main__":
    main()
