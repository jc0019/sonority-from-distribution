"""Witten-Bell bigram baseline, trained on the same training split as the neural
models and scored with the same measure (negative mean per-token surprisal over
BOS -> p1 ... pn -> EOS). Writes bigram_daland_item_scores.csv and
bigram_daland_spearman.csv in the format of sonority_probe.evaluate, with
arch "bigram-wb" and seed 0.

    python -m sonority_probe.bigram --out outputs/eval
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from sonority_probe.config import DALAND_DATA, TRAIN_SPLIT
from sonority_probe.evaluate import daland_correlations
from sonority_probe.vocab import BOS, EOS


def load_train_token_lists() -> list[list[str]]:
    with TRAIN_SPLIT.open(newline="") as f:
        return [row["nosyl"].split() for row in csv.DictReader(f)]


class BigramLM:
    """Bigram estimates interpolated with an add-one unigram; the bigram weight of a
    context is n / (n + t), for n tokens and t distinct types seen after it."""

    def __init__(self, words: list[list[str]]):
        self.unigram = Counter()  # counts of predicted tokens, EOS included
        self.bigram = defaultdict(Counter)
        for tokens in words:
            seq = [BOS, *tokens, EOS]
            self.unigram.update(seq[1:])
            for a, b in zip(seq, seq[1:]):
                self.bigram[a][b] += 1
        self.vocab_size = len(self.unigram)
        self.unigram_total = sum(self.unigram.values())
        self.context_total = {a: sum(c.values()) for a, c in self.bigram.items()}
        self.context_types = {a: len(c) for a, c in self.bigram.items()}

    def logp(self, context: str, target: str) -> float:
        p_unigram = (self.unigram[target] + 1) / (self.unigram_total + self.vocab_size)
        n = self.context_total.get(context, 0)
        if n == 0:
            return math.log(p_unigram)
        wb = n / (n + self.context_types[context])
        return math.log(wb * (self.bigram[context][target] / n) + (1 - wb) * p_unigram)

    def mean_token_surprisal(self, tokens: list[str]) -> float:
        seq = [BOS, *tokens, EOS]
        logps = [self.logp(a, b) for a, b in zip(seq, seq[1:])]
        return -sum(logps) / len(logps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lm = BigramLM(load_train_token_lists())
    daland = pd.read_csv(DALAND_DATA)
    rows = []
    for item in daland.itertuples():
        surprisal = lm.mean_token_surprisal(item.cmu_transcription.split())
        rows.append({
            "arch": "bigram-wb",
            "condition": "-",
            "seed": 0,
            "best_val_epoch": 0,
            "word": item.orthography,
            "sonority_group": item.sonority_group,
            "attested": item.attested,
            "mean_token_surprisal": surprisal,
            "model_acceptability": -surprisal,
            "expt2_2afc_score": item.expt2_2afc_score,
        })
    item_scores = pd.DataFrame(rows)
    item_scores.to_csv(args.out / "bigram_daland_item_scores.csv", index=False)
    daland_correlations(item_scores).to_csv(args.out / "bigram_daland_spearman.csv", index=False)
    print(f"scored {len(item_scores)} Daland items -> {args.out}")


if __name__ == "__main__":
    main()
