"""Predicted sonority contours on the held-out test words.

    python -m sonority_probe.contour --runs-dir runs --out outputs/eval

At every position of every test word, the next-phoneme distribution is summed by
the sonority-rank change the next phoneme would make relative to the current one
(fall 4 ... fall 1, level, rise 1 ... rise 4, or the end of the word). These
distributions are averaged over positions with the same syllabic role, read off
the test split's syllabification (the models only see the phoneme strings):

    onset-1        first consonant of an onset
    onset-2+       later consonant of an onset cluster
    nucleus        vowel
    coda-nonfinal  coda consonant followed by another coda consonant
    coda-final     last coda consonant of its syllable

contour_heldout.csv has one row per role for the empirical proportions
(source "data"), the Witten-Bell bigram, and every model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sonority_probe.analysis import sonority_rank
from sonority_probe.bigram import BigramLM, load_train_token_lists
from sonority_probe.config import TEST_SPLIT
from sonority_probe.evaluate import iter_run_dirs, load_model_from_checkpoint
from sonority_probe.vocab import BOS, EOS, PAD

CATEGORIES = ["fall4", "fall3", "fall2", "fall1", "level", "rise1", "rise2", "rise3", "rise4", "end"]
ROLES = ["onset-1", "onset-2+", "nucleus", "coda-nonfinal", "coda-final"]


def category(current_rank: int, next_token: str) -> str:
    if next_token == EOS:
        return "end"
    d = sonority_rank(next_token) - current_rank
    return "level" if d == 0 else (f"rise{d}" if d > 0 else f"fall{-d}")


def syllabic_roles(phonemes: list[str], coda: list[bool]) -> list[str]:
    roles = []
    for i, p in enumerate(phonemes):
        if sonority_rank(p) == 5:
            roles.append("nucleus")
        elif coda[i]:
            next_is_coda = i + 1 < len(phonemes) and coda[i + 1]
            roles.append("coda-nonfinal" if next_is_coda else "coda-final")
        else:
            after_onset_consonant = i > 0 and sonority_rank(phonemes[i - 1]) < 5 and not coda[i - 1]
            roles.append("onset-2+" if after_onset_consonant else "onset-1")
    return roles


def category_matrix(vocab, current_rank: int) -> torch.Tensor:
    """(vocab_size, n_categories) 0/1 matrix from next-token id to category; PAD and
    BOS, which are never predicted, map to no category."""
    M = np.zeros((len(vocab), len(CATEGORIES)))
    for tid, tok in enumerate(vocab.id_to_token):
        if tok not in (PAD, BOS):
            M[tid, CATEGORIES.index(category(current_rank, tok))] = 1
    return torch.tensor(M, dtype=torch.float32)


def bigram_category_vectors(lm: BigramLM, tokens: set[str]) -> dict[str, np.ndarray]:
    """The bigram's next-token distribution after each token, summed by category."""
    out = {}
    for tok in tokens:
        vec = np.zeros(len(CATEGORIES))
        for nxt in lm.unigram:
            vec[CATEGORIES.index(category(sonority_rank(tok), nxt))] += float(np.exp(lm.logp(tok, nxt)))
        out[tok] = vec / vec.sum()
    return out


def _row(source: str, arch: str, seed: int, role: str, n: int, vec: np.ndarray) -> dict:
    return {"source": source, "arch": arch, "seed": seed, "context": role, "n": n, **dict(zip(CATEGORIES, vec))}


def heldout_rows(words: list[tuple[list[str], list[bool]]], lm: BigramLM, runs, device: torch.device) -> list[dict]:
    role_lists = [syllabic_roles(phonemes, coda) for phonemes, coda in words]
    rows = []

    # Empirical proportions of the actual next phoneme.
    counts = defaultdict(lambda: np.zeros(len(CATEGORIES)))
    for (phonemes, _), roles in zip(words, role_lists):
        for i, tok in enumerate(phonemes):
            nxt = phonemes[i + 1] if i + 1 < len(phonemes) else EOS
            counts[roles[i]][CATEGORIES.index(category(sonority_rank(tok), nxt))] += 1
    for role in ROLES:
        rows.append(_row("data", "data", 0, role, int(counts[role].sum()), counts[role] / counts[role].sum()))

    bigram_vectors = bigram_category_vectors(lm, {tok for phonemes, _ in words for tok in phonemes})
    total, n = defaultdict(lambda: np.zeros(len(CATEGORIES))), defaultdict(int)
    for (phonemes, _), roles in zip(words, role_lists):
        for i, tok in enumerate(phonemes):
            total[roles[i]] += bigram_vectors[tok]
            n[roles[i]] += 1
    for role in ROLES:
        rows.append(_row("bigram", "bigram", 0, role, n[role], total[role] / n[role]))

    for arch, condition, seed, run_dir in runs:
        model, vocab, _ = load_model_from_checkpoint(run_dir / "checkpoints" / "best_val.pt", device)
        category_matrices = {rank: category_matrix(vocab, rank) for rank in range(1, 6)}
        total, n = defaultdict(lambda: np.zeros(len(CATEGORIES))), defaultdict(int)
        with torch.no_grad():
            for start in range(0, len(words), 512):
                chunk, chunk_roles = words[start : start + 512], role_lists[start : start + 512]
                length = max(len(phonemes) for phonemes, _ in chunk) + 1
                ids = torch.full((len(chunk), length), vocab.pad_id, dtype=torch.long)
                for j, (phonemes, _) in enumerate(chunk):
                    ids[j, 0] = vocab.bos_id
                    ids[j, 1 : 1 + len(phonemes)] = torch.tensor(vocab.encode(phonemes))
                logits, _ = model(ids.to(device))
                probs = torch.softmax(logits, dim=-1).cpu()
                for j, (phonemes, _) in enumerate(chunk):
                    for i, tok in enumerate(phonemes):
                        vec = (probs[j, i + 1] @ category_matrices[sonority_rank(tok)]).numpy()
                        total[chunk_roles[j][i]] += vec / vec.sum()
                        n[chunk_roles[j][i]] += 1
        for role in ROLES:
            rows.append(_row(condition, arch, seed, role, n[role], total[role] / n[role]))
        print(f"  held-out contours: {arch}/{condition}/seed_{seed}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(TEST_SPLIT)
    words = [(row.nosyl.split(), [tok.endswith("Coda") for tok in row.syl.split()]) for row in test.itertuples()]
    lm = BigramLM(load_train_token_lists())
    runs = iter_run_dirs(args.runs_dir)
    print(f"{len(words)} held-out words; {len(runs)} runs")
    rows = heldout_rows(words, lm, runs, torch.device(args.device))
    pd.DataFrame(rows).to_csv(args.out / "contour_heldout.csv", index=False)
    print(f"wrote {args.out / 'contour_heldout.csv'}")


if __name__ == "__main__":
    main()
