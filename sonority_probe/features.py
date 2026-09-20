from __future__ import annotations

import csv
import random
from pathlib import Path

import torch

from sonority_probe.vocab import SPECIAL_TOKENS, PhonemeVocab

FEATURE_VALUES = {"+": 1.0, "-": -1.0}


def _feature_value(symbol: str) -> float:
    return FEATURE_VALUES[symbol] if symbol in FEATURE_VALUES else float(symbol)


def load_feature_vectors(path: Path) -> dict[str, torch.Tensor]:
    """Articulatory feature vectors keyed by CMU phoneme."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = csv.DictReader(f)
        return {
            row["phoneme"]: torch.tensor([_feature_value(row[k]) for k in rows.fieldnames[1:]])
            for row in rows
        }


def build_embedding_matrix(
    vocab: PhonemeVocab,
    condition: str,
    *,
    feature_path: Path,
    embedding_dim: int,
    random_init_std: float,
    seed: int,
) -> torch.Tensor | None:
    """Frozen input matrix for feat/perm/rand (special-token rows zero, phoneme rows
    from the features, a seeded permutation of them, or seeded N(0, std^2) vectors).
    None for emb, whose phoneme vectors are the model's own trainable embedding."""
    if condition == "emb":
        return None
    if condition == "rand":
        generator = torch.Generator()
        generator.manual_seed(seed)
        matrix = torch.randn((len(vocab), embedding_dim), generator=generator) * random_init_std
        matrix[vocab.pad_id] = 0.0
        return matrix
    vectors = load_feature_vectors(feature_path)
    phonemes = [token for token in vocab.id_to_token if token not in SPECIAL_TOKENS]
    missing = sorted(p for p in phonemes if p not in vectors)
    if missing:
        raise ValueError(f"Missing feature vectors for phonemes: {missing}")
    if condition == "perm":
        shuffled = list(phonemes)
        random.Random(seed).shuffle(shuffled)
        vectors = {p: vectors[q] for p, q in zip(phonemes, shuffled)}
    matrix = torch.zeros((len(vocab), embedding_dim))
    for phoneme in phonemes:
        matrix[vocab.token_to_id[phoneme]] = vectors[phoneme]
    return matrix
