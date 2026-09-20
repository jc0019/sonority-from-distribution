"""Sonority ranks, probe scoring and nonce-word scoring shared by the analyses."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sonority_probe.vocab import SPECIAL_TOKENS, PhonemeVocab

VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
GLIDES = {"W", "Y"}
LIQUIDS = {"L", "R"}
NASALS = {"M", "N", "NG"}
RIDGE_ALPHA = 1.0


def strip_stress(phoneme: str) -> str:
    return phoneme.rstrip("012")


def is_vowel(phoneme: str) -> bool:
    return strip_stress(phoneme) in VOWELS


def sonority_rank(phoneme: str) -> int:
    """Obstruent 1 < nasal 2 < liquid 3 < glide 4 < vowel 5 (Clements, 1990)."""
    base = strip_stress(phoneme)
    if base in VOWELS:
        return 5
    if base in GLIDES:
        return 4
    if base in LIQUIDS:
        return 3
    if base in NASALS:
        return 2
    return 1


def vocab_from_checkpoint(ckpt: dict) -> PhonemeVocab:
    tokens = tuple(ckpt["vocab"])
    return PhonemeVocab({token: i for i, token in enumerate(tokens)}, tokens)


def real_phoneme_tokens(vocab: PhonemeVocab) -> list[str]:
    return [token for token in vocab.id_to_token if token not in SPECIAL_TOKENS]


def ridge_loocv_predictions(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out predictions of a ridge probe (alpha = 1) on standardised features."""
    preds = np.zeros(len(y))
    for i in range(len(y)):
        train = np.arange(len(y)) != i
        probe = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA)).fit(X[train], y[train])
        preds[i] = probe.predict(X[[i]])[0]
    return preds


def probe_score(X: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation between the targets and their leave-one-out probe
    predictions; 0 when the predictions are constant."""
    rho = spearmanr(y, ridge_loocv_predictions(X, y))[0]
    return 0.0 if math.isnan(rho) else float(rho)


def load_hayes_white_pairs(path: Path) -> pd.DataFrame:
    """The Hayes and White (2013) violator/control pairs, one row per pair."""
    hw = pd.read_csv(
        path,
        usecols=["Orthography", "CMU Transcription - no syllab", "Type", "Violator/Control",
                 "Constraint Violated", "Comparison form"],
    ).dropna(subset=["Orthography", "CMU Transcription - no syllab"])
    role = hw["Violator/Control"].str.lower().str.strip()
    controls = hw[role == "control"]
    control_by_word = {
        word.lower().strip(): (word, phonemes.strip())
        for word, phonemes in zip(controls["Orthography"], controls["CMU Transcription - no syllab"])
    }
    rows = []
    for _, v in hw[role == "violator"].iterrows():
        control_key = re.search(r"Control form is\s+(\w+)", v["Comparison form"]).group(1).lower()
        control_word, control_phonemes = control_by_word[control_key]
        rows.append({
            "pair_id": f"{v['Orthography'].lower()}/{control_word.lower()}",
            "naturalness": v["Type"].lower().strip(),
            "violator_word": v["Orthography"],
            "violator_phonemes": v["CMU Transcription - no syllab"].strip(),
            "control_word": control_word,
            "control_phonemes": control_phonemes,
            "constraint_violated": v["Constraint Violated"],
        })
    return pd.DataFrame(rows).sort_values(["naturalness", "pair_id"]).reset_index(drop=True)


@torch.no_grad()
def mean_token_surprisal(model: torch.nn.Module, vocab: PhonemeVocab, phoneme_string: str, *, device: torch.device) -> float:
    """Mean -log p per predicted token of a phoneme string, EOS included (the
    training loss of a single word)."""
    ids = vocab.encode(phoneme_string.split())
    inputs = torch.tensor([[vocab.bos_id, *ids]], device=device)
    targets = torch.tensor([[*ids, vocab.eos_id]], device=device)
    logits, _ = model(inputs)
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
    return nll.mean().item()
