from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
SPECIAL_TOKENS = (PAD, BOS, EOS)


@dataclass(frozen=True)
class PhonemeVocab:
    token_to_id: dict[str, int]
    id_to_token: tuple[str, ...]

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(self, phonemes: Iterable[str]) -> list[int]:
        return [self.token_to_id[p] for p in phonemes]


def build_vocab(sequences: Iterable[Iterable[str]]) -> PhonemeVocab:
    """Special tokens take ids 0-2, followed by the sorted phoneme inventory."""
    tokens = (*SPECIAL_TOKENS, *sorted({p for seq in sequences for p in seq}))
    return PhonemeVocab({token: i for i, token in enumerate(tokens)}, tokens)
