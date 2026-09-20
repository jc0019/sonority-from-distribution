from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from sonority_probe.vocab import PhonemeVocab


@dataclass(frozen=True)
class Example:
    ortho: str
    syl: str
    phonemes: tuple[str, ...]


class PhonemeDataset(Dataset):
    def __init__(self, examples: list[Example], vocab: PhonemeVocab):
        self.examples = examples
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # BOS predicts the first phoneme; the last phoneme predicts EOS.
        ids = self.vocab.encode(self.examples[index].phonemes)
        inputs = [self.vocab.bos_id, *ids]
        targets = [*ids, self.vocab.eos_id]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def load_training_examples(path: Path) -> list[Example]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        examples = []
        for row in csv.DictReader(f):
            phonemes = tuple(row["nosyl"].split())
            if phonemes:
                examples.append(Example(row["ortho"], row.get("syl", ""), phonemes))
    return examples


def split_examples(
    examples: list[Example], *, val_fraction: float, test_fraction: float, split_seed: int
) -> tuple[list[Example], list[Example], list[Example]]:
    """Fixed train/validation/test split of the entries, shared by all models."""
    shuffled = list(examples)
    random.Random(split_seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = round(n * test_fraction)
    n_val = round(n * val_fraction)
    return shuffled[n_test + n_val :], shuffled[n_test : n_test + n_val], shuffled[:n_test]


def write_split_csvs(
    output_dir: Path, *, train: list[Example], val: list[Example], test: list[Example], split_seed: int
) -> None:
    """The test split keeps the syllabified form, which the contour analysis needs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val), ("test", test)):
        fieldnames = ["ortho", "syl", "nosyl"] if name == "test" else ["ortho", "nosyl"]
        with (output_dir / f"{name}_split_seed{split_seed}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for example in split:
                row = {"ortho": example.ortho, "nosyl": " ".join(example.phonemes)}
                if name == "test":
                    row["syl"] = example.syl
                writer.writerow(row)


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]], *, pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    input_seqs, target_seqs = zip(*batch)
    max_len = max(seq.size(0) for seq in input_seqs)
    inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    targets = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, (input_seq, target_seq) in enumerate(zip(input_seqs, target_seqs)):
        inputs[i, : input_seq.size(0)] = input_seq
        targets[i, : target_seq.size(0)] = target_seq
    return inputs, targets


def make_dataloaders(
    vocab: PhonemeVocab,
    *,
    train_examples: list[Example],
    val_examples: list[Example],
    batch_size: int,
    train_seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Training minibatches are reshuffled every epoch from a generator seeded with the run seed."""

    def loader(examples: list[Example], shuffle: bool) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(train_seed)
        return DataLoader(
            PhonemeDataset(examples, vocab),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda batch: collate_batch(batch, pad_id=vocab.pad_id),
            generator=generator,
        )

    return loader(train_examples, shuffle=True), loader(val_examples, shuffle=False)
