"""Train the phoneme language models.

    python -m sonority_probe.train --arch lstm --seeds 101-120

Each run is written to <output-dir>/<arch>/<condition>/seed_<seed>/: the untrained
model (checkpoints/step_0.pt), every epoch (checkpoints/epoch_<n>.pt), the
best-validation checkpoint (checkpoints/best_val.pt), metrics.jsonl, summary.json
and a DONE marker. Runs with a DONE marker are skipped, so a batch can be resumed.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from sonority_probe.config import (
    ARCH_LAYERS,
    CONDITIONS,
    DATA_DIR,
    FEATURE_DATA,
    RUNS_DIR,
    TRAIN,
    TRAINING_DATA,
    TRANSFORMER,
)
from sonority_probe.data import load_training_examples, make_dataloaders, split_examples, write_split_csvs
from sonority_probe.features import build_embedding_matrix
from sonority_probe.model import build_model
from sonority_probe.vocab import PhonemeVocab, build_vocab


def run_epoch(model: nn.Module, loader, criterion: nn.Module, optimizer, device: torch.device) -> float:
    """One pass over the loader, training if an optimizer is given.
    Returns the cross-entropy per non-PAD target token (EOS included)."""
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens = 0.0, 0
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), TRAIN.grad_clip)
                optimizer.step()
            n_tokens = targets.ne(criterion.ignore_index).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    return total_loss / total_tokens


def train_run(
    arch: str,
    condition: str,
    seed: int,
    *,
    output_dir: Path,
    device: torch.device,
    vocab: PhonemeVocab,
    train_examples: list,
    val_examples: list,
    environment: dict,
) -> None:
    run_dir = output_dir / arch / condition / f"seed_{seed}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)  # left over from an interrupted attempt
    metadata = {
        "arch": arch,
        "condition": condition,
        "seed": seed,
        "train_config": asdict(TRAIN),
        "transformer_config": asdict(TRANSFORMER),
        "transformer_layers": ARCH_LAYERS[arch],
        "vocab": vocab.id_to_token,
        "environment": environment,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # The seed fixes the frozen input matrix (perm/rand) and the initial weights.
    torch.manual_seed(seed)
    frozen_matrix = build_embedding_matrix(
        vocab,
        condition,
        feature_path=FEATURE_DATA,
        embedding_dim=TRAIN.embedding_dim,
        random_init_std=TRAIN.embedding_init_std,
        seed=seed,
    )
    model = build_model(arch, vocab, frozen_matrix).to(device)
    train_loader, val_loader = make_dataloaders(
        vocab,
        train_examples=train_examples,
        val_examples=val_examples,
        batch_size=TRAIN.batch_size,
        train_seed=seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN.learning_rate, weight_decay=TRAIN.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    def save_checkpoint(name: str, epoch: int, val_loss: float | None) -> None:
        payload = {
            "model_state_dict": model.state_dict(),
            "arch": arch,
            "condition": condition,
            "seed": seed,
            "epoch": epoch,
            "val_loss": val_loss,
            "vocab": vocab.id_to_token,
        }
        torch.save(payload, checkpoint_dir / name)

    save_checkpoint("step_0.pt", 0, None)
    best_val, best_epoch = float("inf"), 0
    for epoch in range(1, TRAIN.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        with metrics_path.open("a") as f:
            f.write(json.dumps({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}) + "\n")
        save_checkpoint(f"epoch_{epoch}.pt", epoch, val_loss)
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            save_checkpoint("best_val.pt", epoch, val_loss)
        print(
            f"{arch} {condition} seed={seed} epoch={epoch}/{TRAIN.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} ({time.time() - t0:.1f}s)",
            flush=True,
        )

    summary = {"best_val_loss": best_val, "best_val_epoch": best_epoch}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "DONE").write_text(f"best_val_epoch={best_epoch} best_val={best_val:.6f}\n")


def parse_seeds(spec: str) -> tuple[int, ...]:
    """'101-120', '1,2,5' or a mix such as '101-105,110'."""
    seeds = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return tuple(seeds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train phoneme LMs (LSTM or Transformer).")
    parser.add_argument("--arch", choices=sorted(ARCH_LAYERS), default="lstm")
    parser.add_argument("--condition", choices=[*CONDITIONS, "all"], default="all")
    parser.add_argument("--seeds", default=None, help='e.g. "101-120" or "1,2,5"; default: the configured seeds')
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    examples = load_training_examples(TRAINING_DATA)
    vocab = build_vocab(example.phonemes for example in examples)
    train_examples, val_examples, test_examples = split_examples(
        examples,
        val_fraction=TRAIN.validation_fraction,
        test_fraction=TRAIN.test_fraction,
        split_seed=TRAIN.split_seed,
    )
    write_split_csvs(
        DATA_DIR,
        train=train_examples,
        val=val_examples,
        test=test_examples,
        split_seed=TRAIN.split_seed,
    )
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    conditions = CONDITIONS if args.condition == "all" else (args.condition,)
    seeds = TRAIN.seeds if args.seeds is None else parse_seeds(args.seeds)
    for condition in conditions:
        for seed in seeds:
            if (args.output_dir / args.arch / condition / f"seed_{seed}" / "DONE").exists():
                print(f"[skip] {args.arch} {condition} seed={seed}: already trained", flush=True)
                continue
            train_run(
                args.arch,
                condition,
                seed,
                output_dir=args.output_dir,
                device=torch.device(args.device),
                vocab=vocab,
                train_examples=train_examples,
                val_examples=val_examples,
                environment=environment,
            )
    print("ALL TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
