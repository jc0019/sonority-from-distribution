from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # the repository, which holds data/
RUNS_DIR = PROJECT_ROOT / "runs"


@dataclass(frozen=True)
class TrainConfig:
    """Data split, input scale and optimisation settings shared by all models."""

    hidden_size: int = 128  # LSTM hidden size
    embedding_dim: int = 21  # number of articulatory features
    embedding_init_std: float = 0.6541045281039316  # element-wise std of the feature matrix
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 40
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    seeds: tuple[int, ...] = tuple(range(101, 121))
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    split_seed: int = 999


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int = 128  # = LSTM hidden size
    n_heads: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.0
    max_len: int = 32  # positional embeddings; the longest word is 17 phonemes + BOS


TRAIN = TrainConfig()
TRANSFORMER = TransformerConfig()

DATA_DIR = PROJECT_ROOT / "data" / "derived"
TRAINING_DATA = DATA_DIR / "CMUcleaned.csv"  # 18,233 CMU entries (Daland et al., 2011)
FEATURE_DATA = DATA_DIR / "feat_noSyl.csv"  # 21 articulatory features per phoneme
TRAIN_SPLIT = DATA_DIR / f"train_split_seed{TRAIN.split_seed}.csv"
TEST_SPLIT = DATA_DIR / f"test_split_seed{TRAIN.split_seed}.csv"
DALAND_DATA = DATA_DIR / "human_acceptability.csv"  # 96 nonce words with human judgements
HAYES_WHITE_DATA = DATA_DIR / "Hayeswhite.csv"  # 40 violator/control pairs

# CLI architecture name -> number of Transformer layers (None = LSTM).
ARCH_LAYERS: dict[str, int | None] = {"lstm": None, "tf2": 2, "tf4": 4}

# Input conditions: articulatory features, trainable embeddings, permuted features, random vectors.
CONDITIONS = ("feat", "emb", "perm", "rand")

# Checkpoints of the developmental analyses (besides the untrained model, "step_0").
TRAJECTORY_EPOCHS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40)
