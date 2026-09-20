from __future__ import annotations

import torch
from torch import nn

from sonority_probe.config import ARCH_LAYERS, TRAIN, TRANSFORMER
from sonority_probe.vocab import PhonemeVocab


class ConditionedEmbedding(nn.Module):
    """Phoneme embedding whose PAD row is zero and whose BOS/EOS rows are trainable in
    every condition. Given a frozen matrix (feat/perm/rand), the phoneme rows (ids >= 3)
    are taken from it on every forward pass; without one (emb) they are trained."""

    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        pad_id: int,
        init_std: float,
        frozen_matrix: torch.Tensor | None,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.weight = nn.Parameter(torch.empty(vocab_size, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)
        self.register_buffer("frozen_matrix", None if frozen_matrix is None else frozen_matrix.clone())
        with torch.no_grad():
            self.weight[self.pad_id].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        weight = self.weight.clone()
        if self.frozen_matrix is not None:
            weight[3:] = self.frozen_matrix[3:]
        weight[self.pad_id] = 0.0
        return nn.functional.embedding(input_ids, weight, padding_idx=self.pad_id)


class LSTMPhonemeLM(nn.Module):
    """Single-layer LSTM with a linear output layer; forward returns (logits, hidden states)."""

    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        pad_id: int,
        init_std: float,
        frozen_matrix: torch.Tensor | None,
    ):
        super().__init__()
        self.embedding = ConditionedEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            pad_id=pad_id,
            init_std=init_std,
            frozen_matrix=frozen_matrix,
        )
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)
        # Xavier-uniform input and output weights, orthogonal recurrent weights, zero biases.
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, _ = self.lstm(self.embedding(input_ids))
        return self.output(hidden_states), hidden_states


class CausalTransformerPhonemeLM(nn.Module):
    """Decoder-only Transformer (post-LN encoder layers under a causal mask). A linear
    projection maps the input vectors to d_model; the learned positional embedding
    starts at zero. forward returns (logits, last-layer hidden states)."""

    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        pad_id: int,
        init_std: float,
        frozen_matrix: torch.Tensor | None,
        max_len: int,
    ):
        super().__init__()
        self.embedding = ConditionedEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            pad_id=pad_id,
            init_std=init_std,
            frozen_matrix=frozen_matrix,
        )
        self.input_projection = nn.Linear(embedding_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.position_embedding = nn.Parameter(torch.zeros(max_len, d_model))
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_projection(self.embedding(input_ids))
        x = x + self.position_embedding[: x.size(1)].unsqueeze(0)
        causal_mask = torch.triu(
            torch.ones(x.size(1), x.size(1), device=x.device, dtype=torch.bool), diagonal=1
        )
        hidden = self.encoder(x, mask=causal_mask, src_key_padding_mask=input_ids.eq(self.embedding.pad_id))
        return self.output(hidden), hidden


def build_model(arch: str, vocab: PhonemeVocab, frozen_matrix: torch.Tensor | None) -> nn.Module:
    """An LSTM or Transformer of the configured size. frozen_matrix is None for the
    emb condition and the condition's fixed input matrix otherwise."""
    if arch == "lstm":
        return LSTMPhonemeLM(
            vocab_size=len(vocab),
            embedding_dim=TRAIN.embedding_dim,
            hidden_size=TRAIN.hidden_size,
            pad_id=vocab.pad_id,
            init_std=TRAIN.embedding_init_std,
            frozen_matrix=frozen_matrix,
        )
    return CausalTransformerPhonemeLM(
        vocab_size=len(vocab),
        embedding_dim=TRAIN.embedding_dim,
        d_model=TRANSFORMER.d_model,
        n_heads=TRANSFORMER.n_heads,
        num_layers=ARCH_LAYERS[arch],
        dim_feedforward=TRANSFORMER.dim_feedforward,
        dropout=TRANSFORMER.dropout,
        pad_id=vocab.pad_id,
        init_std=TRAIN.embedding_init_std,
        frozen_matrix=frozen_matrix,
        max_len=TRANSFORMER.max_len,
    )
