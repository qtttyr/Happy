"""PyTorch architectures, trained from scratch. No pretrained weights.

IntentClassifier : BiLSTM + attention pooling -> intent logits.
BiEncoder        : shared BiLSTM + mean/max pooling -> 128-d L2-normalised vector.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        emb: int = 128,
        hid: int = 96,
        layers: int = 2,
        dropout: float = 0.2,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            emb,
            hid,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.att = nn.Linear(2 * hid, 1)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hid, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x.ne(0)
        hidden, _ = self.lstm(self.emb(x))
        scores = self.att(hidden).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1)
        return self.out(self.drop(pooled))


class BiEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb: int = 128,
        hid: int = 128,
        layers: int = 1,
        out_dim: int = 128,
        dropout: float = 0.15,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            emb,
            hid,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(4 * hid, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x.ne(0)
        hidden, _ = self.lstm(self.emb(x))
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom
        masked = hidden.masked_fill(~mask.unsqueeze(-1), -1e9)
        maximum = masked.max(dim=1).values
        z = self.proj(self.drop(torch.cat([mean, maximum], dim=-1)))
        return F.normalize(z, dim=-1)
