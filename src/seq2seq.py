"""A small GRU encoder-decoder with attention, trained from scratch.

This is a fallback generator, not the core of the bot. It is used only when
retrieval has no answer and the intent is small talk/opinion, and only if the
generated text passes a strict quality gate. If anything looks off, the normal
human fallback wins.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 0
BOS = 2  # </s> is used as both start and end marker
EOS = 2


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 128, hid: int = 256, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(emb, hid, batch_first=True)

    def forward(self, src: torch.Tensor):
        out, hidden = self.gru(self.drop(self.emb(src)))
        return out, hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 128, hid: int = 256, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(emb + hid, hid, batch_first=True)
        self.att = nn.Linear(hid * 2, 1)
        self.out = nn.Linear(hid * 2, vocab_size)

    def forward(self, inp, hidden, enc_out, mask):
        emb = self.drop(self.emb(inp)).unsqueeze(1)
        query = hidden[-1].unsqueeze(1).expand(-1, enc_out.size(1), -1)
        scores = self.att(torch.cat([enc_out, query], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = (enc_out * weights.unsqueeze(-1)).sum(dim=1)
        gru_in = torch.cat([emb, context.unsqueeze(1)], dim=-1)
        out, hidden = self.gru(gru_in, hidden)
        logits = self.out(torch.cat([out.squeeze(1), context], dim=-1))
        return logits, hidden


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 128, hid: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = Encoder(vocab_size, emb, hid, dropout)
        self.decoder = Decoder(vocab_size, emb, hid, dropout)

    def encode(self, src):
        mask = src.ne(PAD)
        enc_out, hidden = self.encoder(src)
        return enc_out, hidden, mask

    def forward(self, src, tgt, teacher_forcing: float = 0.6):
        enc_out, hidden, mask = self.encode(src)
        inp = tgt[:, 0]
        total = 0.0
        for step in range(1, tgt.size(1)):
            logits, hidden = self.decoder(inp, hidden, enc_out, mask)
            total = total + F.cross_entropy(logits, tgt[:, step], ignore_index=PAD)
            if torch.rand(1).item() < teacher_forcing:
                inp = tgt[:, step]
            else:
                inp = logits.argmax(dim=-1)
        return total / max(tgt.size(1) - 1, 1)

    @torch.no_grad()
    def generate(self, src, max_len: int = 20, temperature: float = 0.0):
        enc_out, hidden, mask = self.encode(src)
        inp = torch.full((src.size(0),), BOS, dtype=torch.long, device=src.device)
        ids: list[int] = []
        logps: list[float] = []
        for _ in range(max_len):
            logits, hidden = self.decoder(inp, hidden, enc_out, mask)
            logp = F.log_softmax(logits, dim=-1)
            if temperature <= 0.01:
                nxt = logp.argmax(dim=-1)
            else:
                nxt = torch.multinomial((logp / temperature).exp(), 1).squeeze(-1)
            token = int(nxt[0])
            if token == EOS:
                break
            ids.append(token)
            logps.append(float(logp[0, token]))
            inp = nxt
        score = sum(logps) / max(len(logps), 1)
        return ids, score
