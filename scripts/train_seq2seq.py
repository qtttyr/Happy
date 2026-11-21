"""Train the tiny seq2seq fallback from scratch.

    python scripts/train_seq2seq.py --max-pairs 50000 --epochs 3

Data: filtered Russian dialogues + curated pairs. No pretrained weights.
The model is saved to artifacts/seq2seq.pt and used only behind a quality gate.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.optim import Adam  # noqa: E402

from src.seq2seq import BOS, PAD, Seq2Seq  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402

CYRILLIC = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")


def cyr(text: str) -> float:
    letters = [ch for ch in text.lower() if ch.isalpha()]
    return sum(ch in CYRILLIC for ch in letters) / max(len(letters), 1)


def clean_enough(answer: str) -> bool:
    low = answer.lower()
    words = low.split()
    if len(words) < 2:
        return False
    if any(words.count(word) > 2 for word in set(words) if len(word) > 2):
        return False
    if "не не" in low or ")))" in low or "аха" in low:
        return False
    if any(bad in low for bad in ("http", "@", "бот", "нейросет", "ии")):
        return False
    return 10 <= len(answer) <= 100


def load_pairs(max_pairs: int, strict: bool = False) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(q: str, a: str) -> None:
        q, a = (q or "").strip(), (a or "").strip()
        if not (3 <= len(q) <= 120 and 6 <= len(a) <= 200):
            return
        if cyr(q) < 0.6 or cyr(a) < 0.6:
            return
        if strict and not clean_enough(a):
            return
        if (q, a) in seen:
            return
        seen.add((q, a))
        pairs.append((q, a))

    for path in sorted((ROOT / "data" / "dialogues").glob("*.jsonl")):
        print("dialogues:", path.name)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if len(pairs) >= max_pairs:
                    break
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                add(row.get("question", ""), row.get("answer", ""))

    for path in sorted((ROOT / "data").glob("*_qa.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if len(pairs) >= max_pairs:
                    break
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                add(row.get("q", ""), row.get("a", ""))
                for variant in row.get("variants", []):
                    add(variant, row.get("a", ""))

    random.Random(13).shuffle(pairs)
    return pairs[:max_pairs]


def make_batch(pairs, tokenizer, max_len: int, device):
    src_ids = [tokenizer.encode(q, max_len=max_len) for q, _ in pairs]
    tgt_ids = [
        [BOS] + tokenizer.encode(a, max_len=max_len - 1)[: max_len - 1] for _, a in pairs
    ]
    src = np.zeros((len(pairs), max(len(x) for x in src_ids)), dtype=np.int64)
    tgt = np.zeros((len(pairs), max(len(x) for x in tgt_ids)), dtype=np.int64)
    for i, ids in enumerate(src_ids):
        src[i, : len(ids)] = ids
    for i, ids in enumerate(tgt_ids):
        tgt[i, : len(ids)] = ids
    return torch.from_numpy(src).to(device), torch.from_numpy(tgt).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb", type=int, default=128)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, __import__("os").cpu_count() or 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)

    tokenizer = BPETokenizer.load(out / "vocab.json")
    pairs = load_pairs(args.max_pairs, strict=args.strict)
    print(f"pairs: {len(pairs)} | vocab: {tokenizer.vocab_size} | device: {device}")
    if len(pairs) < 1000:
        print("not enough pairs; run scripts/fetch_data.py first")
        return

    model = Seq2Seq(tokenizer.vocab_size, emb=args.emb, hid=args.hid, dropout=0.1).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    split = int(len(pairs) * 0.95)
    train, val = pairs[:split], pairs[split:]

    for epoch in range(args.epochs):
        model.train()
        random.Random(args.seed + epoch).shuffle(train)
        total, seen = 0.0, 0
        started = time.time()
        for i in range(0, len(train), args.batch_size):
            batch = train[i : i + args.batch_size]
            if len(batch) < 2:
                continue
            src, tgt = make_batch(batch, tokenizer, args.max_len, device)
            optimizer.zero_grad()
            loss = model(src, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
            seen += len(batch)
            if (i // args.batch_size) % 100 == 0:
                print(f"  epoch {epoch+1} batch {i//args.batch_size}: loss={loss.item():.3f}", flush=True)
        train_loss = total / max(seen, 1)

        model.eval()
        val_loss, val_seen = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val), args.batch_size):
                batch = val[i : i + args.batch_size]
                src, tgt = make_batch(batch, tokenizer, args.max_len, device)
                val_loss += model(src, tgt).item() * len(batch)
                val_seen += len(batch)
        print(
            f"epoch {epoch+1}/{args.epochs}: train={train_loss:.3f} "
            f"val={val_loss/max(val_seen,1):.3f} time={time.time()-started:.0f}s",
            flush=True,
        )

    checkpoint = {
        "model": model.state_dict(),
        "config": {"emb": args.emb, "hid": args.hid, "dropout": 0.1},
        "vocab_size": tokenizer.vocab_size,
        "pairs": len(pairs),
    }
    torch.save(checkpoint, out / "seq2seq.pt")
    print(f"saved: {out/'seq2seq.pt'}")

    model.eval()
    for question in ["Привет", "Как дела?", "Что делаешь?", "Мне скучно", "Расскажи что-нибудь"]:
        ids = tokenizer.encode(question, max_len=24)
        tensor = torch.tensor([ids], dtype=torch.long)
        out_ids, score = model.generate(tensor, max_len=20, temperature=0.0)
        print(f"  {question} -> {tokenizer.decode(out_ids)!r} ({score:.2f})")


if __name__ == "__main__":
    main()
