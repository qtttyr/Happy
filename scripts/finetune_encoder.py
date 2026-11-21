"""Fine-tune the bi-encoder on curated paraphrase pairs after dialogue pretraining.

Why: training on 100k noisy dialogue pairs improves general semantics but can
wash out fine-grained curated retrieval. This stage runs a few epochs on the
clean pairs only, with a lower LR, and keeps the best checkpoint by Recall@1.

    python scripts/finetune_encoder.py --out artifacts --epochs 8 --lr 3e-4
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
import torch.nn.functional as F  # noqa: E402
from torch.optim import AdamW  # noqa: E402

from scripts.train import build_pairs, collate, evaluate_retrieval, load_corpus, log  # noqa: E402
from src.models import BiEncoder  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, __import__("os").cpu_count() or 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)

    tokenizer = BPETokenizer.load(out / "vocab.json")
    checkpoint = torch.load(out / "bi_encoder.pt", map_location=device, weights_only=False)
    model = BiEncoder(tokenizer.vocab_size, **checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    log(f"loaded encoder: vocab={tokenizer.vocab_size} device={device}")

    corpus = load_corpus()
    pairs = build_pairs(corpus, [])
    log(f"curated pairs: {len(pairs)}")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_recall = 0.0
    best_recall5 = 0.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history = []

    for epoch in range(args.epochs):
        model.train()
        random.Random(args.seed + epoch).shuffle(pairs)
        total, seen = 0.0, 0
        for i in range(0, len(pairs), args.batch_size):
            batch = pairs[i : i + args.batch_size]
            if len(batch) < 2:
                continue
            anchors = collate(tokenizer, [b[0] for b in batch], device)
            positives = collate(tokenizer, [b[1] for b in batch], device)
            za, zp = model(anchors), model(positives)
            logits = za @ zp.T / args.temperature
            target = torch.arange(len(batch), device=device)
            loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
            seen += len(batch)

        metrics = evaluate_retrieval(corpus, model, tokenizer, device, args.seed)
        recall = metrics["retrieval_recall@1"]
        epoch_loss = total / max(seen, 1)
        log(
            f"epoch {epoch + 1}/{args.epochs}: loss={epoch_loss:.3f} "
            f"recall@1={recall:.3f} recall@5={metrics['retrieval_recall@5']:.3f}"
        )
        history.append({"epoch": epoch + 1, "loss": round(epoch_loss, 4), **metrics})

        if recall >= best_recall:
            best_recall = recall
            best_recall5 = metrics["retrieval_recall@5"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    checkpoint["model"] = model.state_dict()
    torch.save(checkpoint, out / "bi_encoder.pt")

    vectors = []
    with torch.no_grad():
        for i in range(0, len(corpus), 128):
            batch = collate(tokenizer, [item.get("q", "") for item in corpus[i : i + 128]], device)
            vectors.append(model(batch).cpu().numpy().astype(np.float16))
    np.save(out / "corpus_vectors.npy", np.vstack(vectors))

    metrics_path = out / "training_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    metrics["retrieval_recall@1"] = round(best_recall, 4)
    metrics["retrieval_recall@5"] = round(best_recall5, 4)
    if history:
        metrics["encoder_loss"] = history[-1]["loss"]
    metrics["finetuned"] = True
    metrics["finetune"] = {
        "epochs": args.epochs,
        "lr": args.lr,
        "best_recall@1": round(best_recall, 4),
        "best_recall@5": round(best_recall5, 4),
        "history": history,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"saved: {out/'bi_encoder.pt'} (best recall@1={best_recall:.3f})")


if __name__ == "__main__":
    main()
