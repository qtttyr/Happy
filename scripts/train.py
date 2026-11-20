"""Train Happy's models from scratch, with crash-safe checkpointing.

Everything is produced here: the BPE vocabulary, the intent classifier, the
Siamese bi-encoder, the BM25 index and the corpus vectors. No pretrained
weights are loaded and no LLM API is called.

Persistence ladder (a dead Kaggle session costs at most one epoch):
  1. atomic local checkpoint in artifacts/checkpoints/
  2. immediate push to Hugging Face after every epoch
  3. final zip in /kaggle/working + `latest.json` pointer
  4. manifest.json with sha256 for every artifact

Usage (Kaggle):
    python scripts/train.py --hf-repo mkstudio/happy-artifacts --run-name kaggle-main --push-every-epoch

Usage (local smoke test):
    python scripts/train.py --smoke --out artifacts_smoke --no-push
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.optim import AdamW  # noqa: E402

from src.hf_store import HFStore, sha256  # noqa: E402
from src.index import BM25Index  # noqa: E402
from src.intent_data import generate, write_jsonl  # noqa: E402
from src.models import BiEncoder, IntentClassifier  # noqa: E402
from src.nlp import tokens  # noqa: E402
from src.retriever import index_text  # noqa: E402
from src.tokenizer import BPETokenizer  # noqa: E402

BAD = ("http://", "https://", "www.", "t.me/", "@gmail", "+7 (", "telegram")
CYRILLIC = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def cyrillic_ratio(text: str) -> float:
    letters = [ch for ch in text.lower() if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(ch in CYRILLIC for ch in letters) / len(letters)


def clean_pair(q: str, a: str) -> tuple[str, str] | None:
    if not isinstance(q, str) or not isinstance(a, str):
        return None
    q, a = q.strip(), a.strip()
    if not (5 <= len(q) <= 140 and 10 <= len(a) <= 240):
        return None
    if cyrillic_ratio(q) < 0.6 or cyrillic_ratio(a) < 0.6:
        return None
    if any(bad in (q + a).lower() for bad in BAD):
        return None
    if a.endswith("?"):
        return None
    return q, a


def load_corpus() -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()
    for path in sorted((ROOT / "data").glob("*_qa.jsonl")):
        for item in load_jsonl(path):
            key = item.get("q", "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                items.append(item)
    return items


def load_dialogues(limit: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted((ROOT / "data" / "dialogues").glob("*.jsonl")):
        log(f"dialogues: reading {path.name}")
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if len(pairs) >= limit:
                    break
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                q = row.get("question") or row.get("q") or row.get("prompt") or ""
                a = row.get("answer") or row.get("a") or row.get("response") or ""
                pair = clean_pair(q, a)
                if pair and pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)
        log(f"dialogues: kept {len(pairs)} pairs so far")
    return pairs


def collate(tokenizer: BPETokenizer, texts: list[str], device: torch.device) -> torch.Tensor:
    encoded = [tokenizer.encode(t) for t in texts]
    width = max(len(x) for x in encoded)
    batch = np.zeros((len(encoded), width), dtype=np.int64)
    for i, ids in enumerate(encoded):
        batch[i, : len(ids)] = ids
    return torch.from_numpy(batch).to(device)


def split(rows: list[dict], ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    data = rows[:]
    rng.shuffle(data)
    cut = max(1, int(len(data) * ratio))
    return data[:cut], data[cut:]


# --------------------------------------------------------------------------- intent
def train_intent(
    rows: list[dict],
    tokenizer: BPETokenizer,
    device: torch.device,
    args: argparse.Namespace,
    store: HFStore,
    out: Path,
) -> dict:
    labels = sorted({r["label"] for r in rows})
    label2id = {label: i for i, label in enumerate(labels)}
    train_rows, val_rows = split(rows, 0.9, args.seed)
    log(f"intent: {len(train_rows)} train / {len(val_rows)} val, {len(labels)} classes")

    model = IntentClassifier(
        tokenizer.vocab_size,
        len(labels),
        emb=args.emb,
        hid=args.hid,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_acc, start_epoch = 0.0, 0
    ckpt_path = out / "checkpoints" / "intent_last.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        if not ckpt_path.exists():
            store.pull(f"{args.run_name}/checkpoints/intent_last.pt", ckpt_path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if ckpt.get("vocab_size") not in (None, tokenizer.vocab_size):
                log("intent: checkpoint vocab mismatch, starting fresh")
                ckpt = None
            if ckpt:
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                best_acc, start_epoch = ckpt.get("best", 0.0), ckpt.get("epoch", 0) + 1
                log(f"intent: resumed from epoch {start_epoch}, best={best_acc:.4f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        random.Random(args.seed + epoch).shuffle(train_rows)
        total, seen = 0.0, 0
        for i in range(0, len(train_rows), args.batch_size):
            batch_rows = train_rows[i : i + args.batch_size]
            x = collate(tokenizer, [r["text"] for r in batch_rows], device)
            y = torch.tensor([label2id[r["label"]] for r in batch_rows], device=device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch_rows)
            seen += len(batch_rows)
        train_loss = total / max(seen, 1)

        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(0, len(val_rows), args.batch_size):
                batch_rows = val_rows[i : i + args.batch_size]
                x = collate(tokenizer, [r["text"] for r in batch_rows], device)
                y = torch.tensor([label2id[r["label"]] for r in batch_rows], device=device)
                pred = model(x).argmax(dim=-1)
                correct += int((pred == y).sum())
        accuracy = correct / max(len(val_rows), 1)
        log(f"intent epoch {epoch + 1}/{args.epochs}: loss={train_loss:.3f} acc={accuracy:.3f}")

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best": max(best_acc, accuracy),
            "labels": labels,
            "vocab_size": tokenizer.vocab_size,
            "config": {
                "emb": args.emb, "hid": args.hid, "layers": args.layers, "dropout": args.dropout
            },
        }
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(ckpt_path)
        if args.push_every_epoch:
            store.push(ckpt_path, f"{args.run_name}/checkpoints/intent_last.pt")

        if accuracy >= best_acc:
            best_acc = accuracy
            torch.save(state, out / "intent_model.pt")
            (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
            if args.push_every_epoch:
                store.push(out / "intent_model.pt", f"{args.run_name}/intent_model.pt")
                store.push(out / "labels.json", f"{args.run_name}/labels.json")

    return {"intent_accuracy": round(best_acc, 4), "intent_labels": labels}


# -------------------------------------------------------------------------- encoder
def build_pairs(corpus: list[dict], dialogues: list[tuple[str, str]]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(a: str, b: str) -> None:
        if a and b and a != b and (a, b) not in seen:
            seen.add((a, b))
            pairs.append((a, b))

    for item in corpus:
        q = item.get("q", "")
        variants = [v for v in item.get("variants", []) if v]
        answer = item.get("a", "")
        for v in variants:
            add(q, v)
            add(v, q)
        for i, a in enumerate(variants):
            for b in variants[i + 1 :]:
                add(a, b)
                add(b, a)
        if answer:
            add(q, answer)

    for q, a in dialogues:
        add(q, a)
    return pairs


def train_encoder(
    pairs: list[tuple[str, str]],
    tokenizer: BPETokenizer,
    device: torch.device,
    args: argparse.Namespace,
    store: HFStore,
    out: Path,
) -> dict:
    train_pairs = pairs
    model = BiEncoder(
        tokenizer.vocab_size,
        emb=args.emb,
        hid=args.hid,
        layers=1,
        out_dim=args.dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ckpt_path = out / "checkpoints" / "encoder_last.pt"
    start_epoch, best_loss = 0, float("inf")

    if args.resume:
        if not ckpt_path.exists():
            store.pull(f"{args.run_name}/checkpoints/encoder_last.pt", ckpt_path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            if ckpt.get("vocab_size") not in (None, tokenizer.vocab_size):
                log("encoder: checkpoint vocab mismatch, starting fresh")
                ckpt = None
            if ckpt:
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])
                start_epoch, best_loss = ckpt.get("epoch", 0) + 1, ckpt.get("loss", float("inf"))
                log(f"encoder: resumed from epoch {start_epoch}")

    log(f"encoder: {len(train_pairs)} pairs")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        random.Random(args.seed + 100 + epoch).shuffle(train_pairs)
        total, seen = 0.0, 0
        for i in range(0, len(train_pairs), args.batch_size):
            batch = train_pairs[i : i + args.batch_size]
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
        epoch_loss = total / max(seen, 1)
        log(f"encoder epoch {epoch + 1}/{args.epochs}: loss={epoch_loss:.3f}")

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": epoch_loss,
            "vocab_size": tokenizer.vocab_size,
            "config": {
                "emb": args.emb, "hid": args.hid, "layers": 1,
                "out_dim": args.dim, "dropout": args.dropout,
            },
        }
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save(state, tmp)
        tmp.replace(ckpt_path)
        if args.push_every_epoch:
            store.push(ckpt_path, f"{args.run_name}/checkpoints/encoder_last.pt")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(state, out / "bi_encoder.pt")
            if args.push_every_epoch:
                store.push(out / "bi_encoder.pt", f"{args.run_name}/bi_encoder.pt")

    return {"encoder_loss": round(best_loss, 4)}


# ----------------------------------------------------------------------------- eval
def evaluate_retrieval(
    corpus: list[dict], model: BiEncoder, tokenizer: BPETokenizer, device: torch.device, seed: int
) -> dict:
    rng = random.Random(seed)
    ids = list(range(len(corpus)))
    rng.shuffle(ids)
    holdout = set(ids[: max(1, len(ids) // 10)])

    pool_texts: list[str] = []
    pool_ids: list[int] = []
    queries: list[tuple[str, int]] = []
    for i, item in enumerate(corpus):
        variants = [v for v in item.get("variants", []) if v]
        texts = [item.get("q", "")] + variants
        if i in holdout:
            queries.append((variants[0] if variants else item.get("q", ""), i))
        query_text = variants[0] if variants else item.get("q", "")
        for text in texts:
            if i in holdout and text == query_text:
                continue  # do not let the query match itself
            pool_texts.append(text)
            pool_ids.append(i)

    model.eval()
    with torch.no_grad():
        pool_vectors = []
        for i in range(0, len(pool_texts), 128):
            x = collate(tokenizer, pool_texts[i : i + 128], device)
            pool_vectors.append(model(x).cpu())
        pool_matrix = torch.cat(pool_vectors)
        pool_matrix = F.normalize(pool_matrix, dim=-1)

    hits1 = hits5 = 0
    for text, target in queries:
        with torch.no_grad():
            qv = model(collate(tokenizer, [text], device)).cpu()
        scores = (pool_matrix @ qv.T).squeeze(-1)
        top = torch.topk(scores, k=min(5, len(scores))).indices.tolist()
        predicted = [pool_ids[i] for i in top]
        hits1 += int(bool(predicted) and predicted[0] == target)
        hits5 += int(target in predicted)

    return {
        "retrieval_recall@1": round(hits1 / max(len(queries), 1), 4),
        "retrieval_recall@5": round(hits5 / max(len(queries), 1), 4),
        "retrieval_queries": len(queries),
    }


# ---------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--run-name", default="local")
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--push-every-epoch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb", type=int, default=128)
    parser.add_argument("--hid", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--merges", type=int, default=2000)
    parser.add_argument("--max-dialogues", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1
        args.merges = 300
        args.max_dialogues = 500
        args.emb, args.hid, args.layers = 64, 48, 1
        args.batch_size = 16
        args.run_name = args.run_name if args.run_name != "local" else "smoke"
    if args.no_push:
        args.hf_repo = None

    set_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log(f"device={device} torch={torch.__version__}")

    store = HFStore(args.hf_repo, run=args.run_name)
    log(f"hf push: {'ON -> ' + store.repo if store.enabled else 'OFF'}")

    corpus = load_corpus()
    dialogues = load_dialogues(args.max_dialogues)
    log(f"corpus={len(corpus)} curated pairs, dialogues={len(dialogues)}")

    intent_rows = generate(corpus, seed=args.seed)
    write_jsonl(intent_rows, ROOT / "data" / "intents_train.jsonl")
    log(f"intent dataset: {len(intent_rows)} rows")
    if args.smoke:
        intent_rows = intent_rows[:80]

    texts = [item.get("q", "") for item in corpus]
    for item in corpus:
        texts.extend(item.get("variants", []))
        if item.get("a"):
            texts.append(item["a"])
    texts.extend([r["text"] for r in intent_rows])
    texts.extend([t for pair in dialogues[:20000] for t in pair])
    log("tokenizer: training BPE")
    tokenizer = BPETokenizer.train(texts, num_merges=args.merges, min_freq=2, max_len=32)
    tokenizer.save(out / "vocab.json")
    log(f"tokenizer: vocab={tokenizer.vocab_size}")
    store.push(out / "vocab.json", f"{args.run_name}/vocab.json")

    metrics: dict = {
        "corpus_size": len(corpus),
        "dialogue_pairs": len(dialogues),
        "intent_rows": len(intent_rows),
        "vocab_size": tokenizer.vocab_size,
        "device": str(device),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    metrics.update(train_intent(intent_rows, tokenizer, device, args, store, out))
    pairs = build_pairs(corpus, dialogues)
    metrics["contrastive_pairs"] = len(pairs)
    metrics.update(train_encoder(pairs, tokenizer, device, args, store, out))

    # final encoder is the best checkpoint on disk
    ckpt = torch.load(out / "bi_encoder.pt", map_location=device, weights_only=False)
    encoder = BiEncoder(tokenizer.vocab_size, **ckpt["config"]).to(device)
    encoder.load_state_dict(ckpt["model"])
    metrics.update(evaluate_retrieval(corpus, encoder, tokenizer, device, args.seed))

    # BM25 + frozen corpus
    docs = [tokens(index_text(item)) for item in corpus]
    BM25Index().build(docs).save(out / "bm25.pkl")
    with (out / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for item in corpus:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    # corpus vectors for the dense half of the hybrid
    encoder.eval()
    vectors = []
    with torch.no_grad():
        for i in range(0, len(corpus), 128):
            batch = collate(tokenizer, [item.get("q", "") for item in corpus[i : i + 128]], device)
            vectors.append(encoder(batch).cpu().numpy().astype(np.float16))
    np.save(out / "corpus_vectors.npy", np.vstack(vectors))

    (out / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log("metrics: " + json.dumps(metrics, ensure_ascii=False))

    files = [
        out / "vocab.json",
        out / "intent_model.pt",
        out / "bi_encoder.pt",
        out / "labels.json",
        out / "bm25.pkl",
        out / "corpus.jsonl",
        out / "corpus_vectors.npy",
        out / "training_metrics.json",
    ]
    for path in files:
        if path.exists():
            store.push(path, f"{args.run_name}/{path.name}")

    manifest = store.push_manifest(files, dest=out / "manifest.json", extra={"metrics": metrics})
    (out / "DONE").write_text(
        json.dumps({"run": args.run_name, "metrics": metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    store.push(out / "DONE", f"{args.run_name}/DONE")
    if store.enabled:
        (out / "latest.json").write_text(json.dumps({"run": args.run_name}), encoding="utf-8")
        store.push(out / "latest.json", "latest.json")

    archive = shutil.make_archive(
        str(out.parent / f"happy-artifacts-{args.run_name}".replace("/", "-")), "zip", root_dir=out
    )
    store.push(archive, f"{args.run_name}/happy-artifacts.zip")
    log(f"done: {archive} ({sha256(archive)[:12]})")
    log(f"manifest files: {list(manifest['files'])}")


if __name__ == "__main__":
    main()
