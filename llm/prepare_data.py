"""Prepare a Russian casual-dialogue corpus for the from-scratch LLM.

Sources (all MIT / authorial):
  * Den4ikAI/russian_dialogues      - 2M Telegram pairs (MIT)
  * Den4ikAI/russian_dialogues_2    - 1.6M multi-turn Telegram samples (MIT)
  * Happy curated KB                - 1.2k entries / 2.5k formulations (authorial)

Output:
  llm/data/tokenizer.json
  llm/data/train.bin   (uint16 token ids)
  llm/data/val.bin
  llm/data/meta.json

    python llm/prepare_data.py --source local --max-pairs 5000 --smoke
    python llm/prepare_data.py --source hf --max-pairs 600000 --hf-repo vaprooll/Happy-LLM --push
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.tokenizer_utils import encode_turn, load_tokenizer, train_tokenizer  # noqa: E402
from src.hf_store import HFStore  # noqa: E402

CYRILLIC = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
BAD = ("http://", "https://", "www.", "t.me/", "@", "telegram")
PROFANITY = re.compile(
    r"(хуй|хуе|хуё|пизд|бля|ебал|ебан|ебат|сука|сук[аи]|мудак|мудил|гандон|залуп|дроч|шлюх|порн)",
    re.IGNORECASE,
)

IDENTITY = [
    ("Ты кто?", "Мирас, 17. из Астаны, учусь в AITU на первом курсе"),
    ("Кто ты?", "Мирас, 17, Астана. учусь на программиста"),
    ("Как тебя зовут?", "Мирас. а тебя?"),
    ("Сколько тебе лет?", "17, а чё"),
    ("Где ты живёшь?", "в астане, на правом. с родителями"),
    ("Расскажи о себе", "Мирас, 17, Астана. учусь на программиста, люблю музыку и поспать"),
    ("Как дела?", "да норм, сижу. у тебя как"),
    ("Что делаешь?", "да ничё, чай пью. а ты"),
    ("Ты бот?", "не, с чего. я с телефона сижу"),
    ("Чем увлекаешься?", "музыка, игры, иногда код. ничего особенного"),
]


def cyr(text: str) -> float:
    letters = [ch for ch in text.lower() if ch.isalpha()]
    return sum(ch in CYRILLIC for ch in letters) / max(len(letters), 1)


def clean_pair(q: str, a: str) -> bool:
    q, a = q.strip(), a.strip()
    if not (3 <= len(q) <= 120 and 6 <= len(a) <= 200):
        return False
    if cyr(q) < 0.6 or cyr(a) < 0.6:
        return False
    text = (q + " " + a).lower()
    if any(bad in text for bad in BAD):
        return False
    if PROFANITY.search(text):
        return False
    words = a.lower().split()
    if any(words.count(word) > 3 for word in set(words) if len(word) > 2):
        return False
    return True


def add_pair(pairs, seen, q: str, a: str) -> None:
    if not clean_pair(q, a):
        return
    key = (q.lower(), a.lower())
    if key in seen:
        return
    seen.add(key)
    pairs.append((q, a))


def download(dataset: str, filename: str) -> Path | None:
    try:
        from huggingface_hub import hf_hub_download

        raw = ROOT / "llm" / "raw" / dataset.replace("/", "__")
        raw.mkdir(parents=True, exist_ok=True)
        print(f"downloading {dataset}/{filename}")
        return Path(
            hf_hub_download(
                repo_id=dataset, filename=filename, repo_type="dataset", local_dir=str(raw)
            )
        )
    except Exception as exc:
        print(f"download failed: {dataset}: {exc}")
        return None


def read_dataset1(path: Path, limit: int, pairs, seen) -> None:
    kept = 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if kept >= limit:
                break
            try:
                row = json.loads(line)
            except Exception:
                continue
            if int(row.get("relevance", 1)) != 1:
                continue
            before = len(pairs)
            add_pair(pairs, seen, str(row.get("question", "")), str(row.get("answer", "")))
            kept += len(pairs) - before


def read_dataset2(path: Path, limit: int, pairs, seen) -> None:
    kept = 0
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if kept >= limit:
                break
            try:
                row = json.loads(line)
            except Exception:
                continue
            sample = row.get("sample") or []
            for i in range(0, len(sample) - 1, 2):
                before = len(pairs)
                add_pair(pairs, seen, str(sample[i]), str(sample[i + 1]))
                kept += len(pairs) - before
                if kept >= limit:
                    break


def read_curated(pairs, seen) -> None:
    for path in sorted((ROOT / "data").glob("*_qa.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                add_pair(pairs, seen, row.get("q", ""), row.get("a", ""))
                for variant in row.get("variants", []):
                    add_pair(pairs, seen, variant, row.get("a", ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["local", "hf"], default="local")
    parser.add_argument("--max-pairs", type=int, default=600000)
    parser.add_argument("--max-tokens", type=int, default=80_000_000)
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--out", default="llm/data")
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.max_pairs = min(args.max_pairs, 3000)
        args.max_tokens = min(args.max_tokens, 2_000_000)
        args.vocab_size = min(args.vocab_size, 4000)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_path, val_path = out / "train.bin", out / "val.bin"
    tok_path, meta_path = out / "tokenizer.json", out / "meta.json"

    store = HFStore(args.hf_repo, run="data") if args.hf_repo else None

    if args.pull and store:
        ok = all(
            store.pull(f"data/{name}", out / name)
            for name in ("train.bin", "val.bin", "tokenizer.json", "meta.json")
        )
        if ok:
            print("pulled prepared data from HF")
            return 0
        print("remote data missing, building locally")

    if args.reuse and train_path.exists() and tok_path.exists():
        print("reusing local bins")
        return 0

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    if args.source == "hf":
        d1 = download("Den4ikAI/russian_dialogues", "dataset.jsonl")
        if d1:
            read_dataset1(d1, args.max_pairs // 2, pairs, seen)
        d2 = download("Den4ikAI/russian_dialogues_2", "dataset.jsonl.gz")
        if d2:
            read_dataset2(d2, args.max_pairs // 2, pairs, seen)
    else:
        local = ROOT / "data" / "dialogues" / "russian_dialogues.jsonl"
        if local.exists():
            with local.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    add_pair(pairs, seen, row.get("question", ""), row.get("answer", ""))
                    if len(pairs) >= args.max_pairs:
                        break

    read_curated(pairs, seen)
    # Identity emphasis: repeat after dedupe so the model hears the persona often.
    for _ in range(30):
        pairs.extend(IDENTITY)

    random.Random(13).shuffle(pairs)
    pairs = pairs[: args.max_pairs]
    print(f"pairs: {len(pairs)}")

    if tok_path.exists():
        tokenizer = load_tokenizer(tok_path)
    else:
        texts = (f"{q} {a}" for q, a in pairs[:30000])
        tokenizer = train_tokenizer(texts, args.vocab_size, tok_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"vocab: {vocab_size}")

    ids: list[int] = []
    started = time.time()
    for index, (q, a) in enumerate(pairs):
        ids.extend(encode_turn(tokenizer, q, a))
        if len(ids) >= args.max_tokens:
            break
        if index % 50000 == 0 and index:
            print(f"  tokenized {index} pairs, {len(ids)/1e6:.1f}M tokens", flush=True)
    print(f"tokens: {len(ids)/1e6:.1f}M in {time.time()-started:.0f}s")

    array = np.array(ids, dtype=np.uint16)
    split = int(len(array) * 0.99)
    array[:split].tofile(train_path)
    array[split:].tofile(val_path)

    meta = {
        "pairs": len(pairs),
        "tokens": int(len(array)),
        "train_tokens": int(split),
        "val_tokens": int(len(array) - split),
        "vocab_size": int(vocab_size),
        "sources": ["Den4ikAI/russian_dialogues", "Den4ikAI/russian_dialogues_2", "Happy curated"],
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    if args.push and store:
        for path in (train_path, val_path, tok_path, meta_path):
            store.push(path, f"data/{path.name}")
        print("pushed data to HF")
    if not args.keep_raw:
        shutil.rmtree(ROOT / "llm" / "raw", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
