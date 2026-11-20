"""Download and filter open Russian dialogue pairs for contrastive training.

Only used to train the bi-encoder. These pairs are never returned as answers:
the runtime answers come from the curated base, the composer and the solvers.

    python scripts/fetch_data.py --max-pairs 100000
    python scripts/fetch_data.py --dataset Den4ikAI/russian_dialogues --keep-raw

The raw 1.4 GB file is deleted after filtering, so /kaggle/working stays small.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BAD = ("http://", "https://", "www.", "t.me/", "@", "telegram", "whatsapp")
PROFANITY = re.compile(
    r"(хуй|хуе|хуё|пизд|бля|ебал|ебан|ебат|сука|сук[аи]|мудак|мудил|гандон|залуп|дроч|шлюх|порн)",
    re.IGNORECASE,
)
CYRILLIC = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")


def cyrillic_ratio(text: str) -> float:
    letters = [ch for ch in text.lower() if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(ch in CYRILLIC for ch in letters) / len(letters)


def keep(q: str, a: str) -> bool:
    if not (6 <= len(q) <= 120 and 8 <= len(a) <= 200):
        return False
    if a.endswith("?"):
        return False
    if cyrillic_ratio(q) < 0.7 or cyrillic_ratio(a) < 0.7:
        return False
    text = (q + " " + a).lower()
    if any(bad in text for bad in BAD):
        return False
    if PROFANITY.search(text):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Den4ikAI/russian_dialogues")
    parser.add_argument("--max-pairs", type=int, default=100000)
    parser.add_argument("--out", default="data/dialogues/russian_dialogues.jsonl")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--raw-dir", default=None)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    raw_dir = Path(args.raw_dir or ("/kaggle/tmp/happy-raw" if Path("/kaggle").exists() else "data/raw"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {args.dataset} -> {raw_dir}")
    raw = Path(
        hf_hub_download(
            repo_id=args.dataset,
            filename="dataset.jsonl",
            repo_type="dataset",
            local_dir=str(raw_dir),
        )
    )
    print(f"raw: {raw.stat().st_size / 1e9:.2f} GB")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    seen: set[tuple[str, str]] = set()
    with raw.open("r", encoding="utf-8", errors="ignore") as src, out.open(
        "w", encoding="utf-8"
    ) as dst:
        for i, line in enumerate(src, 1):
            if kept >= args.max_pairs:
                break
            try:
                row = json.loads(line)
            except Exception:
                continue
            if int(row.get("relevance", 1)) != 1:
                continue
            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()
            if not keep(q, a):
                continue
            key = (q, a)
            if key in seen:
                continue
            seen.add(key)
            dst.write(json.dumps({"question": q, "answer": a}, ensure_ascii=False) + "\n")
            kept += 1
            if i % 200000 == 0:
                print(f"  scanned {i:,} lines, kept {kept:,}", flush=True)

    print(f"kept {kept} pairs -> {out}")
    print(f"size: {out.stat().st_size / 1e6:.1f} MB")
    if not args.keep_raw:
        raw.unlink(missing_ok=True)
        print("raw file removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
