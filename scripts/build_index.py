"""Build the BM25 index + frozen corpus used by the runtime.

Usage:
    python scripts/build_index.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.index import BM25Index  # noqa: E402
from src.nlp import tokens  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    sources = sorted((ROOT / "data").glob("*_qa.jsonl"))
    items: list[dict] = []
    for path in sources:
        chunk = read_jsonl(path)
        if chunk:
            print(f"{path.name}: {len(chunk)} entries")
        items.extend(chunk)

    seen: set[str] = set()
    unique: list[dict] = []
    for item in items:
        key = item.get("q", "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    items = unique

    started = time.perf_counter()
    docs = [
        tokens(" ".join([it.get("q", ""), *it.get("variants", []), *it.get("tags", [])]))
        for it in items
    ]
    index = BM25Index().build(docs)
    index.save(ROOT / "artifacts" / "bm25.pkl")

    with (ROOT / "artifacts" / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - started
    print(f"indexed {index.n_docs} docs, {len(index.postings)} terms, avgdl={index.avgdl:.1f}")
    print(f"built in {elapsed:.2f} s -> artifacts/bm25.pkl")


if __name__ == "__main__":
    main()
