"""Pull the trained LLM artifacts from Hugging Face.

    python llm/pull.py --repo vaprooll/Happy-LLM --run llm-main
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hf_store import HFStore  # noqa: E402

FILES = ["llm_model.pt", "tokenizer.json", "metrics.json", "samples.txt", "config.json"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--run", default="llm-main")
    parser.add_argument("--dest", default=None)
    args = parser.parse_args()

    dest = Path(args.dest or f"llm/artifacts/{args.run}")
    dest.mkdir(parents=True, exist_ok=True)
    store = HFStore(args.repo, run=args.run)
    if not store.enabled:
        print("HF disabled: set HF_TOKEN")
        return 2
    ok = True
    for name in FILES:
        if not store.pull(f"{args.run}/{name}", dest / name):
            print("missing:", name)
            ok = False
    print("OK" if ok else "PARTIAL")
    print(f"dest: {dest}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
