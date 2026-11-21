"""Push an existing artifacts directory as a complete Hugging Face run.

    python scripts/push_run.py --out artifacts --hf-repo vaprooll/Happy \
        --run-name kaggle-ft --latest

Does not delete anything: it adds a new run prefix and only moves `latest.json`
when `--latest` is passed. The previous run stays intact.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hf_store import HFStore  # noqa: E402

CORE = [
    "vocab.json",
    "intent_model.pt",
    "bi_encoder.pt",
    "labels.json",
    "bm25.pkl",
    "corpus.jsonl",
    "corpus_vectors.npy",
    "training_metrics.json",
]
OPTIONAL = ["seq2seq.pt"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--hf-repo", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    files = [out / name for name in CORE]
    missing = [path.name for path in files if not path.exists()]
    if missing:
        print("missing:", missing)
        return 1
    files.extend(out / name for name in OPTIONAL if (out / name).exists())

    metrics = json.loads((out / "training_metrics.json").read_text(encoding="utf-8"))
    store = HFStore(args.hf_repo, run=args.run_name)
    print(f"hf push: {'ON -> ' + store.repo if store.enabled else 'OFF'}")
    if not store.enabled:
        return 2

    for path in files:
        store.push(path, f"{args.run_name}/{path.name}")
    manifest = store.push_manifest(
        files, dest=out / "manifest.json", extra={"metrics": metrics}
    )

    (out / "DONE").write_text(
        json.dumps({"run": args.run_name, "metrics": metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    store.push(out / "DONE", f"{args.run_name}/DONE")

    archive = shutil.make_archive(
        str(out.parent / f"happy-artifacts-{args.run_name}".replace("/", "-")),
        "zip",
        root_dir=out,
    )
    store.push(archive, f"{args.run_name}/happy-artifacts.zip")

    if args.latest:
        (out / "latest.json").write_text(json.dumps({"run": args.run_name}), encoding="utf-8")
        store.push(out / "latest.json", "latest.json")

    print(f"pushed: {args.run_name} | files: {list(manifest['files'])} | latest={args.latest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
