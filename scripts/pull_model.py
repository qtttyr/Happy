"""Download the latest trained artifacts from Hugging Face and verify them.

    python scripts/pull_model.py --repo vaprooll/Happy
    python scripts/pull_model.py --repo vaprooll/Happy --run kaggle-main

The command is idempotent: files with a matching sha256 are skipped.
After it finishes, `artifacts/` is ready for a fully offline run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hf_store import get_token, sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Hugging Face repo id, e.g. user/happy-artifacts")
    parser.add_argument("--run", default=None, help="run name; default = latest.json")
    parser.add_argument("--dest", default="artifacts")
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    token = args.token or get_token()
    if not token:
        print("HF_TOKEN не найден. Экспортируй токен: export HF_TOKEN=hf_...")
        return 2

    from huggingface_hub import hf_hub_download

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    run = args.run
    if not run:
        latest = hf_hub_download(
            repo_id=args.repo, filename="latest.json", repo_type="model", token=token
        )
        run = json.loads(Path(latest).read_text(encoding="utf-8"))["run"]
    print(f"repo={args.repo} run={run} -> {dest}")

    manifest_src = hf_hub_download(
        repo_id=args.repo, filename=f"{run}/manifest.json", repo_type="model", token=token
    )
    manifest = json.loads(Path(manifest_src).read_text(encoding="utf-8"))
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ok = True
    for name, meta in manifest.get("files", {}).items():
        target = dest / name
        if target.exists() and sha256(target) == meta["sha256"]:
            print(f"  = {name} (already ok)")
            continue
        print(f"  ↓ {name} ({meta['bytes'] / 1e6:.2f} MB)")
        cached = hf_hub_download(
            repo_id=args.repo, filename=f"{run}/{name}", repo_type="model", token=token
        )
        shutil.copy2(cached, target)
        actual = sha256(target)
        if actual != meta["sha256"]:
            print(f"  ! checksum mismatch: {name}")
            ok = False
        else:
            print(f"  ✓ {name}")

    (dest / "latest.json").write_text(json.dumps({"run": run}), encoding="utf-8")
    print("OK" if ok else "FAILED: checksum mismatch")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
