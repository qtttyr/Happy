"""Hugging Face artifact store with retries, resume and checksums.

Kaggle sessions die. This module is the reason a dead session never costs more
than one epoch: every checkpoint is pushed to the Hub immediately, and the
local copy stays in /kaggle/working as a second copy.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    try:  # Kaggle Secrets
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


class HFStore:
    def __init__(self, repo: str | None, token: str | None = None, run: str = "run") -> None:
        self.repo = repo
        self.run = run
        self.token = token or get_token()
        self.enabled = bool(repo and self.token)
        self.api = None
        self.last_error: str | None = None
        if self.enabled:
            try:
                from huggingface_hub import HfApi

                self.api = HfApi(token=self.token)
                self.api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
            except Exception as exc:  # keep training even if the Hub is down
                self.enabled = False
                self.last_error = repr(exc)

    # ------------------------------------------------------------------ upload
    def push(self, local: str | Path, path_in_repo: str | None = None, retries: int = 5) -> bool:
        if not self.enabled or self.api is None:
            return False
        local = Path(local)
        if not local.exists():
            return False
        target = path_in_repo or f"{self.run}/{local.name}"
        for attempt in range(retries):
            try:
                self.api.upload_file(
                    path_or_fileobj=str(local),
                    path_in_repo=target,
                    repo_id=self.repo,
                    repo_type="model",
                    commit_message=f"{self.run}: {local.name}",
                )
                return True
            except Exception as exc:
                self.last_error = repr(exc)
                time.sleep(min(30, 2 ** attempt))
        print(f"[hf] FAILED after {retries} tries: {target} ({self.last_error})")
        return False

    def push_dir(self, local_dir: str | Path, path_in_repo: str | None = None) -> bool:
        if not self.enabled or self.api is None:
            return False
        try:
            self.api.upload_folder(
                folder_path=str(local_dir),
                path_in_repo=path_in_repo or self.run,
                repo_id=self.repo,
                repo_type="model",
                commit_message=f"{self.run}: folder {path_in_repo or self.run}",
            )
            return True
        except Exception as exc:
            self.last_error = repr(exc)
            print(f"[hf] folder upload failed: {exc}")
            return False

    def pull(self, path_in_repo: str, local: str | Path) -> bool:
        if not self.enabled:
            return False
        try:
            from huggingface_hub import hf_hub_download

            local = Path(local)
            local.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=self.repo,
                filename=path_in_repo,
                repo_type="model",
                token=self.token,
                local_dir=str(local.parent),
            )
            if Path(downloaded) != local:
                Path(downloaded).replace(local)
            return True
        except Exception as exc:
            self.last_error = repr(exc)
            return False

    # ---------------------------------------------------------------- manifests
    def push_manifest(self, files: list[str | Path], dest: str | None = None, extra: dict | None = None) -> dict:
        payload = {
            "run": self.run,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "repo": self.repo,
            "files": {},
            **(extra or {}),
        }
        for path in files:
            path = Path(path)
            if path.exists():
                payload["files"][path.name] = {
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
        local = Path(dest or "manifest.json")
        local.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.push(local, f"{self.run}/manifest.json")
        return payload
