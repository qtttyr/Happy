"""Train the from-scratch Russian dialogue LLM on Kaggle, with crash-safe checkpoints.

Persistence:
  * model-only checkpoint every `--save-every` steps -> llm/checkpoints/{run}/latest.pt
  * push to HF every `--push-every` steps          -> {run}/ckpt/latest.pt
  * final model + tokenizer + metrics + DONE       -> {run}/...
  * `--resume` pulls the last checkpoint from HF and continues from `step`

    python llm/train.py --hf-repo vaprooll/Happy-LLM --run-name llm-main --resume --max-hours 8
    python llm/train.py --smoke --data-dir llm/data_smoke --no-push
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.model import GPT, GPTConfig, make_model  # noqa: E402
from llm.tokenizer_utils import decode, load_tokenizer, special_id  # noqa: E402
from src.hf_store import HFStore  # noqa: E402

PROMPTS = [
    "Ты кто?",
    "Как дела?",
    "Что делаешь?",
    "Мне скучно",
    "Расскажи что-нибудь",
]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def get_batch(data: np.memmap, block: int, batch: int, device: torch.device):
    ix = torch.randint(len(data) - block - 1, (batch,))
    x = np.stack([data[i : i + block].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block].astype(np.int64) for i in ix])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def estimate_loss(model, data, block, batch, device, batches=20):
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, block, batch, device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            _, loss = model(x, y)
        losses.append(float(loss))
    model.train()
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def sample(model, tokenizer, prompt, device, max_new_tokens=40):
    user = special_id(tokenizer, "<|user|>")
    bot = special_id(tokenizer, "<|bot|>")
    end = special_id(tokenizer, "<|end|>")
    ids = [user] + tokenizer.encode(prompt).ids + [bot]
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    out, score = model.generate_scored(
        tensor, max_new_tokens, temperature=0.8, top_k=40, stop_id=end
    )
    return decode(tokenizer, out[0].tolist()[len(ids):]), score


def get_lr(step: int, total_steps: int, lr: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return lr * (step + 1) / max(warmup, 1)
    if step >= total_steps:
        return min_lr
    ratio = (step - warmup) / max(total_steps - warmup, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * ratio))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="llm/data")
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--run-name", default="llm-main")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--push-every", type=int, default=1000)
    parser.add_argument("--size", default="small", choices=["smoke", "small", "medium"])
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    if args.smoke:
        args.size = "smoke"
        args.max_steps = args.max_steps or 20
        args.eval_every = 5
        args.save_every = 10
        args.push_every = 10
        args.max_hours = min(args.max_hours, 0.2)
        args.batch_size = min(args.batch_size, 8)
        args.grad_accum = 1
        args.block_size = 64

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 4))

    data_dir = Path(args.data_dir)
    train_path, val_path = data_dir / "train.bin", data_dir / "val.bin"
    tok_path = data_dir / "tokenizer.json"
    if not train_path.exists() or not tok_path.exists():
        raise SystemExit(f"missing data in {data_dir}; run llm/prepare_data.py first")

    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8")) if (data_dir / "meta.json").exists() else {}
    tokenizer = load_tokenizer(tok_path)
    vocab_size = tokenizer.get_vocab_size()
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r") if val_path.exists() else train_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} vocab={vocab_size} train_tokens={len(train_data)/1e6:.1f}M")

    ckpt_dir = ROOT / "llm" / "checkpoints" / args.run_name
    final_dir = ROOT / "llm" / "artifacts" / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "latest.pt"

    store = None
    if args.hf_repo and not args.no_push:
        store = HFStore(args.hf_repo, run=args.run_name)
        log(f"hf push: {'ON -> ' + store.repo if store.enabled else 'OFF'}")

    model = make_model(vocab_size, args.size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_step, best_val = 0, float("inf")

    if args.resume:
        if not ckpt_path.exists() and store:
            store.pull(f"{args.run_name}/ckpt/latest.pt", ckpt_path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            cfg = GPTConfig(**ckpt["config"])
            if cfg.vocab_size == vocab_size:
                model = GPT(cfg).to(device)
                model.load_state_dict(ckpt["model"])
                start_step = int(ckpt.get("step", 0))
                best_val = float(ckpt.get("best_val", float("inf")))
                if ckpt.get("optimizer"):
                    optimizer.load_state_dict(ckpt["optimizer"])
                if ckpt.get("scaler") and device.type == "cuda":
                    scaler.load_state_dict(ckpt["scaler"])
                log(f"resumed from step {start_step}, best_val={best_val:.3f}")
            else:
                log("checkpoint vocab mismatch; starting fresh")

    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    total_steps = args.max_steps or max(
        1, int(args.epochs * len(train_data) / max(tokens_per_step, 1))
    )
    log(f"tokens/step={tokens_per_step} total_steps={total_steps} params={model.num_params()/1e6:.1f}M")

    started = time.time()
    step = start_step
    last_loss = float("inf")
    model.train()

    while step < total_steps:
        if (time.time() - started) / 3600 >= args.max_hours:
            log("max-hours reached, stopping gracefully")
            break
        lr = get_lr(step, total_steps, args.lr, args.min_lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, args.block_size, args.batch_size, device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            step_loss += loss.detach().item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        last_loss = step_loss
        step += 1

        if step % 20 == 0:
            elapsed = max(time.time() - started, 1e-6)
            log(
                f"step {step}/{total_steps} loss={last_loss:.3f} lr={lr:.2e} "
                f"tok/s={step*tokens_per_step/elapsed/1000:.0f}k"
            )

        if step % args.eval_every == 0:
            val_loss = estimate_loss(model, val_data, args.block_size, args.batch_size, device)
            log(f"eval step {step}: val_loss={val_loss:.3f} ppl={math.exp(min(val_loss, 20)):.1f}")
            if val_loss < best_val:
                best_val = val_loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if step % args.save_every == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if device.type == "cuda" else None,
                "step": step,
                "best_val": best_val,
                "config": asdict(model.cfg),
                "vocab_size": vocab_size,
            }
            tmp = ckpt_path.with_suffix(".tmp")
            torch.save(checkpoint, tmp)
            tmp.replace(ckpt_path)
            log(f"checkpoint saved: step {step} ({ckpt_path.stat().st_size/1e6:.0f} MB)")
            if store and step % args.push_every == 0:
                store.push(ckpt_path, f"{args.run_name}/ckpt/latest.pt")

    # Final artifacts
    final_model = {
        "model": model.state_dict(),
        "config": asdict(model.cfg),
        "vocab_size": vocab_size,
        "step": step,
    }
    torch.save(final_model, final_dir / "llm_model.pt")

    samples = {}
    model.eval()
    for prompt in PROMPTS:
        try:
            text, score = sample(model, tokenizer, prompt, device)
            samples[prompt] = {"text": text, "score": round(score, 2)}
        except Exception as exc:
            samples[prompt] = f"ERROR: {exc!r}"

    metrics = {
        "run": args.run_name,
        "steps": step,
        "tokens_seen": step * tokens_per_step,
        "train_loss": round(last_loss, 4),
        "val_loss": round(best_val, 4) if best_val < float("inf") else None,
        "perplexity": round(math.exp(min(best_val, 20)), 2) if best_val < float("inf") else None,
        "params": model.num_params(),
        "config": asdict(model.cfg),
        "data": meta,
        "samples": samples,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (final_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (final_dir / "samples.txt").write_text(
        "\n\n".join(f"{q}\n-> {v['text']}  [{v['score']}]" for q, v in samples.items()), encoding="utf-8"
    )
    (final_dir / "config.json").write_text(
        json.dumps(asdict(model.cfg), indent=2), encoding="utf-8"
    )

    if store:
        for name in ("llm_model.pt", "metrics.json", "samples.txt", "config.json"):
            store.push(final_dir / name, f"{args.run_name}/{name}")
        store.push(tok_path, f"{args.run_name}/tokenizer.json")
        (final_dir / "DONE").write_text(
            json.dumps({"run": args.run_name, "metrics": metrics}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        store.push(final_dir / "DONE", f"{args.run_name}/DONE")
        (final_dir / "latest.json").write_text(json.dumps({"run": args.run_name}), encoding="utf-8")
        store.push(final_dir / "latest.json", "latest.json")
        store.push_manifest(
            [final_dir / n for n in ("llm_model.pt", "metrics.json", "samples.txt", "config.json")],
            dest=final_dir / "manifest.json",
            extra={"metrics": metrics},
        )
    log("done")
    log(json.dumps(samples, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
