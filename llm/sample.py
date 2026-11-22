"""Sample from the trained Happy LLM and apply a sanity gate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm.model import GPT, GPTConfig  # noqa: E402
from llm.tokenizer_utils import decode, load_tokenizer, special_id  # noqa: E402

BANNED = ("http", "<", ">", "бот", "нейросет", "chatgpt", "gpt")


def cyr(text: str) -> float:
    letters = [ch for ch in text.lower() if ch.isalpha()]
    return sum("а" <= ch <= "я" or ch == "ё" for ch in letters) / max(len(letters), 1)


def quality_ok(text: str, score: float) -> bool:
    low = text.lower().strip()
    if not (6 <= len(text) <= 200):
        return False
    if cyr(text) < 0.6:
        return False
    if any(bad in low for bad in BANNED):
        return False
    words = text.split()
    if any(words.count(word) > 3 for word in set(words) if len(word) > 2):
        return False
    if text.count("?") > 1:
        return False
    if score < -2.5:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="llm-main")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default="llm/data/tokenizer.json")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    model_path = Path(args.model or f"llm/artifacts/{args.run}/llm_model.pt")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    tokenizer = load_tokenizer(args.tokenizer)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompts = [args.prompt] if args.prompt else [
        "Ты кто?", "Как дела?", "Что делаешь?", "Мне скучно", "Расскажи что-нибудь"
    ]
    for prompt in prompts:
        user = special_id(tokenizer, "<|user|>")
        bot = special_id(tokenizer, "<|bot|>")
        end = special_id(tokenizer, "<|end|>")
        ids = [user] + tokenizer.encode(prompt).ids + [bot]
        tensor = torch.tensor([ids], dtype=torch.long)
        out, score = model.generate_scored(
            tensor, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, stop_id=end
        )
        text = decode(tokenizer, out[0].tolist()[len(ids):])
        print(f"{prompt}\n-> {text}\n   score={score:.2f} gate={'PASS' if quality_ok(text, score) else 'FAIL'}\n")


if __name__ == "__main__":
    main()
