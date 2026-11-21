"""Terminal demo - the "Ollama-like" local experience.

    python run_cli.py            # normal mode
    python run_cli.py --pace     # emulate human typing delay
    python run_cli.py --raw      # no humanizer (debug)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.engine import TuringBot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pace", action="store_true", help="human typing pause")
    parser.add_argument("--raw", action="store_true", help="disable humanizer")
    parser.add_argument("--gen", action="store_true", help="enable seq2seq fallback (off by default)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    bot = TuringBot(seed=args.seed, humanize=not args.raw, use_generator=True if args.gen else None)
    print("Happy · локальный режим (offline)")
    print("Команды: /reset, /debug, /quit\n")

    debug = False
    while True:
        try:
            question = input("ты > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question in {"/quit", "/exit", "exit", "quit"}:
            break
        if question == "/reset":
            bot.reset()
            print("— память раунда очищена\n")
            continue
        if question == "/debug":
            debug = not debug
            print(f"— debug: {'on' if debug else 'off'}\n")
            continue

        answer = bot.answer(question, pacing=args.pace)
        if args.pace and answer.typing_delay:
            time.sleep(answer.typing_delay)
        print(f"он > {answer.text}")
        print(
            f"      {answer.latency_ms:.0f} мс · {answer.source} · "
            f"conf {answer.confidence:.2f} · {answer.topic}/{answer.intent}"
        )
        if debug:
            print(f"      debug: {answer.debug}")
        print()


if __name__ == "__main__":
    main()
