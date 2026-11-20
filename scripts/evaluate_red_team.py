"""Run the 300 unseen red-team questions and build a review sheet.

    python scripts/evaluate_red_team.py

Outputs:
  results/red_team_results.csv   - answers + empty manual score columns
  results/red_team_summary.json  - coverage by category, latency, tells
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.engine import TuringBot  # noqa: E402

TELLS = (
    "во-первых", "во вторых", "таким образом", "кроме того", "важно отметить",
    "стоит отметить", "как языковая модель", "как ии", "я не могу",
    "отличный вопрос", "безусловно", "в заключение", "резюмируя",
)


def tell_flags(text: str) -> str:
    low = text.lower()
    flags = [t for t in TELLS if t in low]
    if len(text) > 270:
        flags.append("too_long")
    if low.count(".") >= 3 and len(text) > 220:
        flags.append("over_structured")
    if text.endswith(".") and len(text) > 120:
        flags.append("terminal_period")
    return ",".join(flags)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/eval/red_team_300.jsonl")
    parser.add_argument("--name", default="red_team")
    args = parser.parse_args()

    questions = [
        json.loads(line)
        for line in (ROOT / args.file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bot = TuringBot(seed=2026)
    print(f"corpus={len(bot.corpus)} dense={'ON' if bot.dense else 'OFF'} questions={len(questions)}\n")

    rows = []
    by_cat: dict[str, list[dict]] = defaultdict(list)
    latencies = []

    for item in questions:
        answer = bot.answer(item["q"])
        answered = answer.source not in {"dodge", "panic"}
        row = {
            "id": item["id"],
            "category": item["category"],
            "question": item["q"],
            "answer": answer.text,
            "source": answer.source,
            "confidence": f"{answer.confidence:.3f}",
            "latency_ms": f"{answer.latency_ms:.1f}",
            "chars": len(answer.text),
            "tells": tell_flags(answer.text),
            "score_1_5": "",
            "notes": "",
        }
        rows.append(row)
        by_cat[item["category"]].append({"answered": answered, "row": row, "latency": answer.latency_ms})
        latencies.append(answer.latency_ms)

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with (out / f"{args.name}_results.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "questions": len(rows),
        "answered": sum(1 for r in rows if r["source"] not in {"dodge", "panic"}),
        "answered_rate": round(
            sum(1 for r in rows if r["source"] not in {"dodge", "panic"}) / len(rows), 3
        ),
        "empty": sum(1 for r in rows if not r["answer"].strip()),
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 1),
        "avg_chars": round(sum(r["chars"] for r in rows) / len(rows), 1),
        "tell_hits": sum(1 for r in rows if r["tells"]),
        "categories": {},
    }
    for cat, items in by_cat.items():
        rate = sum(i["answered"] for i in items) / len(items)
        summary["categories"][cat] = {
            "answered_rate": round(rate, 3),
            "dodges": sum(1 for i in items if not i["answered"]),
        }

    (out / f"{args.name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nCSV: {out / (args.name + '_results.csv')}")
    print("Открой CSV, поставь score_1_5 (1 — плохо, 5 — как человек) и notes.")


if __name__ == "__main__":
    main()
