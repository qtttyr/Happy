"""Build runtime JSONL from the human-editable knowledge base sources.

Source format (one row per line, pipe-separated):
    id | question | answer | intent | tags | variants

- tags: comma-separated
- variants: semicolon-separated alternative phrasings
- topic comes from the file name

Run:
    python scripts/build_kb.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "src"
OUT = ROOT / "data"

TOPICS = {
    "science": "science",
    "astana": "astana",
    "social": "social",
    "traps": "trap",
    "everyday": "everyday",
    "opinions": "opinion",
    "smalltalk": "smalltalk",
    "math": "logic",
    "fallbacks": "fallback",
    "common": "smalltalk",
}

BANNED_OPENERS = ("конечно", "отличный вопрос", "хороший вопрос", "важно отметить", "таким образом")
SPACES = re.compile(r"\s+")


def parse(path: Path, topic: str) -> list[dict]:
    rows: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        while len(parts) < 6:
            parts.append("")
        ident, question, answer, intent = parts[:4]
        tags_raw, variants_raw = parts[4], parts[5]
        answers_raw = parts[6] if len(parts) > 6 else ""
        if not ident or not question or not answer:
            print(f"  ! {path.name}:{lineno}: missing field, skipped")
            continue
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        variants = [v.strip() for v in variants_raw.split(";") if v.strip()]
        answer = SPACES.sub(" ", answer).strip()
        question = SPACES.sub(" ", question).strip()
        if len(answer) > 240:
            print(f"  ! {path.name}:{lineno}: answer {len(answer)} chars (trimmed)")
            answer = answer[:237].rstrip() + "…"
        if answer.lower().startswith(BANNED_OPENERS):
            print(f"  ! {path.name}:{lineno}: bookish opener")
        row = {
            "id": ident,
            "q": question,
            "a": answer,
            "topic": topic,
            "intent": intent or "smalltalk",
            "tags": tags,
            "variants": variants,
        }
        for index, extra in enumerate(
            [a.strip() for a in answers_raw.split(";") if a.strip()], start=2
        ):
            row[f"a{index}"] = extra
        rows.append(row)
    return rows


def main() -> None:
    total = 0
    variants_total = 0
    stats = {}
    seen_ids: set[str] = set()
    seen_q: set[str] = set()

    for name, topic in TOPICS.items():
        path = SRC / f"{name}.txt"
        if not path.exists():
            continue
        rows = parse(path, topic)
        clean: list[dict] = []
        for row in rows:
            key = row["q"].lower()
            if row["id"] in seen_ids:
                print(f"  ! duplicate id: {row['id']}")
                continue
            if key in seen_q:
                print(f"  ! duplicate question: {row['q'][:60]}")
                continue
            seen_ids.add(row["id"])
            seen_q.add(key)
            clean.append(row)
        out = OUT / f"kb_{name}_qa.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for row in clean:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        total += len(clean)
        variants_total += sum(len(r["variants"]) for r in clean)
        stats[name] = {"rows": len(clean), "variants": sum(len(r["variants"]) for r in clean)}
        print(f"{name:10s} {len(clean):4d} rows -> {out.name}")

    (OUT / "kb_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\ntotal: {total} rows, {variants_total} variants, {total + variants_total} indexed formulations")


if __name__ == "__main__":
    main()
