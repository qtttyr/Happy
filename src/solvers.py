"""Deterministic solvers: arithmetic, percentages, letter counting, dates.

These exist so the bot never fails a question a human would answer instantly,
and so it does not have to "hallucinate" numbers from a retrieval base.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from datetime import datetime

from .nlp import normalize


@dataclass
class Solution:
    text: str
    kind: str
    confidence: float = 1.0


_ALLOWED = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


def _safe_eval(expr: str) -> float | None:
    expr = expr.replace(",", ".").replace("^", "**").replace("×", "*").replace("÷", "/")
    expr = re.sub(r"[^0-9+\-*/().% ]", "", expr)
    if not expr or not any(ch.isdigit() for ch in expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            return None
        if isinstance(node, ast.Pow) and isinstance(node.right, ast.Constant):
            if abs(float(node.right.value)) > 12:
                return None
    try:
        value = eval(compile(tree, "<math>", "eval"), {"__builtins__": {}}, {})
    except Exception:
        return None
    if isinstance(value, complex) or value != value:
        return None
    return float(value)


def _fmt(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", ",")


def _percent(q: str) -> Solution | None:
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*%?\s*(?:от|из)\s*(\d+(?:[.,]\d+)?)", q)
    if not m or "%" not in q:
        return None
    pct = float(m.group(1).replace(",", "."))
    base = float(m.group(2).replace(",", "."))
    return Solution(f"{_fmt(base * pct / 100)}", "percent")


def _letters(q: str) -> Solution | None:
    m = re.search(r"сколько\s+(?:букв|звуков)\s+(?:в\s+слове\s+)?[«\"']?([а-яa-z\-]+)", q)
    if not m:
        return None
    word = m.group(1)
    letters = [ch for ch in word if ch.isalpha()]
    return Solution(f"{len(letters)}", "letters")


def _weekday(q: str) -> Solution | None:
    if not re.search(r"(какой|какое)\s+(сегодня\s+)?(день|число|дата|год)", q):
        return None
    now = datetime.now()
    months = (
        "января", "февраля", "марта", "апреля", "мая", "июня",
        "июля", "августа", "сентября", "октября", "ноября", "декабря",
    )
    weekdays = ("понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье")
    if "год" in q:
        return Solution(str(now.year), "date")
    if "день" in q:
        return Solution(f"{weekdays[now.weekday()]}, {now.day} {months[now.month - 1]}", "date")
    return Solution(f"{now.day} {months[now.month - 1]} {now.year}", "date")


def _sequence(q: str) -> Solution | None:
    nums = re.findall(r"-?\d+", q)
    if len(nums) < 4 or not re.search(r"продолжи|следующ|закономерн", q):
        return None
    vals = [int(n) for n in nums]
    diffs = [b - a for a, b in zip(vals, vals[1:])]
    if len(set(diffs)) == 1:
        return Solution(str(vals[-1] + diffs[-1]), "sequence")
    if all(a != 0 and b % a == 0 for a, b in zip(vals, vals[1:])):
        ratios = {b // a for a, b in zip(vals, vals[1:])}
        if len(ratios) == 1:
            return Solution(str(vals[-1] * ratios.pop()), "sequence")
    return None


def solve(question: str) -> Solution | None:
    q = normalize(question)

    sol = _weekday(q)
    if sol:
        return sol
    sol = _letters(q)
    if sol:
        return sol
    sol = _percent(q)
    if sol:
        return sol
    sol = _sequence(q)
    if sol:
        return sol

    if re.search(r"сколько\s+(будет|это|равно)", q) or re.fullmatch(r"[\d\s+\-*/().,%^]+", q):
        expr = re.sub(r"^(сколько\s+(будет|это|равно)\s*)", "", q)
        expr = re.sub(r"(посчитай|реши|ответь)\s*", "", expr)
        # "17 умножить на 23" and friends
        expr = expr.replace("умножить на", "*").replace("умножить", "*")
        expr = expr.replace("плюс", "+").replace("минус", "-").replace("делить на", "/")
        value = _safe_eval(expr)
        if value is not None:
            return Solution(_fmt(value), "math")

    if re.fullmatch(r"[\d\s+\-*/().,%^]+", q) and any(op in q for op in "+-*/"):
        value = _safe_eval(q)
        if value is not None:
            return Solution(_fmt(value), "math")

    return None
