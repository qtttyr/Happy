"""Russian text normalisation, tokenisation and lemmatisation.

pymorphy3 is optional: without it the bot still runs, using a light stemmer.
Everything here is local, deterministic and fast (< 2 ms per short question).
"""
from __future__ import annotations

import functools
import re

try:  # optional dependency
    import pymorphy3
except Exception:  # pragma: no cover
    pymorphy3 = None

_MORPH = pymorphy3.MorphAnalyzer() if pymorphy3 else None

TOKEN_RE = re.compile(r"[а-яa-z0-9]+")
_SPACES = re.compile(r"\s+")
_YO = str.maketrans({"ё": "е", "Ё": "Е"})

# Deliberately keeps negations and question words: they carry meaning in short
# questions ("почему не", "как"), while pure function words are dropped.
STOPWORDS = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
    только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли
    если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя
    ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без
    будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой
    совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех
    никогда можно при наконец два об другой хоть после над больше тот через эти нас
    про всего них какая много разве три эту моя впрочем хорошо свою этой перед
    иногда лучше чуть том нельзя такой им более всегда конечно всю между это
    который которые которая которых кроме очень просто вообще тогда самый самая
    сегодня вчера завтра сейчас потом нибудь рассказать объяснить посоветовать
    подсказать сказать говорить знать думать хотеть мочь делать сделать дать найти
    показать помочь нравиться любить понимать понимание считать казаться
    становиться стать уметь почему отчего
    """.split()
)


def normalize(text: str) -> str:
    """Lowercase, unify ё, strip punctuation that BM25 does not need."""
    if not text:
        return ""
    text = text.translate(_YO).lower()
    text = text.replace("—", " ").replace("–", " ").replace("…", " ")
    text = re.sub(r"[^а-яa-z0-9\s%+\-*/=.,?()]", " ", text)
    return _SPACES.sub(" ", text).strip()


@functools.lru_cache(maxsize=300_000)
def lemma(word: str) -> str:
    """Dictionary lemma. Cache is essential: the index calls this millions of times."""
    if _MORPH is None:
        # Crude but stable fallback for environments without pymorphy3.
        return word[:6] if len(word) > 6 else word
    try:
        return _MORPH.parse(word)[0].normal_form
    except Exception:
        return word


def tokens(text: str, lemmatize: bool = True, drop_stop: bool = True) -> list[str]:
    out: list[str] = []
    for word in TOKEN_RE.findall(normalize(text)):
        if drop_stop and word in STOPWORDS:
            continue
        word = lemma(word) if lemmatize else word
        if drop_stop and word in STOPWORDS:
            continue
        out.append(word)
    return out


def content_words(text: str) -> list[str]:
    """Content lemmas sorted by length (longest first) - used by the composer."""
    words = [w for w in tokens(text) if len(w) > 2]
    return sorted(set(words), key=len, reverse=True)


def overlap(a: str, b: str) -> float:
    """Jaccard over content lemmas. Used by the novelty guard."""
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_math(text: str) -> bool:
    raw = normalize(text)
    if not raw:
        return False
    digits = sum(ch.isdigit() for ch in raw)
    ops = sum(ch in "+-*/=^%" for ch in raw)
    return digits >= 2 and ops >= 1
