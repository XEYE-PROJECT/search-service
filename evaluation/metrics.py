"""Métricas de ranking. Funciones puras, sin E/S.

Cada query evaluada se representa como un dict con al menos:
  rank         posición 1-based del esperado en los resultados, o None si no aparece
  duration_ms  latencia reportada por el servicio
  semantic_won True si el esperado ganó por su score semántico (breakdown), None sin dato
  category     etiqueta opcional del dataset (sinonimo, intencion, marca, typo, atributo)
"""

import re
import unicodedata

RECALL_KS = [1, 3, 5, 10]

# Réplica de app/domain/normalization.py para que evaluation/ no dependa de la app
# (el script debe poder correr contra producción con solo httpx instalado).
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_WHITESPACE = re.compile(r"\s+")


def normalize(value: str) -> str:
    text = value.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = _NON_WORD.sub(" ", text)
    text = _NON_ASCII.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


def rank_of_expected(items: list[str], expected: str) -> int | None:
    target = normalize(expected)
    for position, item in enumerate(items, start=1):
        if normalize(item) == target:
            return position
    return None


def recall_at(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hits / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / rank for rank in ranks if rank is not None) / len(ranks)


def mean_found_rank(ranks: list[int | None]) -> float | None:
    found = [rank for rank in ranks if rank is not None]
    if not found:
        return None
    return sum(found) / len(found)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return ordered[index]


def summarize(queries: list[dict]) -> dict:
    ranks = [q["rank"] for q in queries]
    durations = [q["duration_ms"] for q in queries if q.get("duration_ms") is not None]
    semantic_flags = [q["semantic_won"] for q in queries if q.get("semantic_won") is not None]
    metrics = {
        "queries": len(queries),
        "top1_accuracy": recall_at(ranks, 1),
        **{f"recall@{k}": recall_at(ranks, k) for k in RECALL_KS},
        "mrr": mrr(ranks),
        "mean_found_rank": mean_found_rank(ranks),
        "not_found_count": sum(1 for rank in ranks if rank is None),
        "avg_duration_ms": sum(durations) / len(durations) if durations else None,
        "p95_duration_ms": percentile(durations, 95),
        "semantic_contribution": (
            sum(1 for flag in semantic_flags if flag) / len(semantic_flags)
            if semantic_flags
            else None
        ),
    }
    categories = sorted({q["category"] for q in queries if q.get("category")})
    if categories:
        metrics["by_category"] = {
            category: {
                "queries": len(subset),
                "top1_accuracy": recall_at(subset, 1),
                "recall@3": recall_at(subset, 3),
                "mrr": mrr(subset),
            }
            for category in categories
            for subset in [[q["rank"] for q in queries if q.get("category") == category]]
        }
    return metrics
