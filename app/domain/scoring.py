"""Combinación híbrida texto+semántica. Funciones puras — sin E/S y sin numpy.

Reglas heredadas del servicio original: match normalizado exacto -> 1.0 (lo aplica el
llamante); cualquier puntuación >= umbral gana ella sola; si no, suma ponderada; ambas a
0 -> el elemento se descarta. Fix respecto al original: sin semántica vale el texto tal
cual (antes se aplicaba w_text igualmente, capando todo match textual a 0.25).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringConfig:
    text_weight: float = 0.25
    semantic_weight: float = 0.75
    override_threshold: float = 0.75


def combine(text_score: float, semantic_score: float | None, config: ScoringConfig) -> float | None:
    """Puntuación final de un elemento, o None si debe descartarse."""
    if semantic_score is None:
        return text_score if text_score > 0.0 else None
    if text_score <= 0.0 and semantic_score <= 0.0:
        return None
    if text_score >= config.override_threshold:
        return text_score
    if semantic_score >= config.override_threshold:
        return semantic_score
    return config.text_weight * text_score + config.semantic_weight * semantic_score


def text_score_from_fuzz(ratio: float, partial_ratio: float) -> float:
    """Mezcla del ratio completo y el parcial de rapidfuzz (entradas 0-100, salida 0-1)."""
    return (0.4 * ratio + 0.6 * partial_ratio) / 100.0
