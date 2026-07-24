"""Normalización de texto para el matching (las mismas reglas que asume el entrenamiento).

Minúsculas, sin tildes (NFD), sin puntuación ni restos no ASCII, espacios colapsados. Se
aplica al texto indexado y a la query entrante para que el matching exacto y el fuzzy
ignoren tildes, mayúsculas y signos.
"""

import re
import unicodedata

_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    text = value.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = _NON_WORD.sub(" ", text)
    text = _NON_ASCII.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()
