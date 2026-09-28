"""Normalización de texto para el matching (las mismas reglas en el texto indexado y en la query).

Compatibilidad Unicode (NFKC: anchos completos, ligaduras), minúsculas, sin marcas diacríticas
(NFD → se descartan las ``Mn``: tildes, diéresis…), sin puntuación ni símbolos, espacios
colapsados. Las letras de cualquier alfabeto se CONSERVAN: cirílico, griego, árabe, CJK o
hebreo se comparan entre sí igual que el latino (antes se eliminaba todo lo no ASCII y una
lista en ruso o japonés quedaba en blanco para el matching textual).
"""

import re
import unicodedata

_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_UNDERSCORE = re.compile(r"_+")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = _NON_WORD.sub(" ", text)
    text = _UNDERSCORE.sub(" ", text)  # \w incluye '_' pero no es una letra
    return _WHITESPACE.sub(" ", text).strip()
