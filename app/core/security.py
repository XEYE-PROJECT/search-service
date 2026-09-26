"""Primitivas de seguridad compartidas: hash de API keys y comparación en tiempo constante."""

from __future__ import annotations

import hashlib
import hmac


def hash_api_key(raw_key: str) -> str:
    """SHA-256 hex en minúsculas del valor en claro.

    Es exactamente lo que guarda el backend (``ApiKeyHasher`` en Java, ``SHA2(api_key, 256)``
    en su migración V6): el bootstrap y los pushes traen este hash, y aquí se calcula el mismo
    sobre la cabecera ``X-API-Key`` recibida. Cambiarlo rompe la autenticación de todas las keys.
    """
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def constant_time_equals(expected: str | None, provided: str | None) -> bool:
    """``True`` solo si ambos existen y coinciden.

    Falla cerrado: un secreto esperado en blanco devuelve siempre ``False``. Se comparan los
    SHA-256 de ambos valores porque ``compare_digest`` solo es de tiempo constante con
    longitudes iguales, y así tampoco se filtra la longitud del secreto real.
    """
    if not expected or not expected.strip() or provided is None:
        return False
    return hmac.compare_digest(
        hashlib.sha256(expected.encode("utf-8")).digest(),
        hashlib.sha256(provided.encode("utf-8")).digest(),
    )
