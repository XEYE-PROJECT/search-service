"""Identificador de la petición en curso, visible desde cualquier log del mismo contexto.

Lo fija el middleware ``RequestContextMiddleware`` (lo toma de ``X-Request-Id`` si el proxy lo
manda, o genera uno) y lo lee el formateador JSON de logs: así cada línea de una búsqueda —
auditoría, degradación, errores— lleva el mismo ``request_id`` que devuelve la respuesta.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

REQUEST_ID_HEADER = "X-Request-Id"
_MAX_LENGTH = 128

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


def current_request_id() -> str | None:
    return request_id_var.get()


def new_request_id() -> str:
    return uuid.uuid4().hex


def sanitize_request_id(value: str | None) -> str | None:
    """Acepta el id del proxy si es corto y solo tiene caracteres imprimibles seguros."""
    if not value:
        return None
    value = value.strip()
    if not value or len(value) > _MAX_LENGTH:
        return None
    if not all(ch.isalnum() or ch in "-_.:" for ch in value):
        return None
    return value
