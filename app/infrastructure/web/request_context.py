"""Middleware ASGI puro: request id, access log y métricas HTTP por petición.

- Toma ``X-Request-Id`` del proxy (si es sano) o genera uno, lo deja en el contexto (los logs
  lo incluyen) y lo devuelve en la respuesta.
- Escribe una línea por petición en el logger ``xeye.access`` (método, ruta, estado, duración,
  IP, prefijo de la key, request id). Las sondas y ``/metrics`` no se loguean (ruido).
- Cuenta cada petición y su latencia en Prometheus.
"""

from __future__ import annotations

import logging
import time

from app.core.request_context import (
    REQUEST_ID_HEADER,
    new_request_id,
    request_id_var,
    sanitize_request_id,
)
from app.infrastructure import metrics

access_log = logging.getLogger("xeye.access")

_QUIET_PATHS = frozenset({"/health", "/ready", "/metrics"})
_HEADER_BYTES = REQUEST_ID_HEADER.lower().encode("latin-1")
KEY_PREFIX_LENGTH = 12


def _header(scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key == name:
            return value.decode("latin-1")
    return None


def _client_ip(scope) -> str:
    client = scope.get("client")
    return client[0] if client else "unknown"


class RequestContextMiddleware:
    def __init__(self, app) -> None:
        self._app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_id = sanitize_request_id(_header(scope, _HEADER_BYTES)) or new_request_id()
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_holder = {"status": 0}

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                headers = list(message.get("headers", []))
                headers.append((_HEADER_BYTES, request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self._app(scope, receive, send_with_id)
        finally:
            elapsed = time.perf_counter() - started
            path = scope.get("path", "")
            method = scope.get("method", "-")
            status = status_holder["status"] or 500
            metrics.observe_request(method, path, status, elapsed)
            if path not in _QUIET_PATHS:
                raw_key = (_header(scope, b"x-api-key") or "").strip()
                key = "-"
                if raw_key:
                    key = raw_key[:KEY_PREFIX_LENGTH] + ("…" if len(raw_key) > KEY_PREFIX_LENGTH else "")
                ip = _client_ip(scope)
                access_log.info(
                    "%s %s %d %.1fms ip=%s key=%s", method, path, status, elapsed * 1000, ip, key,
                    extra={
                        "method": method, "path": path, "status": status,
                        "duration_ms": round(elapsed * 1000, 1), "ip": ip, "key_prefix": key,
                    },
                )
            request_id_var.reset(token)
