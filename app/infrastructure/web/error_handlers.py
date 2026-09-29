"""Un solo formato de error para toda la API (el sobre de ``app.application.errors``) y el log
de auditoría de los rechazos de seguridad.

Cubre los errores propios (``ApiException``), los de validación de pydantic (422 con
``details`` por campo), los HTTP de Starlette/FastAPI (404 de ruta, 405…) y el último recurso
(500 sin detalles internos). Cada 401/403/429 se anota en el logger ``xeye.audit`` con IP, ruta
y el prefijo de la key (nunca la key entera ni el token interno).
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.application.errors import ApiException, RateLimitedError, error_body
from app.infrastructure import metrics
from app.infrastructure.web.deps import client_ip

logger = logging.getLogger(__name__)
audit = logging.getLogger("xeye.audit")

_AUDITED_STATUSES = {401, 403, 429}
_HTTP_CODES = {
    400: "BAD_REQUEST",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    413: "REQUEST_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
}
KEY_PREFIX_LENGTH = 12  # lo mismo que muestra la consola (key_prefix del backend)


def key_prefix(request: Request) -> str:
    raw = (request.headers.get("x-api-key") or "").strip()
    if not raw:
        return "-"
    return raw[:KEY_PREFIX_LENGTH] + ("…" if len(raw) > KEY_PREFIX_LENGTH else "")


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiException)
    async def handle_api_exception(request: Request, exc: ApiException) -> JSONResponse:
        if isinstance(exc, RateLimitedError):
            metrics.RATE_LIMITED.labels(exc.scope).inc()
        if exc.status_code in _AUDITED_STATUSES:
            audit.warning(
                "%s status=%d ip=%s method=%s path=%s key=%s",
                exc.code,
                exc.status_code,
                client_ip(request),
                request.method,
                request.url.path,
                key_prefix(request),
            )
        return JSONResponse(status_code=exc.status_code, content=exc.to_body(), headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        details: dict[str, str] = {}
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ()) if part != "body") or "body"
            details.setdefault(location, str(error.get("msg", "invalid value")))
        return JSONResponse(
            status_code=422,
            content=error_body(422, "VALIDATION_FAILED", "Request validation failed", details),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        status = exc.status_code
        code = _HTTP_CODES.get(status, "HTTP_ERROR")
        message = str(exc.detail) if exc.detail else code.replace("_", " ").capitalize()
        if status in _AUDITED_STATUSES:
            audit.warning(
                "%s status=%d ip=%s method=%s path=%s key=%s",
                code,
                status,
                client_ip(request),
                request.method,
                request.url.path,
                key_prefix(request),
            )
        return JSONResponse(
            status_code=status, content=error_body(status, code, message), headers=dict(exc.headers or {})
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Starlette relanza la excepción tras enviar esta respuesta: el stack sigue llegando a
        # los logs de uvicorn y a Sentry; al cliente solo le llega el sobre genérico.
        logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=500, content=error_body(500, "INTERNAL_ERROR", "Unexpected error"))
