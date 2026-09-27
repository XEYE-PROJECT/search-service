"""Dependencias FastAPI: acceso al contenedor DI, auth por API key (+ rate limits) y auth interna.

La auth es una *dependencia* a propósito, no middleware: las dependencias nunca corren en
el preflight OPTIONS de CORS, lo que arregla el bug del servicio original (los preflight
recibían 401 antes de que CORS pudiera responder).

Orden en la API pública: límite por IP (antes de mirar la key: acota los 401 de fuerza bruta)
→ resolución de la key (401) → límite del USUARIO dueño de la key (todas sus keys y la consola
comparten el cupo; el valor es el del plan por defecto o el que fijó un admin). Las cabeceras
``X-RateLimit-*`` salen en cada respuesta autenticada y, con ``Retry-After``, en los 429.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request, Response

from app.application.errors import (
    InvalidApiKeyError,
    InvalidInternalTokenError,
    MissingApiKeyError,
    RateLimitedError,
)
from app.core.security import constant_time_equals
from app.domain.models import ApiKeyInfo


def get_container(request: Request):
    return request.app.state.container


Container = Annotated[object, Depends(get_container)]


def client_ip(request: Request) -> str:
    """IP del cliente. Detrás del proxy es la de ``X-Forwarded-For`` porque uvicorn corre con
    ``--proxy-headers`` y ``FORWARDED_ALLOW_IPS`` (ver Dockerfile); sin proxy, la del socket."""
    return request.client.host if request.client else "unknown"


def enforce_user_limit(request: Request, response: Response, user_id: int) -> None:
    """Cuenta una búsqueda del usuario contra su cupo (429 si se agota)."""
    container = request.app.state.container
    limit = container.user_limits.get(user_id, container.settings.rate_limit_per_minute)
    decision = container.rate_limiter.check(f"user:{user_id}", limit)
    if not decision.allowed:
        raise RateLimitedError(decision.limit, decision.reset_seconds, scope="account")
    response.headers.update(decision.headers())


async def require_api_key(
    request: Request,
    response: Response,
    x_api_key: Annotated[str | None, Header()] = None,
) -> ApiKeyInfo:
    container = request.app.state.container
    ip_decision = container.ip_rate_limiter.check(f"ip:{client_ip(request)}")
    if not ip_decision.allowed:
        raise RateLimitedError(ip_decision.limit, ip_decision.reset_seconds, scope="IP address")

    if not x_api_key or not x_api_key.strip():
        raise MissingApiKeyError()
    raw = x_api_key.strip()
    info = container.api_keys.resolve_raw(raw)
    if info is None and await container.catalog_service.refresh_on_miss():
        info = container.api_keys.resolve_raw(raw)
    if info is None:
        raise InvalidApiKeyError()
    enforce_user_limit(request, response, info.user_id)
    return info


async def require_internal_token(
    request: Request,
    x_internal_token: Annotated[str | None, Header()] = None,
) -> None:
    # Comparación en tiempo constante y fallo cerrado: un token configurado en blanco lo
    # rechaza todo en vez de permitirlo (un despliegue mal configurado no debe exponer la
    # API interna). En producción, además, Settings no arranca sin un token fuerte.
    expected = request.app.state.container.settings.internal_token.get_secret_value()
    if not constant_time_equals(expected, x_internal_token):
        raise InvalidInternalTokenError()
