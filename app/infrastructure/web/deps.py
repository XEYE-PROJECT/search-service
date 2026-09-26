"""Dependencias FastAPI: acceso al contenedor DI, auth por API key (+ rate limit) y auth interna.

La auth es una *dependencia* a propósito, no middleware: las dependencias nunca corren en
el preflight OPTIONS de CORS, lo que arregla el bug del servicio original (los preflight
recibían 401 antes de que CORS pudiera responder).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request

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


async def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header()] = None,
) -> ApiKeyInfo:
    container = request.app.state.container
    if not x_api_key or not x_api_key.strip():
        raise MissingApiKeyError()
    raw = x_api_key.strip()
    info = container.api_keys.resolve_raw(raw)
    if info is None and await container.catalog_service.refresh_on_miss():
        info = container.api_keys.resolve_raw(raw)
    if info is None:
        raise InvalidApiKeyError()
    if not container.rate_limiter.allow(str(info.id)):
        raise RateLimitedError(container.settings.rate_limit_per_minute)
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
