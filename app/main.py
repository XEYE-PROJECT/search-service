"""Microservicio de búsqueda de XEYE.

Dos superficies: /api/v1/* es la API pública (X-API-Key) y /v1/* la interna
(X-Internal-Token, solo el backend). El arranque hace bootstrap de los catálogos con
reintentos, lanza el re-sync periódico y el consumidor de logs, y precalienta los modelos
de embedding que anuncia el backend para que ninguna primera búsqueda pague la carga.
Los datos de búsqueda de cada lista son perezosos: se cargan en la primera búsqueda y
quedan en la caché LRU en RAM.

Sondas: /health es liveness (el proceso responde) y /ready es readiness (503 hasta que el
bootstrap contra el backend ha terminado): la que debe vigilar un monitor externo.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import Settings, get_settings
from app.core.container import Container, build_container
from app.core.logging import configure_logging
from app.infrastructure.web.error_handlers import register_error_handlers
from app.infrastructure.web.guards import RequestGuardMiddleware
from app.infrastructure.web.internal_router import router as internal_router
from app.infrastructure.web.public_router import router as public_router

logger = logging.getLogger(__name__)

#: Cabeceras que nunca deben salir hacia Sentry aunque el SDK capture la petición.
_SENSITIVE_HEADERS = frozenset({"x-api-key", "x-internal-token", "authorization", "cookie"})


def _scrub_sensitive_headers(event: dict[str, Any], _hint: dict[str, Any]) -> dict[str, Any]:
    request = event.get("request")
    if isinstance(request, dict) and isinstance(request.get("headers"), dict):
        request["headers"] = {
            name: value
            for name, value in request["headers"].items()
            if name.lower() not in _SENSITIVE_HEADERS
        }
    return event


def _init_sentry(settings: Settings) -> None:
    """Error tracking opcional: solo con SENTRY_DSN. Import perezoso para no exigir el SDK en dev."""
    if not settings.sentry_dsn:
        return
    import sentry_sdk

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.environment,
        release=settings.sentry_release or None,
        traces_sample_rate=0.0,
        send_default_pii=False,
        before_send=_scrub_sensitive_headers,
    )
    logger.info("Sentry enabled (environment=%s, release=%s)",
                settings.environment, settings.sentry_release or "-")


async def _preload_available_models(container: Container) -> None:
    """Precalienta todos los modelos de embedding que anuncia el backend (nunca lanza)."""
    preload = getattr(container.embedder, "preload_models", None)
    models = container.catalog_service.embedding_models
    if preload is not None and models:
        await preload(models)


async def _bootstrap_with_retry(container: Container) -> None:
    delay = 2.0
    while True:
        try:
            await container.catalog_service.refresh()
            if container.catalog_service.ready:  # un refresh puede saltarse si hay carreras
                await _preload_available_models(container)
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Bootstrap against the backend failed (%s); retrying in %.0fs", exc, delay)
        await asyncio.sleep(delay)
        delay = min(delay * 2, 60.0)


async def _periodic_refresh(container: Container) -> None:
    interval = container.settings.refresh_interval_seconds
    while True:
        await asyncio.sleep(interval)
        try:
            await container.catalog_service.refresh()
            await _preload_available_models(container)  # recoge modelos recién configurados
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Periodic catalog refresh failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    container: Container = app.state.container_override or build_container(settings)
    app.state.container = container

    tasks = [
        asyncio.create_task(_bootstrap_with_retry(container), name="bootstrap"),
        asyncio.create_task(container.log_queue.run(), name="log-consumer"),
    ]
    if settings.refresh_interval_seconds > 0:
        tasks.append(asyncio.create_task(_periodic_refresh(container), name="periodic-refresh"))
    preload = getattr(container.embedder, "preload_default", None)
    if preload is not None:
        tasks.append(asyncio.create_task(preload(), name="model-preload"))
    container.background_tasks = tasks

    yield

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    try:
        # Acotado: con el backend caído, los reintentos por lote pararían el apagado
        # mientras quedaran logs encolados.
        await asyncio.wait_for(container.log_queue.flush(), timeout=10.0)
    except (Exception, asyncio.TimeoutError):
        logger.warning("Could not flush all pending search logs on shutdown "
                       "(%d dropped)", container.log_queue.pending)
    await container.aclose()


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    _init_sentry(settings)

    # Swagger/OpenAPI solo en desarrollo: en producción el esquema (que incluye la API
    # interna) no existe, ni siquiera detrás del proxy.
    app = FastAPI(
        title=settings.service_name,
        version=settings.service_version,
        lifespan=lifespan,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
    )
    app.state.settings = settings
    app.state.container_override = container
    if container is not None:  # los tests corren sin lifespan
        app.state.container = container

    # La auth vive en dependencias (nunca corren en OPTIONS), así el preflight CORS funciona.
    # Los middlewares se ejecutan en orden inverso al de registro: la guarda de Host/tamaño
    # envuelve a CORS, así un Host falso se rechaza antes de nada.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Retry-After", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
    )
    app.add_middleware(
        RequestGuardMiddleware,
        allowed_hosts=settings.allowed_host_list,
        max_public_body_bytes=settings.max_request_bytes,
    )
    register_error_handlers(app)

    @app.get("/", tags=["meta"])
    async def root():
        return {
            "service": settings.service_name,
            "version": settings.service_version,
            "status": "running",
            "description": "XEYE semantic search microservice",
        }

    @app.get("/health", tags=["meta"])
    async def health():
        """Liveness: el proceso atiende. Lo usa el healthcheck de docker."""
        return {"status": "ok"}

    @app.get("/ready", tags=["meta"])
    async def ready(request: Request):
        """Readiness: 200 solo con los catálogos cargados desde el backend (si no, 503).
        Es la sonda para el monitor de uptime externo."""
        current = getattr(request.app.state, "container", None)
        is_ready = current is not None and current.catalog_service.ready
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content={"status": "ready" if is_ready else "starting"},
        )

    app.include_router(public_router)
    app.include_router(internal_router)
    return app


app = create_app()
