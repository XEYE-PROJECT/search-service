"""Microservicio de búsqueda de XEYE.

Dos superficies: /api/v1/* es la API pública (X-API-Key) y /v1/* la interna
(X-Internal-Token, solo el backend). El arranque hace bootstrap de los catálogos con
reintentos, lanza el re-sync periódico y el consumidor de logs, y precalienta los modelos
de embedding que anuncia el backend para que ninguna primera búsqueda pague la carga.
Los datos de búsqueda de cada lista son perezosos: se cargan en la primera búsqueda y
quedan en la caché LRU en RAM.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.application.errors import ApiException
from app.core.config import Settings, get_settings
from app.core.container import Container, build_container
from app.core.logging import configure_logging
from app.infrastructure.web.internal_router import router as internal_router
from app.infrastructure.web.public_router import router as public_router

logger = logging.getLogger(__name__)


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

    if not settings.internal_token.strip() or settings.internal_token == "dev-internal-token":
        logger.warning(
            "INTERNAL_TOKEN is %s — the /v1/* internal API is effectively unprotected. "
            "Set a strong shared secret (and the same SEARCH_INTERNAL_TOKEN on the backend) "
            "before exposing this service.",
            "blank" if not settings.internal_token.strip() else "the dev default",
        )

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

    app = FastAPI(
        title=settings.service_name,
        version=settings.service_version,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.container_override = container
    if container is not None:  # los tests corren sin lifespan
        app.state.container = container

    # La auth vive en dependencias (nunca corren en OPTIONS), así el preflight CORS funciona.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ApiException)
    async def handle_api_exception(_request: Request, exc: ApiException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.error, "detail": exc.detail},
        )

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
        return {"status": "ok"}

    app.include_router(public_router)
    app.include_router(internal_router)
    return app


app = create_app()
