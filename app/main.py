"""Microservicio de búsqueda de XEYE.

Dos superficies: /api/v1/* es la API pública (X-API-Key) y /v1/* la interna
(X-Internal-Token, solo el backend). El arranque hace bootstrap de los catálogos con
reintentos, lanza el re-sync periódico, el consumidor de logs (con reenvío del spool en
disco) y precalienta los modelos de embedding que anuncia el backend para que ninguna
primera búsqueda pague la carga. Los datos de búsqueda de cada lista son perezosos: se
cargan en la primera búsqueda y quedan en la caché LRU en RAM, con TTL y revalidación en
segundo plano.

Sondas: /health es liveness (el proceso responde; siempre 200, con el estado real dentro) y
/ready es readiness (503 hasta que el bootstrap contra el backend ha terminado): la que debe
vigilar un monitor externo. /metrics expone Prometheus (solo por la red docker).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import Settings, get_settings
from app.core.container import Container, build_container
from app.core.logging import configure_logging
from app.infrastructure import metrics
from app.infrastructure.web.error_handlers import register_error_handlers
from app.infrastructure.web.guards import RequestGuardMiddleware
from app.infrastructure.web.internal_router import router as internal_router
from app.infrastructure.web.public_router import router as public_router
from app.infrastructure.web.request_context import RequestContextMiddleware

if TYPE_CHECKING:
    from sentry_sdk.types import Event, Hint

logger = logging.getLogger(__name__)

#: Cabeceras que nunca deben salir hacia Sentry aunque el SDK capture la petición.
_SENSITIVE_HEADERS = frozenset({"x-api-key", "x-internal-token", "authorization", "cookie"})


def _scrub_sensitive_headers(event: Event, _hint: Hint) -> Event | None:
    request = event.get("request")
    if isinstance(request, dict):
        headers = request.get("headers")
        if isinstance(headers, dict):
            request["headers"] = {
                name: value for name, value in headers.items() if name.lower() not in _SENSITIVE_HEADERS
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
    logger.info("Sentry enabled (environment=%s, release=%s)", settings.environment, settings.sentry_release or "-")


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
            container.list_data.reconcile()  # listas borradas cuya notificación se perdió
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
        asyncio.create_task(container.log_queue.replay(), name="log-spool-replay"),
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
        # Acotado: lo que no se entregue a tiempo va al spool en disco (no se pierde).
        await asyncio.wait_for(container.log_queue.flush(), timeout=10.0)
    except (TimeoutError, Exception):
        logger.warning(
            "Could not flush all pending search logs on shutdown (%d still pending)", container.log_queue.pending
        )
    await container.aclose()


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format_resolved)
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
    # Los middlewares se ejecutan en orden inverso al de registro: request id/métricas
    # envuelven a la guarda de Host/tamaño, que envuelve a CORS — un Host falso se rechaza
    # antes de nada, pero aun así queda contado y con request id.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "Retry-After",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-Request-Id",
            "X-Search-Degraded",
        ],
    )
    app.add_middleware(
        RequestGuardMiddleware,
        allowed_hosts=settings.allowed_host_list,
        max_public_body_bytes=settings.max_request_bytes,
    )
    app.add_middleware(RequestContextMiddleware)
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
    async def health(request: Request):
        """Liveness: el proceso atiende (siempre 200; lo usa el healthcheck de docker). El
        cuerpo lleva el estado real: ``ready``, ``degraded`` y cada ``check``."""
        current = getattr(request.app.state, "container", None)
        if current is None:
            return {"status": "ok", "ready": False, "degraded": False, "checks": {"catalog": "starting"}}
        status = current.health()
        return {"status": "ok", **status}

    @app.get("/ready", tags=["meta"])
    async def ready(request: Request):
        """Readiness: 200 solo con los catálogos cargados desde el backend (si no, 503).
        Es la sonda para el monitor de uptime externo. ``degraded`` avisa de que se sirve,
        pero peor (sin modelo de embedding, logs esperando en disco)."""
        current = getattr(request.app.state, "container", None)
        status = current.health() if current is not None else {"ready": False, "degraded": False, "checks": {}}
        return JSONResponse(
            status_code=200 if status["ready"] else 503,
            content={"status": "ready" if status["ready"] else "starting", **status},
        )

    if settings.metrics_enabled:

        @app.get("/metrics", tags=["meta"], include_in_schema=False)
        async def prometheus_metrics(request: Request):
            current = getattr(request.app.state, "container", None)
            if current is not None:
                metrics.refresh_gauges(current)
            body, content_type = metrics.render()
            return Response(content=body, media_type=content_type)

    app.include_router(public_router)
    app.include_router(internal_router)
    return app


app = create_app()
