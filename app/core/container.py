"""Cableado de dependencias: un único sitio construye todo el grafo de objetos; los tests
montan su propio contenedor con fakes del gateway al backend y del embedder."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

from app.application.catalog import ApiKeyStore, CatalogService, ListCatalog, UserLimits
from app.application.list_data import ListDataService
from app.application.ports import BackendGateway, QueryEmbedder
from app.application.search_use_case import SearchUseCase
from app.core.config import Settings
from app.domain.scoring import ScoringConfig
from app.infrastructure.backend_client import BackendClient
from app.infrastructure.embeddings import ModelRegistry
from app.infrastructure.list_builder import build_list_data
from app.infrastructure.list_cache import ListDataCache
from app.infrastructure.log_queue import SearchLogQueue
from app.infrastructure.log_spool import LogSpool
from app.infrastructure.rate_limiter import RateLimiter


@dataclass
class Container:
    settings: Settings
    backend: BackendGateway
    embedder: QueryEmbedder
    api_keys: ApiKeyStore
    lists: ListCatalog
    user_limits: UserLimits
    catalog_service: CatalogService
    cache: ListDataCache
    list_data: ListDataService
    log_queue: SearchLogQueue
    rate_limiter: RateLimiter  # por usuario (todas sus keys + consola)
    ip_rate_limiter: RateLimiter  # por IP, antes de resolver la key
    search: SearchUseCase
    background_tasks: list = field(default_factory=list)

    def health(self) -> dict:
        """Estado real del servicio para /health, /ready y las métricas.

        ``ready`` = catálogos cargados del backend (sin ellos nada se puede servir).
        ``degraded`` = se sirve, pero peor: el modelo de embedding por defecto no está
        cargado (búsquedas solo por texto) o hay logs esperando en el spool de disco.
        """
        catalog_ok = self.catalog_service.ready
        model_status = "unknown"
        status_fn = getattr(self.embedder, "default_model_status", None)
        if status_fn is not None:
            model_status = status_fn()
        spooled = self.log_queue.spooled
        checks = {
            "catalog": "ok" if catalog_ok else "loading",
            "embedding_model": model_status,
            "search_logs": "ok" if spooled == 0 else f"{spooled} spooled",
        }
        degraded = model_status == "failed" or spooled > 0
        return {"ready": catalog_ok, "degraded": degraded, "checks": checks}

    async def aclose(self) -> None:
        closer = getattr(self.backend, "aclose", None)
        if closer is not None:
            await closer()


def build_container(
    settings: Settings,
    *,
    backend: BackendGateway | None = None,
    embedder: QueryEmbedder | None = None,
) -> Container:
    backend = backend or BackendClient(
        settings.backend_url,
        settings.internal_token.get_secret_value(),
        timeout=settings.backend_timeout_seconds,
        connect_timeout=settings.backend_connect_timeout_seconds,
        retries=settings.backend_retries,
    )
    embedder = embedder or ModelRegistry(settings.embedding_model_default, settings.models_max_loaded)

    api_keys = ApiKeyStore()
    lists = ListCatalog()
    user_limits = UserLimits()
    catalog_service = CatalogService(
        backend,
        api_keys,
        lists,
        user_limits=user_limits,
        min_refresh_interval=settings.refresh_min_interval_seconds,
    )
    cache = ListDataCache(settings.cache_max_bytes, ttl_seconds=settings.cache_ttl_seconds)
    list_data = ListDataService(cache, backend, lists, partial(build_list_data, settings=settings))
    spool = LogSpool(settings.log_spool_dir, settings.log_spool_max_bytes) if settings.log_spool_dir else None
    log_queue = SearchLogQueue(
        backend,
        queue_max=settings.log_queue_max,
        batch_max=settings.log_batch_max,
        flush_seconds=settings.log_flush_seconds,
        retries=settings.log_push_retries,
        spool=spool,
        replay_seconds=settings.log_spool_replay_seconds,
    )
    rate_limiter = RateLimiter(settings.rate_limit_per_minute)
    ip_rate_limiter = RateLimiter(settings.rate_limit_per_ip_per_minute)
    search = SearchUseCase(
        catalog_service,
        list_data,
        embedder,
        log_queue,
        ScoringConfig(
            text_weight=settings.search_text_weight,
            semantic_weight=settings.search_semantic_weight,
            override_threshold=settings.score_override_threshold,
        ),
    )
    return Container(
        settings=settings,
        backend=backend,
        embedder=embedder,
        api_keys=api_keys,
        lists=lists,
        user_limits=user_limits,
        catalog_service=catalog_service,
        cache=cache,
        list_data=list_data,
        log_queue=log_queue,
        rate_limiter=rate_limiter,
        ip_rate_limiter=ip_rate_limiter,
        search=search,
    )
