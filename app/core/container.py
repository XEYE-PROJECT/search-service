"""Cableado de dependencias: un único sitio construye todo el grafo de objetos; los tests
montan su propio contenedor con fakes del gateway al backend y del embedder."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

from app.application.catalog import ApiKeyStore, CatalogService, ListCatalog
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
from app.infrastructure.rate_limiter import RateLimiter


@dataclass
class Container:
    settings: Settings
    backend: BackendGateway
    embedder: QueryEmbedder
    api_keys: ApiKeyStore
    lists: ListCatalog
    catalog_service: CatalogService
    cache: ListDataCache
    list_data: ListDataService
    log_queue: SearchLogQueue
    rate_limiter: RateLimiter
    search: SearchUseCase
    background_tasks: list = field(default_factory=list)

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
        settings.backend_url, settings.internal_token, timeout=settings.backend_timeout_seconds
    )
    embedder = embedder or ModelRegistry(
        settings.embedding_model_default, settings.models_max_loaded
    )

    api_keys = ApiKeyStore()
    lists = ListCatalog()
    catalog_service = CatalogService(
        backend, api_keys, lists, min_refresh_interval=settings.refresh_min_interval_seconds
    )
    cache = ListDataCache(settings.cache_max_bytes)
    list_data = ListDataService(cache, backend, lists, partial(build_list_data, settings=settings))
    log_queue = SearchLogQueue(
        backend,
        queue_max=settings.log_queue_max,
        batch_max=settings.log_batch_max,
        flush_seconds=settings.log_flush_seconds,
        retries=settings.log_push_retries,
    )
    rate_limiter = RateLimiter(settings.rate_limit_per_minute)
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
        catalog_service=catalog_service,
        cache=cache,
        list_data=list_data,
        log_queue=log_queue,
        rate_limiter=rate_limiter,
        search=search,
    )
