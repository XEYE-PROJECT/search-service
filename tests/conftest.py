import asyncio
import base64
import io
import os

import numpy as np
import pytest

# `app.main` construye la app al importarse con Settings() reales: en producción (el default)
# exige un INTERNAL_TOKEN fuerte. Los tests son desarrollo, pase lo que pase en el .env local.
os.environ.setdefault("ENVIRONMENT", "development")

from app.application.catalog import ApiKeyStore, CatalogService, ListCatalog
from app.application.list_data import ListDataService
from app.application.ports import BootstrapData, ListDataPayload, LogEntry
from app.application.search_use_case import SearchUseCase
from app.core.config import Settings
from app.core.container import Container
from app.domain.models import ListMeta
from app.domain.scoring import ScoringConfig
from app.infrastructure.list_builder import build_list_data
from app.infrastructure.list_cache import ListDataCache
from app.infrastructure.log_queue import SearchLogQueue
from app.infrastructure.rate_limiter import RateLimiter
from app.main import create_app


def embeddings_b64(matrix: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, matrix.astype(np.float32))
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeBackend:
    """Sustituto en memoria de la API interna del backend Java."""

    def __init__(self) -> None:
        self.api_keys: list[tuple[int, int, str]] = []
        self.lists: list[ListMeta] = []
        self.list_payloads: dict[int, ListDataPayload] = {}
        self.pushed_logs: list[LogEntry] = []
        self.bootstrap_calls = 0
        self.list_fetches: list[int] = []
        self.fail_bootstrap = False
        self.fail_list_fetch = False
        self.fail_push_logs = 0  # falla los próximos N pushes

    async def fetch_bootstrap(self) -> BootstrapData:
        self.bootstrap_calls += 1
        await asyncio.sleep(0)  # cede el control, como una ida y vuelta HTTP real
        if self.fail_bootstrap:
            raise ConnectionError("backend down")
        return BootstrapData(api_keys=list(self.api_keys), lists=list(self.lists))

    async def fetch_list_data(self, list_id: int) -> ListDataPayload | None:
        self.list_fetches.append(list_id)
        if self.fail_list_fetch:
            raise ConnectionError("backend down")
        return self.list_payloads.get(list_id)

    async def push_logs(self, entries: list[LogEntry]) -> None:
        if self.fail_push_logs > 0:
            self.fail_push_logs -= 1
            raise ConnectionError("backend down")
        self.pushed_logs.extend(entries)


class FakeEmbedder:
    """Devuelve vectores unitarios preconfigurados por texto de query; None si no lo conoce."""

    def __init__(self, vectors: dict[str, np.ndarray] | None = None) -> None:
        self.vectors = vectors or {}
        self.calls: list[tuple[str | None, str]] = []

    async def embed_query(self, model_name: str | None, text: str) -> np.ndarray | None:
        self.calls.append((model_name, text))
        vector = self.vectors.get(text)
        if vector is None:
            return None
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector


def make_settings(**overrides) -> Settings:
    defaults = dict(
        environment="development",  # producción exige un token interno fuerte
        backend_url="http://backend.test",
        internal_token="test-token",
        refresh_min_interval_seconds=0.0,
        log_flush_seconds=0.05,
        rate_limit_per_minute=1000,
    )
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def make_container(settings: Settings, backend: FakeBackend, embedder: FakeEmbedder) -> Container:
    from functools import partial

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
        catalog_service, list_data, embedder, log_queue,
        ScoringConfig(
            text_weight=settings.search_text_weight,
            semantic_weight=settings.search_semantic_weight,
            override_threshold=settings.score_override_threshold,
        ),
    )
    return Container(
        settings=settings, backend=backend, embedder=embedder, api_keys=api_keys,
        lists=lists, catalog_service=catalog_service, cache=cache, list_data=list_data,
        log_queue=log_queue, rate_limiter=rate_limiter, search=search,
    )


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def container(settings, backend, embedder) -> Container:
    return make_container(settings, backend, embedder)


@pytest.fixture
def app(settings, container):
    return create_app(settings=settings, container=container)


@pytest.fixture
async def client(app):
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://search.test") as http_client:
        yield http_client


@pytest.fixture
def anyio_backend():
    return "asyncio"


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)
