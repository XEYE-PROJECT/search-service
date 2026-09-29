"""Tests de regresión de las carreras carga-vs-push y refresh-vs-push."""

import asyncio

from app.application.catalog import ApiKeyStore, CatalogService, ListCatalog
from app.core.security import hash_api_key
from app.infrastructure.embeddings import ModelRegistry
from app.infrastructure.list_cache import ListDataCache
from tests.conftest import FakeBackend
from tests.test_list_cache import make_data


async def test_invalidate_during_load_is_not_lost():
    cache = ListDataCache(max_bytes=10_000)
    loading = asyncio.Event()
    proceed = asyncio.Event()

    async def slow_loader():
        loading.set()
        await proceed.wait()  # fetch al backend en vuelo...
        return make_data(1)

    task = asyncio.create_task(cache.get_or_load(1, slow_loader))
    await loading.wait()
    cache.invalidate(1)  # el push llega en plena carga (p. ej. edición solo de params)
    proceed.set()
    result = await task
    assert result is not None  # la petición en vuelo se sirve igualmente
    assert cache.get(1) is None  # ...pero los datos pre-edición NO se cachearon


async def test_index_push_during_load_is_not_overwritten():
    cache = ListDataCache(max_bytes=10_000)
    loading = asyncio.Event()
    proceed = asyncio.Event()
    stale = make_data(1, memory=100)
    fresh = make_data(1, memory=200)

    async def slow_loader():
        loading.set()
        await proceed.wait()
        return stale

    task = asyncio.create_task(cache.get_or_load(1, slow_loader))
    await loading.wait()
    cache.put(1, fresh)  # el push de fin de entrenamiento llega en plena carga
    proceed.set()
    await task
    assert cache.get(1) is fresh  # el loader no debe machacar los datos empujados


async def test_catalog_refresh_discards_stale_snapshot():
    backend = FakeBackend()
    backend.api_keys = [(1, 10, hash_api_key("revoked-key"))]
    api_keys, lists = ApiKeyStore(), ListCatalog()
    service = CatalogService(backend, api_keys, lists, min_refresh_interval=0)

    fetching = asyncio.Event()
    proceed = asyncio.Event()
    original_fetch = backend.fetch_bootstrap

    async def slow_fetch():
        fetching.set()
        snapshot = await original_fetch()
        await proceed.wait()  # snapshot serializado, respuesta aún en vuelo...
        return snapshot

    backend.fetch_bootstrap = slow_fetch
    task = asyncio.create_task(service.refresh())
    await fetching.wait()
    api_keys.remove(1)  # el push DELETE llega con el snapshot en vuelo
    backend.api_keys = []  # el backend tampoco tiene ya la key
    proceed.set()
    await task
    # El snapshot caduco no debe resucitar la key revocada (en su lugar se refetchea).
    assert api_keys.resolve_raw("revoked-key") is None
    assert service.ready


async def test_refresh_on_miss_reports_concurrent_refresh():
    backend = FakeBackend()
    backend.api_keys = [(1, 10, hash_api_key("new-key"))]
    service = CatalogService(backend, ApiKeyStore(), ListCatalog(), min_refresh_interval=999)
    # Dos misses concurrentes: una sola llamada de bootstrap, pero a AMBOS llamantes se
    # les debe decir que hubo refresh para que re-resuelvan la key.
    results = await asyncio.gather(service.refresh_on_miss(), service.refresh_on_miss())
    assert backend.bootstrap_calls == 1
    assert results == [True, True]


async def test_model_registry_negative_caches_failures():
    registry = ModelRegistry("default-model", max_loaded=2)
    attempts = 0

    import anyio

    original = anyio.to_thread.run_sync

    async def failing_run_sync(fn, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("no such model")

    anyio.to_thread.run_sync = failing_run_sync
    try:
        assert await registry.embed_query("bogus/model", "hola") is None
        assert await registry.embed_query("bogus/model", "hola") is None  # fallo cacheado
    finally:
        anyio.to_thread.run_sync = original
    assert attempts == 1
