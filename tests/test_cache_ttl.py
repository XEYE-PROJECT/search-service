"""TTL de la caché de listas: stale-while-revalidate, revalidación fallida = degradado,
reconciliación tras el re-sync del catálogo."""

import asyncio

from app.application.catalog import ListCatalog
from app.application.list_data import ListDataService
from app.application.ports import ListDataPayload
from app.domain.models import ListMeta
from app.infrastructure.list_cache import ListDataCache
from tests.conftest import FakeBackend
from tests.test_list_cache import make_data


def payload(list_id: int, name: str = "L") -> ListDataPayload:
    return ListDataPayload(
        meta=ListMeta(id=list_id, user_id=1, name=name, is_public=True),
        elements=[{"id": 1, "text": name, "params": None, "description": None}],
        embeddings_data=None, model=None,
    )


async def builder(p: ListDataPayload):
    return make_data(p.meta.id, user_id=p.meta.user_id, memory=len(p.elements))


def service(backend: FakeBackend, ttl: float):
    catalog = ListCatalog()
    cache = ListDataCache(max_bytes=10_000, ttl_seconds=ttl)
    return ListDataService(cache, backend, catalog, builder), cache, catalog


def test_is_stale_only_after_ttl():
    cache = ListDataCache(max_bytes=1000, ttl_seconds=1000)
    cache.put(1, make_data(1))
    assert not cache.is_stale(1)
    cache._loaded_at[1] -= 2000  # envejecer a mano
    assert cache.is_stale(1)
    assert not ListDataCache(max_bytes=1000).is_stale(1)  # TTL desactivado: nunca


async def test_stale_entry_is_served_and_revalidated_in_background():
    backend = FakeBackend()
    backend.list_payloads[1] = payload(1, "v1")
    svc, cache, _ = service(backend, ttl=1000)
    first = await svc.get_for_search(1)
    assert backend.list_fetches == [1]

    backend.list_payloads[1] = payload(1, "v2")
    cache._loaded_at[1] -= 5000  # caducada
    served = await svc.get_for_search(1)
    assert served is first  # se sirve lo cacheado sin esperar
    await asyncio.sleep(0.05)  # la revalidación corre aparte
    assert backend.list_fetches == [1, 1]
    assert cache.get(1) is not first  # y la caché ya es la nueva
    assert not cache.is_stale(1)
    assert not svc.is_serving_stale(1)


async def test_failed_revalidation_marks_stale_until_it_succeeds():
    backend = FakeBackend()
    backend.list_payloads[1] = payload(1)
    svc, cache, _ = service(backend, ttl=1000)
    original = await svc.get_for_search(1)

    backend.fail_list_fetch = True
    cache._loaded_at[1] -= 5000
    assert await svc.get_for_search(1) is original
    await asyncio.sleep(0.05)
    assert svc.is_serving_stale(1)  # el backend no responde: lo viejo, pero degradado
    assert cache.get(1) is original

    backend.fail_list_fetch = False
    assert await svc.revalidate_now(1)
    assert not svc.is_serving_stale(1)


async def test_push_during_revalidation_wins():
    backend = FakeBackend()
    backend.list_payloads[1] = payload(1)
    svc, cache, _ = service(backend, ttl=1000)
    await svc.get_for_search(1)
    pushed = make_data(1, memory=999)

    async def slow_loader():
        await asyncio.sleep(0.05)
        return make_data(1, memory=5)

    task = asyncio.create_task(cache.reload(1, slow_loader))
    await asyncio.sleep(0)  # la revalidación ya está dentro del loader...
    cache.put(1, pushed)  # ...cuando llega el push de fin de entrenamiento
    assert await task is True
    assert cache.get(1) is pushed


async def test_reconcile_drops_lists_missing_from_catalog():
    backend = FakeBackend()
    backend.list_payloads[1] = payload(1)
    backend.list_payloads[2] = payload(2)
    svc, cache, catalog = service(backend, ttl=0)
    await svc.get_for_search(1)
    await svc.get_for_search(2)
    catalog.replace_all([ListMeta(id=2, user_id=1, name="L", is_public=True)])  # re-sync sin la 1
    assert svc.reconcile() == 1
    assert cache.get(1) is None and cache.get(2) is not None
