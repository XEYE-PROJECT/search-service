import asyncio

from app.domain.models import ListSearchData
from app.infrastructure.list_cache import ListDataCache


def make_data(list_id: int, user_id: int = 1, memory: int = 100) -> ListSearchData:
    return ListSearchData(
        list_id=list_id, user_id=user_id, element_ids=[], texts=[], processed=[],
        params=[], embeddings=None, vector_rows=None, element_of_row=None,
        model_name=None, index=None, memory_bytes=memory,
    )


def test_lru_evicts_by_bytes():
    cache = ListDataCache(max_bytes=250)
    cache.put(1, make_data(1, memory=100))
    cache.put(2, make_data(2, memory=100))
    cache.put(3, make_data(3, memory=100))  # 300 > 250 -> expulsa la lista 1 (la más vieja)
    assert cache.get(1) is None
    assert cache.get(2) is not None
    assert cache.get(3) is not None
    assert cache.total_bytes == 200


def test_lru_recency_updated_on_get():
    cache = ListDataCache(max_bytes=250)
    cache.put(1, make_data(1, memory=100))
    cache.put(2, make_data(2, memory=100))
    cache.get(1)  # la 1 pasa a ser la más reciente
    cache.put(3, make_data(3, memory=100))  # ahora la 2 es la entrada LRU
    assert cache.get(2) is None
    assert cache.get(1) is not None


def test_single_oversized_entry_is_kept():
    cache = ListDataCache(max_bytes=100)
    cache.put(1, make_data(1, memory=500))
    assert cache.get(1) is not None


def test_put_replaces_and_adjusts_bytes():
    cache = ListDataCache(max_bytes=1000)
    cache.put(1, make_data(1, memory=100))
    cache.put(1, make_data(1, memory=300))
    assert cache.total_bytes == 300


def test_invalidate_and_remove_user():
    cache = ListDataCache(max_bytes=1000)
    cache.put(1, make_data(1, user_id=7))
    cache.put(2, make_data(2, user_id=8))
    cache.invalidate(1)
    assert cache.get(1) is None
    cache.remove_user(8)
    assert cache.get(2) is None
    assert cache.total_bytes == 0


async def test_get_or_load_is_single_flight():
    cache = ListDataCache(max_bytes=1000)
    loads = 0

    async def loader():
        nonlocal loads
        loads += 1
        await asyncio.sleep(0.05)
        return make_data(1)

    results = await asyncio.gather(*[cache.get_or_load(1, loader) for _ in range(5)])
    assert loads == 1
    assert all(r is not None for r in results)


async def test_get_or_load_negative_result_not_cached():
    cache = ListDataCache(max_bytes=1000)

    async def loader_none():
        return None

    assert await cache.get_or_load(1, loader_none) is None
    assert len(cache) == 0
