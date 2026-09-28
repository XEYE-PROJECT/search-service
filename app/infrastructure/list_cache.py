"""Caché en RAM acotada de los datos de búsqueda por lista.

LRU por BYTES, no por entradas (cada ``ListSearchData`` trae su huella real), y carga
single-flight con lock asyncio por lista. Las cargas compiten con los pushes: si un
push/invalidate/delete llega con una carga en vuelo, cachear lo cargado fijaría datos
pre-push para siempre. Por eso cada mutación sube una generación por lista y
``get_or_load`` solo cachea si la generación no se movió: la petición actual se sirve
(quizá un pelo caduca) y la siguiente recarga en fresco.

TTL (``ttl_seconds``, 0 = sin TTL): una entrada más vieja que el TTL sigue sirviéndose, pero
``is_stale`` la señala para que ``ListDataService`` la revalide en segundo plano
(stale-while-revalidate). Es la red de seguridad para invalidaciones que nunca llegaron.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable

from app.domain.models import ListSearchData
from app.infrastructure import metrics

logger = logging.getLogger(__name__)


class ListDataCache:
    def __init__(self, max_bytes: int, ttl_seconds: float = 0.0) -> None:
        self._max_bytes = max_bytes
        self._ttl = ttl_seconds
        self._data: OrderedDict[int, ListSearchData] = OrderedDict()
        self._loaded_at: dict[int, float] = {}
        self._locks: dict[int, asyncio.Lock] = {}
        self._generations: dict[int, int] = {}
        self._total_bytes = 0

    def get(self, list_id: int) -> ListSearchData | None:
        data = self._data.get(list_id)
        if data is not None:
            self._data.move_to_end(list_id)
        return data

    def is_stale(self, list_id: int) -> bool:
        """True si la entrada existe y supera el TTL (nunca con TTL desactivado)."""
        if self._ttl <= 0:
            return False
        loaded_at = self._loaded_at.get(list_id)
        return loaded_at is not None and time.monotonic() - loaded_at > self._ttl

    def age_seconds(self, list_id: int) -> float | None:
        loaded_at = self._loaded_at.get(list_id)
        return None if loaded_at is None else time.monotonic() - loaded_at

    async def get_or_load(
        self,
        list_id: int,
        loader: Callable[[], Awaitable[ListSearchData | None]],
    ) -> ListSearchData | None:
        data = self.get(list_id)
        if data is not None:
            metrics.CACHE_EVENTS.labels("hit").inc()
            return data
        metrics.CACHE_EVENTS.labels("miss").inc()
        lock = self._locks.setdefault(list_id, asyncio.Lock())
        async with lock:
            data = self.get(list_id)  # otra petición pudo cargarlo entretanto
            if data is not None:
                return data
            generation = self._generations.get(list_id, 0)
            data = await loader()
            metrics.CACHE_EVENTS.labels("load").inc()
            if data is not None:
                if self._generations.get(list_id, 0) == generation:
                    self.put(list_id, data)
                else:
                    # Llegó un push/invalidate/delete en plena carga: se sirve este
                    # resultado pero no se cachea — la próxima búsqueda recarga.
                    logger.info("List %d changed during load; result not cached", list_id)
            return data

    async def reload(
        self,
        list_id: int,
        loader: Callable[[], Awaitable[ListSearchData | None]],
    ) -> bool:
        """Revalidación en segundo plano: recarga y sustituye la entrada si nadie la tocó
        mientras tanto. Devuelve True si la caché quedó fresca (recargada o ya sustituida por
        un push); False si el loader falló o la lista ya no existe."""
        lock = self._locks.setdefault(list_id, asyncio.Lock())
        async with lock:
            generation = self._generations.get(list_id, 0)
            try:
                data = await loader()
            except Exception as exc:
                metrics.CACHE_EVENTS.labels("revalidate_failed").inc()
                logger.warning("Revalidation of list %d failed; serving stale data: %s", list_id, exc)
                return False
            if self._generations.get(list_id, 0) != generation:
                return True  # un push/invalidate ganó: lo suyo es más fresco que esto
            if data is None:
                self.remove(list_id)
                return False
            self.put(list_id, data)
            metrics.CACHE_EVENTS.labels("revalidate").inc()
            return True

    def put(self, list_id: int, data: ListSearchData) -> None:
        self._drop(list_id)
        self._data[list_id] = data
        self._loaded_at[list_id] = time.monotonic()
        self._total_bytes += data.memory_bytes
        self._bump(list_id)
        self._evict_if_needed()

    def invalidate(self, list_id: int) -> None:
        """Descarta lo cacheado (los metadatos siguen en el catálogo); la próxima búsqueda recarga."""
        self._drop(list_id)
        self._bump(list_id)
        metrics.CACHE_EVENTS.labels("invalidate").inc()

    def remove(self, list_id: int) -> None:
        self._drop(list_id)
        self._bump(list_id)
        # La generación se conserva (es diminuta) para que un loader en vuelo vea el cambio.
        self._locks.pop(list_id, None)

    def remove_user(self, user_id: int) -> None:
        for list_id in [d.list_id for d in self._data.values() if d.user_id == user_id]:
            self.remove(list_id)

    def cached_ids(self) -> list[int]:
        return list(self._data.keys())

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def __len__(self) -> int:
        return len(self._data)

    def _bump(self, list_id: int) -> None:
        self._generations[list_id] = self._generations.get(list_id, 0) + 1

    def _drop(self, list_id: int) -> None:
        previous = self._data.pop(list_id, None)
        self._loaded_at.pop(list_id, None)
        if previous is not None:
            self._total_bytes -= previous.memory_bytes

    def _evict_if_needed(self) -> None:
        # Nunca expulsar la única entrada: una lista sobredimensionada debe poder servirse.
        while self._total_bytes > self._max_bytes and len(self._data) > 1:
            evicted_id, evicted = self._data.popitem(last=False)
            self._loaded_at.pop(evicted_id, None)
            self._total_bytes -= evicted.memory_bytes
            self._locks.pop(evicted_id, None)
            metrics.CACHE_EVENTS.labels("evict").inc()
            logger.info(
                "Evicted list %d from cache (%.1f MiB freed, %.1f MiB in use)",
                evicted_id, evicted.memory_bytes / 2**20, self._total_bytes / 2**20,
            )
