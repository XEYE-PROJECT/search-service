"""Caché en RAM acotada de los datos de búsqueda por lista.

LRU por BYTES, no por entradas (cada ``ListSearchData`` trae su huella real), y carga
single-flight con lock asyncio por lista. Las cargas compiten con los pushes: si un
push/invalidate/delete llega con una carga en vuelo, cachear lo cargado fijaría datos
pre-push para siempre. Por eso cada mutación sube una generación por lista y
``get_or_load`` solo cachea si la generación no se movió: la petición actual se sirve
(quizá un pelo caduca) y la siguiente recarga en fresco.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable

from app.domain.models import ListSearchData

logger = logging.getLogger(__name__)


class ListDataCache:
    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes
        self._data: OrderedDict[int, ListSearchData] = OrderedDict()
        self._locks: dict[int, asyncio.Lock] = {}
        self._generations: dict[int, int] = {}
        self._total_bytes = 0

    def get(self, list_id: int) -> ListSearchData | None:
        data = self._data.get(list_id)
        if data is not None:
            self._data.move_to_end(list_id)
        return data

    async def get_or_load(
        self,
        list_id: int,
        loader: Callable[[], Awaitable[ListSearchData | None]],
    ) -> ListSearchData | None:
        data = self.get(list_id)
        if data is not None:
            return data
        lock = self._locks.setdefault(list_id, asyncio.Lock())
        async with lock:
            data = self.get(list_id)  # otra petición pudo cargarlo entretanto
            if data is not None:
                return data
            generation = self._generations.get(list_id, 0)
            data = await loader()
            if data is not None:
                if self._generations.get(list_id, 0) == generation:
                    self.put(list_id, data)
                else:
                    # Llegó un push/invalidate/delete en plena carga: se sirve este
                    # resultado pero no se cachea — la próxima búsqueda recarga.
                    logger.info("List %d changed during load; result not cached", list_id)
            return data

    def put(self, list_id: int, data: ListSearchData) -> None:
        self._drop(list_id)
        self._data[list_id] = data
        self._total_bytes += data.memory_bytes
        self._bump(list_id)
        self._evict_if_needed()

    def invalidate(self, list_id: int) -> None:
        """Descarta lo cacheado (los metadatos siguen en el catálogo); la próxima búsqueda recarga."""
        self._drop(list_id)
        self._bump(list_id)

    def remove(self, list_id: int) -> None:
        self._drop(list_id)
        self._bump(list_id)
        # La generación se conserva (es diminuta) para que un loader en vuelo vea el cambio.
        self._locks.pop(list_id, None)

    def remove_user(self, user_id: int) -> None:
        for list_id in [d.list_id for d in self._data.values() if d.user_id == user_id]:
            self.remove(list_id)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def __len__(self) -> int:
        return len(self._data)

    def _bump(self, list_id: int) -> None:
        self._generations[list_id] = self._generations.get(list_id, 0) + 1

    def _drop(self, list_id: int) -> None:
        previous = self._data.pop(list_id, None)
        if previous is not None:
            self._total_bytes -= previous.memory_bytes

    def _evict_if_needed(self) -> None:
        # Nunca expulsar la única entrada: una lista sobredimensionada debe poder servirse.
        while self._total_bytes > self._max_bytes and len(self._data) > 1:
            evicted_id, evicted = self._data.popitem(last=False)
            self._total_bytes -= evicted.memory_bytes
            self._locks.pop(evicted_id, None)
            logger.info(
                "Evicted list %d from cache (%.1f MiB freed, %.1f MiB in use)",
                evicted_id, evicted.memory_bytes / 2**20, self._total_bytes / 2**20,
            )
