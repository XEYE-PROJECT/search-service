"""Decide de dónde salen los datos de búsqueda de una lista: caché, carga perezosa desde
el backend o push directo de índice. De paso mantiene el catálogo coherente.

Con TTL, una entrada caduca se sirve igualmente y se revalida en segundo plano (una sola
revalidación en vuelo por lista). Si la revalidación falla (backend caído) la lista queda
marcada como ``stale`` hasta que alguna recarga tenga éxito: la búsqueda lo señala como
degradación en la respuesta.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from app.application.catalog import ListCatalog
from app.application.errors import BackendUnavailableError
from app.application.ports import BackendGateway, ListDataPayload
from app.domain.models import ListSearchData

logger = logging.getLogger(__name__)

Builder = Callable[[ListDataPayload], Awaitable[ListSearchData]]


class ListDataService:
    def __init__(
        self,
        cache,  # ListDataCache
        backend: BackendGateway,
        catalog: ListCatalog,
        builder: Builder,
    ) -> None:
        self._cache = cache
        self._backend = backend
        self._catalog = catalog
        self._builder = builder
        self._revalidating: set[int] = set()
        self._stale: set[int] = set()  # revalidación fallida: se sirve lo viejo (degradado)

    async def get_for_search(self, list_id: int) -> ListSearchData | None:
        """Datos cacheados, cargados perezosamente del backend si faltan.

        Devuelve None si el backend dice que la lista ya no existe; lanza
        ``BackendUnavailableError`` si no es alcanzable.
        """
        data = await self._cache.get_or_load(list_id, lambda: self._load(list_id))
        if data is not None and self._cache.is_stale(list_id):
            self._schedule_revalidation(list_id)
        return data

    def is_serving_stale(self, list_id: int) -> bool:
        return list_id in self._stale

    async def _load(self, list_id: int) -> ListSearchData | None:
        try:
            payload = await self._backend.fetch_list_data(list_id)
        except Exception as exc:
            logger.warning("Backend fetch for list %d failed: %s", list_id, exc)
            raise BackendUnavailableError() from exc
        if payload is None:  # 404: ya no existe — limpiamos nuestra propia vista
            self._catalog.remove(list_id)
            return None
        self._catalog.upsert(payload.meta)
        return await self._builder(payload)

    def _schedule_revalidation(self, list_id: int) -> None:
        if list_id in self._revalidating:
            return
        self._revalidating.add(list_id)
        asyncio.create_task(self._revalidate(list_id), name=f"revalidate-{list_id}")

    async def _revalidate(self, list_id: int) -> None:
        try:
            fresh = await self._cache.reload(list_id, lambda: self._load(list_id))
            if fresh:
                self._stale.discard(list_id)
            else:
                self._stale.add(list_id)
        except Exception as exc:  # el loader ya está protegido; esto es un cinturón
            logger.warning("Revalidation task for list %d crashed: %s", list_id, exc)
            self._stale.add(list_id)
        finally:
            self._revalidating.discard(list_id)

    async def revalidate_now(self, list_id: int) -> bool:
        """Revalidación síncrona (tests y operación)."""
        await self._revalidate(list_id)
        return list_id not in self._stale

    async def apply_push(self, payload: ListDataPayload) -> None:
        """Reemplazo completo desde el push de fin de entrenamiento (deja la caché caliente)."""
        self._catalog.upsert(payload.meta)
        data = await self._builder(payload)
        self._cache.put(payload.meta.id, data)
        self._stale.discard(payload.meta.id)

    def invalidate(self, list_id: int) -> None:
        self._cache.invalidate(list_id)
        self._stale.discard(list_id)

    def remove_list(self, list_id: int) -> None:
        self._catalog.remove(list_id)
        self._cache.remove(list_id)
        self._stale.discard(list_id)

    def remove_user(self, user_id: int) -> None:
        for list_id in self._catalog.remove_user(user_id):
            self._cache.remove(list_id)
            self._stale.discard(list_id)

    def reconcile(self) -> int:
        """Tras un re-sync del catálogo: descarta las listas cacheadas que el backend ya no
        tiene (borradas mientras la notificación se perdió). Devuelve cuántas quitó."""
        removed = 0
        for list_id in self._cache.cached_ids():
            if self._catalog.get(list_id) is None:
                self._cache.remove(list_id)
                self._stale.discard(list_id)
                removed += 1
        if removed:
            logger.info("Dropped %d cached list(s) no longer present in the backend", removed)
        return removed
