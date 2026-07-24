"""Decide de dónde salen los datos de búsqueda de una lista: caché, carga perezosa desde
el backend o push directo de índice. De paso mantiene el catálogo coherente."""

from __future__ import annotations

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

    async def get_for_search(self, list_id: int) -> ListSearchData | None:
        """Datos cacheados, cargados perezosamente del backend si faltan.

        Devuelve None si el backend dice que la lista ya no existe; lanza
        ``BackendUnavailableError`` si no es alcanzable.
        """

        async def loader() -> ListSearchData | None:
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

        return await self._cache.get_or_load(list_id, loader)

    async def apply_push(self, payload: ListDataPayload) -> None:
        """Reemplazo completo desde el push de fin de entrenamiento (deja la caché caliente)."""
        self._catalog.upsert(payload.meta)
        data = await self._builder(payload)
        self._cache.put(payload.meta.id, data)

    def invalidate(self, list_id: int) -> None:
        self._cache.invalidate(list_id)

    def remove_list(self, list_id: int) -> None:
        self._catalog.remove(list_id)
        self._cache.remove(list_id)

    def remove_user(self, user_id: int) -> None:
        for list_id in self._catalog.remove_user(user_id):
            self._cache.remove(list_id)
