"""Catálogos en RAM: API keys (auth) y metadatos de listas (nombres + gate de pública).

Son pequeños: residen completos (sin LRU) y solo mutan en el event loop (sin locks). El
backend los mantiene frescos por push; el refresh periódico cubre pushes perdidos.
Carrera refresh vs push: un push puede llegar con un snapshot de bootstrap en vuelo y
aplicarlo tal cual lo desharía (p. ej. resucitar una key revocada) — por eso ambos stores
cuentan mutaciones y el refresh descarta su snapshot y refetchea si el contador se movió.
"""

from __future__ import annotations

import asyncio
import logging
import time

from app.application.ports import BackendGateway
from app.domain.models import ApiKeyInfo, ListMeta

logger = logging.getLogger(__name__)


class ApiKeyStore:
    def __init__(self) -> None:
        self._by_raw: dict[str, ApiKeyInfo] = {}
        self._raw_by_id: dict[int, str] = {}
        self.mutations = 0  # lo suben los cambios por push (no replace_all)

    def resolve(self, raw_key: str) -> ApiKeyInfo | None:
        return self._by_raw.get(raw_key)

    def replace_all(self, entries: list[tuple[int, int, str]]) -> None:
        self._by_raw = {raw: ApiKeyInfo(id=key_id, user_id=user_id) for key_id, user_id, raw in entries}
        self._raw_by_id = {key_id: raw for key_id, user_id, raw in entries}

    def upsert(self, key_id: int, user_id: int, raw_key: str) -> None:
        self._remove_only(key_id)
        self._by_raw[raw_key] = ApiKeyInfo(id=key_id, user_id=user_id)
        self._raw_by_id[key_id] = raw_key
        self.mutations += 1

    def remove(self, key_id: int) -> None:
        self._remove_only(key_id)
        self.mutations += 1

    def remove_user(self, user_id: int) -> None:
        for key_id in [k.id for k in self._by_raw.values() if k.user_id == user_id]:
            self._remove_only(key_id)
        self.mutations += 1

    def _remove_only(self, key_id: int) -> None:
        raw = self._raw_by_id.pop(key_id, None)
        if raw is not None:
            self._by_raw.pop(raw, None)

    def __len__(self) -> int:
        return len(self._by_raw)


class ListCatalog:
    def __init__(self) -> None:
        self._by_id: dict[int, ListMeta] = {}
        self._id_by_name: dict[tuple[int, str], int] = {}
        self.mutations = 0  # lo suben los cambios por push (no replace_all)

    def get(self, list_id: int) -> ListMeta | None:
        return self._by_id.get(list_id)

    def resolve(self, user_id: int, name: str) -> ListMeta | None:
        list_id = self._id_by_name.get((user_id, name))
        return self._by_id.get(list_id) if list_id is not None else None

    def replace_all(self, metas: list[ListMeta]) -> None:
        self._by_id = {meta.id: meta for meta in metas}
        self._id_by_name = {(meta.user_id, meta.name): meta.id for meta in metas}

    def upsert(self, meta: ListMeta) -> None:
        self._remove_only(meta.id)
        self._by_id[meta.id] = meta
        self._id_by_name[(meta.user_id, meta.name)] = meta.id
        self.mutations += 1

    def remove(self, list_id: int) -> None:
        self._remove_only(list_id)
        self.mutations += 1

    def remove_user(self, user_id: int) -> list[int]:
        removed = [meta.id for meta in self._by_id.values() if meta.user_id == user_id]
        for list_id in removed:
            self._remove_only(list_id)
        self.mutations += 1
        return removed

    def _remove_only(self, list_id: int) -> None:
        previous = self._by_id.pop(list_id, None)
        if previous is not None:
            self._id_by_name.pop((previous.user_id, previous.name), None)

    def __len__(self) -> int:
        return len(self._by_id)


class CatalogService:
    """Orquesta el bootstrap y el refresh de ambos catálogos (single-flight, con throttle)."""

    _MAX_REFRESH_ATTEMPTS = 3

    def __init__(
        self,
        backend: BackendGateway,
        api_keys: ApiKeyStore,
        lists: ListCatalog,
        *,
        min_refresh_interval: float = 30.0,
    ) -> None:
        self._backend = backend
        self.api_keys = api_keys
        self.lists = lists
        self._min_refresh_interval = min_refresh_interval
        self._refresh_lock = asyncio.Lock()
        self._last_refresh = 0.0
        self._refreshes_applied = 0
        self.ready = False
        # Modelos de embedding disponibles según el último bootstrap (se precalientan al
        # arrancar para que ninguna primera búsqueda pague la carga del modelo).
        self.embedding_models: list[str] = []

    async def refresh(self) -> None:
        """Re-sync completo desde el backend (lanza si el fetch falla)."""
        async with self._refresh_lock:
            await self._refresh_locked()

    async def refresh_on_miss(self) -> bool:
        """Refresh al no resolver una key/lista (quizá recién creada y con el push perdido).
        Con throttle y single-flight con los refresh normales. Devuelve True si corrió un
        refresh — incluido uno terminado mientras esperábamos el lock — para que el
        llamante re-resuelva contra datos frescos en ambos casos."""
        applied_before = self._refreshes_applied
        async with self._refresh_lock:
            if self._refreshes_applied != applied_before:
                return True  # otro refrescó mientras esperábamos
            if time.monotonic() - self._last_refresh < self._min_refresh_interval:
                return False
            try:
                await self._refresh_locked()
                return True
            except Exception:
                logger.exception("Miss-triggered catalog refresh failed")
                return False

    async def _refresh_locked(self) -> None:
        for _attempt in range(self._MAX_REFRESH_ATTEMPTS):
            marker = self.api_keys.mutations + self.lists.mutations
            snapshot = await self._backend.fetch_bootstrap()
            if self.api_keys.mutations + self.lists.mutations != marker:
                continue  # un push llegó en pleno fetch; el snapshot puede ser anterior — refetch
            self.api_keys.replace_all(snapshot.api_keys)
            self.lists.replace_all(snapshot.lists)
            self.embedding_models = list(snapshot.embedding_models)
            self._last_refresh = time.monotonic()
            self._refreshes_applied += 1
            self.ready = True
            logger.info(
                "Catalog refreshed: %d api keys, %d lists", len(self.api_keys), len(self.lists)
            )
            return
        # Siguieron llegando pushes en pleno fetch: esos mismos pushes mantienen fresco el
        # catálogo, así que saltarse este snapshot es seguro. Se reintenta el próximo ciclo.
        self._last_refresh = time.monotonic()
        logger.warning("Catalog refresh skipped after %d attempts (concurrent updates)",
                       self._MAX_REFRESH_ATTEMPTS)
