"""Catálogos en RAM: API keys (auth, solo hashes), metadatos de listas (nombres + gate de
pública) y límites de búsqueda por usuario (los que difieren del plan por defecto).

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
from app.core.security import hash_api_key
from app.domain.models import ApiKeyInfo, ListMeta
from app.infrastructure import metrics

logger = logging.getLogger(__name__)


class ApiKeyStore:
    """Catálogo de API keys indexado por su SHA-256: aquí nunca vive una key en claro.

    El backend envía solo hashes (bootstrap y ``PUT /v1/api-keys/{id}``); la cabecera
    ``X-API-Key`` de cada búsqueda se hashea con ``hash_api_key`` y se busca tal cual.
    """

    def __init__(self) -> None:
        self._by_hash: dict[str, ApiKeyInfo] = {}
        self._hash_by_id: dict[int, str] = {}
        self.mutations = 0  # lo suben los cambios por push (no replace_all)

    def resolve_raw(self, raw_key: str) -> ApiKeyInfo | None:
        """Resuelve una key tal como llega en la cabecera (en claro)."""
        return self._by_hash.get(hash_api_key(raw_key))

    def resolve_hash(self, key_hash: str) -> ApiKeyInfo | None:
        return self._by_hash.get(key_hash)

    def replace_all(self, entries: list[tuple[int, int, str]]) -> None:
        """``entries`` = ``(id, user_id, key_hash)``."""
        self._by_hash = {key_hash: ApiKeyInfo(id=key_id, user_id=user_id) for key_id, user_id, key_hash in entries}
        self._hash_by_id = {key_id: key_hash for key_id, user_id, key_hash in entries}

    def upsert(self, key_id: int, user_id: int, key_hash: str) -> None:
        self._remove_only(key_id)
        self._by_hash[key_hash] = ApiKeyInfo(id=key_id, user_id=user_id)
        self._hash_by_id[key_id] = key_hash
        self.mutations += 1

    def remove(self, key_id: int) -> None:
        self._remove_only(key_id)
        self.mutations += 1

    def remove_user(self, user_id: int) -> None:
        for key_id in [k.id for k in self._by_hash.values() if k.user_id == user_id]:
            self._remove_only(key_id)
        self.mutations += 1

    def _remove_only(self, key_id: int) -> None:
        key_hash = self._hash_by_id.pop(key_id, None)
        if key_hash is not None:
            self._by_hash.pop(key_hash, None)

    def __len__(self) -> int:
        return len(self._by_hash)


class UserLimits:
    """Búsquedas/minuto por usuario fijadas desde el backend (admin). Solo guarda las que
    difieren del valor por defecto (``RATE_LIMIT_PER_MINUTE``); todas las keys del usuario y
    las búsquedas desde la consola comparten ese cupo."""

    def __init__(self) -> None:
        self._by_user: dict[int, int] = {}
        self.mutations = 0  # lo suben los cambios por push (no replace_all)

    def get(self, user_id: int, default: int) -> int:
        return self._by_user.get(user_id, default)

    def replace_all(self, entries: list[tuple[int, int]]) -> None:
        """``entries`` = ``(user_id, rate_limit_per_minute)``."""
        self._by_user = {user_id: limit for user_id, limit in entries}

    def set(self, user_id: int, limit: int | None) -> None:
        """``None`` (o un valor no positivo) vuelve al límite por defecto."""
        if limit is None or limit <= 0:
            self._by_user.pop(user_id, None)
        else:
            self._by_user[user_id] = limit
        self.mutations += 1

    def remove_user(self, user_id: int) -> None:
        self._by_user.pop(user_id, None)
        self.mutations += 1

    def __len__(self) -> int:
        return len(self._by_user)


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
        user_limits: UserLimits | None = None,
        min_refresh_interval: float = 30.0,
    ) -> None:
        self._backend = backend
        self.api_keys = api_keys
        self.lists = lists
        self.user_limits = user_limits if user_limits is not None else UserLimits()
        self._min_refresh_interval = min_refresh_interval
        self._refresh_lock = asyncio.Lock()
        # None = nunca refrescado. (No usar 0.0: time.monotonic() cuenta desde el arranque de la
        # máquina y en una recién encendida el throttle se saltaría el primer refresh.)
        self._last_refresh: float | None = None
        self._refreshes_applied = 0
        self.ready = False
        # Modelos de embedding disponibles según el último bootstrap (se precalientan al
        # arrancar para que ninguna primera búsqueda pague la carga del modelo).
        self.embedding_models: list[str] = []

    async def refresh(self) -> None:
        """Re-sync completo desde el backend (lanza si el fetch falla)."""
        async with self._refresh_lock:
            try:
                await self._refresh_locked()
            except Exception:
                metrics.CATALOG_REFRESHES.labels("error").inc()
                raise
            metrics.CATALOG_REFRESHES.labels("ok").inc()

    async def refresh_on_miss(self) -> bool:
        """Refresh al no resolver una key/lista (quizá recién creada y con el push perdido).
        Con throttle y single-flight con los refresh normales. Devuelve True si corrió un
        refresh — incluido uno terminado mientras esperábamos el lock — para que el
        llamante re-resuelva contra datos frescos en ambos casos."""
        applied_before = self._refreshes_applied
        async with self._refresh_lock:
            if self._refreshes_applied != applied_before:
                return True  # otro refrescó mientras esperábamos
            if self._last_refresh is not None and time.monotonic() - self._last_refresh < self._min_refresh_interval:
                return False
            try:
                await self._refresh_locked()
                return True
            except Exception:
                logger.exception("Miss-triggered catalog refresh failed")
                return False

    async def _refresh_locked(self) -> None:
        for _attempt in range(self._MAX_REFRESH_ATTEMPTS):
            marker = self._mutation_marker()
            snapshot = await self._backend.fetch_bootstrap()
            if self._mutation_marker() != marker:
                continue  # un push llegó en pleno fetch; el snapshot puede ser anterior — refetch
            self.api_keys.replace_all(snapshot.api_keys)
            self.lists.replace_all(snapshot.lists)
            self.user_limits.replace_all(snapshot.user_limits)
            self.embedding_models = list(snapshot.embedding_models)
            self._last_refresh = time.monotonic()
            self._refreshes_applied += 1
            self.ready = True
            logger.info(
                "Catalog refreshed: %d api keys, %d lists, %d user limits",
                len(self.api_keys),
                len(self.lists),
                len(self.user_limits),
            )
            return
        # Siguieron llegando pushes en pleno fetch: esos mismos pushes mantienen fresco el
        # catálogo, así que saltarse este snapshot es seguro. Se reintenta el próximo ciclo.
        self._last_refresh = time.monotonic()
        logger.warning("Catalog refresh skipped after %d attempts (concurrent updates)", self._MAX_REFRESH_ATTEMPTS)

    def _mutation_marker(self) -> int:
        return self.api_keys.mutations + self.lists.mutations + self.user_limits.mutations
