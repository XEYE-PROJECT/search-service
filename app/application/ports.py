"""Puertos (interfaces Protocol) de los que depende la capa de aplicación: infraestructura
aporta las implementaciones reales y los tests, fakes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from app.domain.models import ListMeta


@dataclass(frozen=True)
class BootstrapData:
    """Snapshot que sirve el backend en el arranque / refresh completo."""

    api_keys: list[tuple[int, int, str]]  # (id, user_id, key_hash) — SHA-256 hex, nunca en claro
    lists: list[ListMeta]
    # Modelos con los que pueden lanzarse entrenamientos — se precalientan al arrancar
    # para que la primera búsqueda con cada uno no pague la carga/descarga.
    embedding_models: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ListDataPayload:
    """Datos de búsqueda completos de una lista, como los sirve (o empuja) el backend."""

    meta: ListMeta
    elements: list[dict[str, Any]]  # {id, text, params, description}
    embeddings_data: str | None
    model: str | None  # string opaco del modelo del entrenamiento (JSON codificado)
    # Ids de elementos al lanzar: la fila i de embeddings pertenece a trained_element_ids[i].
    # None en entrenamientos legacy (previos a capturar ids) — entonces se asume que las
    # filas casan 1:1 con los elementos actuales y se descartan si el conteo no cuadra.
    trained_element_ids: list[int] | None = None


@dataclass
class LogEntry:
    """Una llamada de search/target a persistir en la base de datos del backend."""

    user_id: int
    api_key_id: int | None
    list_id: int | None
    list_name: str
    endpoint: str
    search_term: str
    total_results: int
    duration_ms: int
    session: str | None
    results: dict[str, float] | None
    searched_at: str  # ISO-8601 UTC
    extra: dict[str, Any] = field(default_factory=dict)


class BackendGateway(Protocol):
    async def fetch_bootstrap(self) -> BootstrapData: ...

    async def fetch_list_data(self, list_id: int) -> ListDataPayload | None:
        """None significa que el backend respondió 404 (lista borrada). Los errores lanzan."""
        ...

    async def push_logs(self, entries: list[LogEntry]) -> None:
        """Lanza si falla (la cola reintenta)."""
        ...


class QueryEmbedder(Protocol):
    async def embed_query(self, model_name: str | None, text: str) -> np.ndarray | None: ...
