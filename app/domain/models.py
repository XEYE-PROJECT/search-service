"""Modelos de dominio: datos puros, sin dependencias de framework ni E/S."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from app.infrastructure.vector_index import VectorIndex


@dataclass(frozen=True)
class ApiKeyInfo:
    """Resolución de una API key (por su hash): a quién pertenece."""

    id: int
    user_id: int


@dataclass(frozen=True)
class ListMeta:
    """Metadatos ligeros que se guardan de cada lista (auth + resolución de nombre)."""

    id: int
    user_id: int
    name: str
    is_public: bool


@dataclass
class ListSearchData:
    """Todo lo necesario para buscar en una lista, residente por completo en RAM.

    ``embeddings``: matriz float32 L2-normalizada de los elementos CON vector entrenado,
    ya alineada por id de elemento (los creados tras el lanzamiento no tienen vector; los
    vectores de borrados se descartan). ``vector_rows[i]`` es la fila del elemento i
    (-1 = sin vector → puntúa solo por texto) y ``element_of_row`` la inversa; las tres
    son None si la lista no tiene embeddings usables.
    """

    list_id: int
    user_id: int
    element_ids: list[int | None]
    texts: list[str]
    processed: list[str]
    params: list[str | None]
    embeddings: "np.ndarray | None"
    vector_rows: "np.ndarray | None"
    element_of_row: "np.ndarray | None"
    model_name: str | None
    index: "VectorIndex | None"
    memory_bytes: int = 0

    @property
    def size(self) -> int:
        return len(self.texts)


@dataclass
class SearchHit:
    """Un resultado puntuado antes de serializar."""

    item: str
    score: float
    params_raw: str | None
    text_score: float = 0.0
    semantic_score: float = 0.0


@dataclass
class SearchOutcome:
    """Resultado de ejecutar el caso de uso de búsqueda."""

    hits: list[SearchHit] = field(default_factory=list)
    duration_ms: int = 0
