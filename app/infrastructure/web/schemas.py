"""Esquemas de petición/respuesta.

La API pública va en snake_case (el contrato histórico del frontend y la doc de clientes).
Los cuerpos internos del backend llegan en camelCase (defaults de Jackson) — aquí se
mapean con alias; el backend serializa los null explícitos, de ahí tanto Optional.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- pública --


class PublicSearchRequest(BaseModel):
    list_name: str = Field(min_length=1, max_length=100)
    search_term: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=50, ge=1, le=1000)
    session: str | None = Field(default=None, max_length=255)
    include_score_breakdown: bool = False
    register_log: bool = True
    allow_private: bool = False


class PublicTargetRequest(BaseModel):
    list_name: str = Field(min_length=1, max_length=100)
    target_term: str = Field(min_length=1, max_length=500)
    session: str = Field(min_length=1, max_length=255)


class SearchResultItem(BaseModel):
    item: str
    score: float
    params: Any | None = None
    text_score: float | None = None
    semantic_score: float | None = None


class SearchResponse(BaseModel):
    success: bool = True
    results: list[SearchResultItem]
    total_results: int
    search_term: str
    list_name: str
    duration_ms: int
    error: str | None = None


class TargetResponse(BaseModel):
    success: bool = True


# --------------------------------------------------------------------------- interna --


class InternalAck(BaseModel):
    success: bool = True
    message: str = "ok"


class IndexPushElement(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int | None = None
    text: str | None = None
    params: str | None = None
    description: str | None = None
    generated_description: str | None = Field(default=None, alias="generatedDescription")


class IndexPushRequest(BaseModel):
    """Cuerpo del push de fin de entrenamiento del backend (SearchIndexCommand, camelCase)."""

    model_config = ConfigDict(populate_by_name=True)

    list_id: int | None = Field(default=None, alias="listId")
    user_id: int = Field(alias="userId")
    list_name: str = Field(alias="listName")
    is_public: bool = Field(default=False, alias="isPublic")
    embeddings_data: str | None = Field(default=None, alias="embeddingsData")
    model: str | None = None
    trained_element_ids: list[int] | None = Field(default=None, alias="trainedElementIds")
    elements: list[IndexPushElement] = Field(default_factory=list)


class ListMetaUpdateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_id: int = Field(alias="userId")
    name: str
    is_public: bool = Field(default=False, alias="isPublic")


class ApiKeyUpsertRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    user_id: int = Field(alias="userId")
    api_key: str = Field(alias="apiKey")


class HealthResponse(BaseModel):
    status: str = "ok"
    ready: bool
    api_keys: int
    lists: int
    lists_cached: int
    cache_bytes: int
    logs_pending: int
