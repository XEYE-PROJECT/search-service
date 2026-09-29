"""Esquemas de petición/respuesta.

La API pública va en snake_case (el contrato histórico del frontend y la doc de clientes) y es
ESTRICTA: campos desconocidos → 422 (``extra="forbid"``) y los textos se recortan antes de
validar, así un término en blanco también da 422. Los cuerpos internos del backend llegan en
camelCase (defaults de Jackson) — aquí se mapean con alias; el backend serializa los null
explícitos, de ahí tanto Optional. Los internos NO son estrictos a propósito: el backend y este
servicio se despliegan por separado y un campo nuevo del backend no debe romper el push.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.security import hash_api_key

# --------------------------------------------------------------------------- pública --

_STRICT = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PublicSearchRequest(BaseModel):
    """``POST /api/v1/search``. Solo listas PÚBLICAS del dueño de la key: la visibilidad la
    decide el propietario en la consola, nunca el cliente (el antiguo ``allow_private`` ya no
    existe y, como cualquier campo desconocido, responde 422)."""

    model_config = _STRICT

    list_name: str = Field(min_length=1, max_length=100)
    search_term: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=50, ge=1, le=1000)
    session: str | None = Field(default=None, min_length=1, max_length=255)
    include_score_breakdown: bool = False
    #: OBSOLETO e ignorado: el registro de uso ya no es opcional (toda búsqueda pública se
    #: registra). Se sigue aceptando para no romper clientes que aún lo envían.
    register_log: bool = Field(default=True, deprecated=True)


class PublicTargetRequest(BaseModel):
    model_config = _STRICT

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
    #: True si la respuesta se sirvió sin toda la calidad posible; ``degradation_reasons``
    #: dice por qué (``no_embeddings``, ``model_unavailable``, ``model_mismatch``, ``stale_data``).
    degraded: bool = False
    degradation_reasons: list[str] = Field(default_factory=list)
    error: str | None = None


class TargetResponse(BaseModel):
    success: bool = True


# --------------------------------------------------------------------------- interna --


class InternalAck(BaseModel):
    success: bool = True
    message: str = "ok"


class ConsoleSearchRequest(BaseModel):
    """``POST /v1/lists/{id}/search``: la búsqueda del playground de la consola, que el backend
    reenvía tras comprobar que la lista es del usuario autenticado (por eso aquí no se mira
    la visibilidad). Sin API key de por medio, y sin log de búsqueda."""

    model_config = _STRICT

    search_term: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=50, ge=1, le=1000)
    include_score_breakdown: bool = False


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
    """Alta/actualización de una key: el backend manda su SHA-256 (``keyHash``), nunca el valor.

    ``apiKey`` (en claro) se acepta de forma transitoria para backends anteriores a las keys
    hasheadas; quitarlo cuando el backend con V6 lleve un tiempo en producción.
    """

    model_config = ConfigDict(populate_by_name=True)

    user_id: int = Field(alias="userId")
    key_hash: str | None = Field(default=None, alias="keyHash", pattern=r"^[0-9a-f]{64}$")
    api_key: str | None = Field(default=None, alias="apiKey", min_length=1)

    @model_validator(mode="after")
    def _require_a_key(self) -> ApiKeyUpsertRequest:
        if not self.key_hash and not self.api_key:
            raise ValueError("keyHash is required")
        return self

    def resolved_hash(self) -> str:
        return self.key_hash or hash_api_key(self.api_key or "")


class UserLimitsRequest(BaseModel):
    """``PUT /v1/users/{id}/limits``: búsquedas/minuto del usuario; ``null`` vuelve al por defecto."""

    model_config = ConfigDict(populate_by_name=True)

    rate_limit_per_minute: int | None = Field(default=None, alias="rateLimitPerMinute", ge=1, le=1_000_000)


class HealthResponse(BaseModel):
    status: str = "ok"
    ready: bool
    degraded: bool = False
    checks: dict[str, str] = Field(default_factory=dict)
    api_keys: int
    lists: int
    lists_cached: int
    cache_bytes: int
    logs_pending: int
    logs_spooled: int = 0
