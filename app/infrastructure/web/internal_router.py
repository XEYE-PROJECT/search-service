"""API interna (/v1, X-Internal-Token): solo la llama el backend.

El push de índice es el contrato ya existente del backend (POST /v1/lists/{id}/index al
completar un entrenamiento); el resto son las notificaciones de cambio y endpoints operativos.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request, Response

from app.application.ports import ListDataPayload
from app.domain.models import ListMeta
from app.infrastructure.web.deps import enforce_user_limit, require_internal_token, require_ready
from app.infrastructure.web.public_router import mark_degraded
from app.infrastructure.web.schemas import (
    ApiKeyUpsertRequest,
    ConsoleSearchRequest,
    HealthResponse,
    IndexPushRequest,
    InternalAck,
    ListMetaUpdateRequest,
    SearchResponse,
    UserLimitsRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["internal"], dependencies=[Depends(require_internal_token)])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    container = request.app.state.container
    status = container.health()
    return HealthResponse(
        ready=status["ready"],
        degraded=status["degraded"],
        checks=status["checks"],
        api_keys=len(container.api_keys),
        lists=len(container.lists),
        lists_cached=len(container.cache),
        cache_bytes=container.cache.total_bytes,
        logs_pending=container.log_queue.pending,
        logs_spooled=container.log_queue.spooled,
    )


@router.post("/lists/{list_id}/search", response_model=SearchResponse, response_model_exclude_none=True)
async def console_search(request: Request, response: Response, list_id: int, body: ConsoleSearchRequest):
    """Búsqueda del playground de la consola, reenviada por el backend (que ya comprobó que la
    lista pertenece al usuario autenticado): sirve también listas privadas, sin API key y sin
    log de búsqueda. Cuenta contra el cupo por usuario igual que la API pública."""
    container = request.app.state.container
    await require_ready(request)
    meta = await container.search.resolve_list_by_id(list_id)
    enforce_user_limit(request, response, meta.user_id)
    result = await container.search.search(
        meta,
        None,
        body.search_term,
        body.limit,
        include_breakdown=body.include_score_breakdown,
        register_log=False,
    )
    mark_degraded(response, result)
    return result


@router.post("/lists/{list_id}/index", response_model=InternalAck)
async def index_list(request: Request, list_id: int, body: IndexPushRequest):
    """Entrenamiento completado: reemplazo total de los datos de búsqueda de la lista.

    Debe responder rápido y nunca dar 5xx por embeddings malos — la construcción degrada
    a solo texto. Idempotente: empujar dos veces el mismo payload es inocuo.
    """
    container = request.app.state.container
    payload = ListDataPayload(
        meta=ListMeta(id=list_id, user_id=body.user_id, name=body.list_name, is_public=body.is_public),
        elements=[
            {"id": e.id, "text": e.text or "", "params": e.params, "description": e.description} for e in body.elements
        ],
        embeddings_data=body.embeddings_data,
        model=body.model,
        trained_element_ids=body.trained_element_ids,
    )
    await container.list_data.apply_push(payload)
    return InternalAck(message=f"list {list_id} indexed ({len(body.elements)} elements)")


@router.put("/lists/{list_id}/meta", response_model=InternalAck)
async def update_list_meta(request: Request, list_id: int, body: ListMetaUpdateRequest):
    """Renombrado / cambio de visibilidad: actualiza el catálogo; la caché sigue valiendo."""
    container = request.app.state.container
    container.lists.upsert(ListMeta(id=list_id, user_id=body.user_id, name=body.name, is_public=body.is_public))
    return InternalAck(message=f"list {list_id} meta updated")


@router.delete("/lists/{list_id}", response_model=InternalAck)
async def delete_list(request: Request, list_id: int):
    request.app.state.container.list_data.remove_list(list_id)
    return InternalAck(message=f"list {list_id} removed")


@router.post("/lists/{list_id}/invalidate", response_model=InternalAck)
async def invalidate_list(request: Request, list_id: int):
    """Cambiaron elementos: descarta la caché; la próxima búsqueda recarga perezosamente."""
    request.app.state.container.list_data.invalidate(list_id)
    return InternalAck(message=f"list {list_id} invalidated")


@router.put("/api-keys/{api_key_id}", response_model=InternalAck)
async def upsert_api_key(request: Request, api_key_id: int, body: ApiKeyUpsertRequest):
    request.app.state.container.api_keys.upsert(api_key_id, body.user_id, body.resolved_hash())
    return InternalAck(message=f"api key {api_key_id} upserted")


@router.delete("/api-keys/{api_key_id}", response_model=InternalAck)
async def delete_api_key(request: Request, api_key_id: int):
    request.app.state.container.api_keys.remove(api_key_id)
    return InternalAck(message=f"api key {api_key_id} removed")


@router.put("/users/{user_id}/limits", response_model=InternalAck)
async def update_user_limits(request: Request, user_id: int, body: UserLimitsRequest):
    """Un admin cambió las búsquedas/minuto del usuario (``null`` = volver al por defecto)."""
    request.app.state.container.user_limits.set(user_id, body.rate_limit_per_minute)
    return InternalAck(message=f"user {user_id} limits updated")


@router.delete("/users/{user_id}", response_model=InternalAck)
async def delete_user(request: Request, user_id: int):
    container = request.app.state.container
    container.api_keys.remove_user(user_id)
    container.user_limits.remove_user(user_id)
    container.list_data.remove_user(user_id)
    return InternalAck(message=f"user {user_id} removed")


@router.post("/refresh", response_model=InternalAck)
async def refresh(request: Request):
    """Operativo: fuerza un re-sync completo del catálogo desde el backend (y descarta de la
    caché las listas que ya no existen)."""
    container = request.app.state.container
    await container.catalog_service.refresh()
    dropped = container.list_data.reconcile()
    return InternalAck(message=f"catalog refreshed ({dropped} stale cached lists dropped)")


@router.post("/logs/replay", response_model=InternalAck)
async def replay_logs(request: Request):
    """Operativo: reenvía ahora los logs de búsqueda spooleados en disco."""
    sent = await request.app.state.container.log_queue.replay_once()
    return InternalAck(message=f"{sent} spooled search-log entries replayed")
