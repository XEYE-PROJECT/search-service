"""API pública (X-API-Key): los endpoints que llaman las integraciones de los usuarios.

Solo sirve listas PÚBLICAS del dueño de la key. Las privadas se buscan desde la consola, que
pasa por el backend y llega aquí por la API interna (``/v1/lists/{id}/search``). Toda
búsqueda pública se registra (el ``register_log`` del cliente es obsoleto y se ignora).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from app.domain.models import ApiKeyInfo
from app.infrastructure.web.deps import require_api_key
from app.infrastructure.web.schemas import (
    PublicSearchRequest,
    PublicTargetRequest,
    SearchResponse,
    TargetResponse,
)

router = APIRouter(prefix="/api/v1", tags=["public"])

DEGRADED_HEADER = "X-Search-Degraded"


def mark_degraded(response: Response, result: dict) -> None:
    """Cabecera ``X-Search-Degraded: true`` además del campo del cuerpo (para proxies y logs)."""
    if result.get("degraded"):
        response.headers[DEGRADED_HEADER] = "true"


@router.post("/search", response_model=SearchResponse, response_model_exclude_none=True)
async def search(
    request: Request,
    response: Response,
    body: PublicSearchRequest,
    auth: Annotated[ApiKeyInfo, Depends(require_api_key)],
):
    search_use_case = request.app.state.container.search
    meta = await search_use_case.resolve_public_list(auth.user_id, body.list_name)
    result = await search_use_case.search(
        meta,
        auth,
        body.search_term,
        body.limit,
        include_breakdown=body.include_score_breakdown,
        session=body.session,
        register_log=True,
    )
    mark_degraded(response, result)
    return result


@router.post("/target", response_model=TargetResponse)
async def target(
    request: Request,
    body: PublicTargetRequest,
    auth: Annotated[ApiKeyInfo, Depends(require_api_key)],
):
    search_use_case = request.app.state.container.search
    meta = await search_use_case.resolve_public_list(auth.user_id, body.list_name)
    return await search_use_case.target(meta, auth, body.target_term, body.session)
