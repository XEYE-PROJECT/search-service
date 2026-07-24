"""API pública (X-API-Key): los endpoints que llaman directamente clientes y frontend."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.domain.models import ApiKeyInfo
from app.infrastructure.web.deps import require_api_key
from app.infrastructure.web.schemas import (
    PublicSearchRequest,
    PublicTargetRequest,
    SearchResponse,
    TargetResponse,
)

router = APIRouter(prefix="/api/v1", tags=["public"])


@router.post("/search", response_model=SearchResponse, response_model_exclude_none=True)
async def search(
    request: Request,
    body: PublicSearchRequest,
    auth: Annotated[ApiKeyInfo, Depends(require_api_key)],
):
    search_use_case = request.app.state.container.search
    meta = await search_use_case.resolve_public_list(auth.user_id, body.list_name)
    return await search_use_case.search(
        meta,
        auth,
        body.search_term,
        body.limit,
        include_breakdown=body.include_score_breakdown,
        session=body.session,
    )


@router.post("/target", response_model=TargetResponse)
async def target(
    request: Request,
    body: PublicTargetRequest,
    auth: Annotated[ApiKeyInfo, Depends(require_api_key)],
):
    search_use_case = request.app.state.container.search
    meta = await search_use_case.resolve_public_list(auth.user_id, body.list_name)
    return await search_use_case.target(meta, auth, body.target_term, body.session)
