"""Cliente HTTP de la API interna del backend (/internal/search/*).

Auth: el secreto compartido X-Internal-Token; X-Internal-Service nos identifica en sus
logs. Todos los payloads van en camelCase (los defaults de Jackson en el backend).
Las API keys llegan como hash SHA-256 (``keyHash``), nunca en claro.
"""

from __future__ import annotations

import logging

import httpx

from app.application.ports import BootstrapData, ListDataPayload, LogEntry
from app.core.security import hash_api_key
from app.domain.models import ListMeta

logger = logging.getLogger(__name__)


class BackendClient:
    def __init__(self, base_url: str, internal_token: str, timeout: float = 30.0) -> None:
        self._http = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers={
                "X-Internal-Service": "search-service",
                "X-Internal-Token": internal_token,
            },
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def fetch_bootstrap(self) -> BootstrapData:
        response = await self._http.get("/internal/search/bootstrap")
        response.raise_for_status()
        body = response.json()
        api_keys = [
            (entry["id"], entry["userId"], _key_hash(entry))
            for entry in body.get("apiKeys", [])
        ]
        lists = [
            ListMeta(
                id=entry["id"],
                user_id=entry["userId"],
                name=entry["name"],
                is_public=bool(entry.get("isPublic", False)),
            )
            for entry in body.get("lists", [])
        ]
        embedding_models = [
            name for name in body.get("embeddingModels") or [] if isinstance(name, str) and name.strip()
        ]
        return BootstrapData(api_keys=api_keys, lists=lists, embedding_models=embedding_models)

    async def fetch_list_data(self, list_id: int) -> ListDataPayload | None:
        response = await self._http.get(f"/internal/search/lists/{list_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        body = response.json()
        meta = ListMeta(
            id=body["id"],
            user_id=body["userId"],
            name=body["name"],
            is_public=bool(body.get("isPublic", False)),
        )
        return ListDataPayload(
            meta=meta,
            elements=body.get("elements") or [],
            embeddings_data=body.get("embeddingsData"),
            model=body.get("model"),
            trained_element_ids=body.get("trainedElementIds"),
        )

    async def push_logs(self, entries: list[LogEntry]) -> None:
        payload = {
            "logs": [
                {
                    "userId": entry.user_id,
                    "apiKeyId": entry.api_key_id,
                    "listId": entry.list_id,
                    "listName": entry.list_name,
                    "endpoint": entry.endpoint,
                    "searchTerm": entry.search_term,
                    "totalResults": entry.total_results,
                    "durationMs": entry.duration_ms,
                    "session": entry.session,
                    "results": entry.results,
                    "searchedAt": entry.searched_at,
                }
                for entry in entries
            ]
        }
        response = await self._http.post("/internal/search/logs", json=payload)
        response.raise_for_status()


def _key_hash(entry: dict) -> str:
    """``keyHash`` del backend actual. Transitorio: un backend anterior a las keys hasheadas
    aún manda ``apiKey`` en claro; se hashea aquí para que el orden de despliegue no importe.
    Quitar el fallback cuando el backend con V6 lleve un tiempo en producción."""
    key_hash = entry.get("keyHash")
    if key_hash:
        return str(key_hash)
    return hash_api_key(str(entry["apiKey"]))
